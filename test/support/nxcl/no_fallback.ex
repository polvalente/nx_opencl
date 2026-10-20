defmodule NxCL.TestSupport.NoFallback do
  @moduledoc """
  A backend that raises on every compute callback, to show that a program runs on
  the device and not on the host CPU.

  `guard/1` runs a function with this backend as the default. Code on the compute
  path that builds an intermediate with the default backend raises `Fallback`.
  Only data movement works: `from_binary`, `to_binary`, `constant`, `inspect`, the
  copy and free callbacks, and `reshape` and `squeeze`, which change only the shape
  of the host copy of a result.

  A host call may compute on the host: a compute callback that runs inside
  `NxCL.Compiler.Executor.Host` goes to `Nx.BinaryBackend` instead of raising.

  `guard/1` also traces calls into `Nx.BinaryBackend` by the running process (an
  op that names `Nx.BinaryBackend` itself). Such a call is a fallback, also after
  a host call: the trace goes silent while a callback runs, so every call it sees
  was made outside a callback.
  """

  @behaviour Nx.Backend

  defstruct [:binary]

  defmodule Fallback do
    @moduledoc "Raised when a compute callback runs on the no-fallback backend."
    defexception [:message]
  end

  alias NxCL.TestSupport.Collector

  @installed {__MODULE__, :installed}

  # Callbacks that move data or change a shape.
  @data [
    :init,
    :from_binary,
    :to_binary,
    :inspect,
    :backend_copy,
    :backend_transfer,
    :backend_deallocate,
    :constant,
    :reshape,
    :squeeze
  ]

  @doc """
  Runs `fun` with this backend as the default and returns its result. Raises
  `Fallback` when `Nx.BinaryBackend` computed during `fun` outside a host call callback.
  """
  @spec guard((-> result)) :: result when result: var
  def guard(fun) do
    {result, %{binary: seen_calls}} =
      trace(fn -> Nx.with_default_backend(__MODULE__, fun) end)

    binary_calls = compute_calls(seen_calls)

    if binary_calls != [] do
      raise Fallback,
            "Nx.BinaryBackend computed outside a host call: " <>
              inspect(Enum.uniq(binary_calls))
    end

    result
  end

  @doc """
  Runs `fun` and returns `{result, seen}`. `seen.binary` lists each distinct
  `Nx.BinaryBackend` compute callback (as `{name, arity, caller}`) that the calling
  process ran meanwhile; data movement callbacks are not listed.
  `seen.host_calls` counts the host steps that ran a function; an `:io_call` with no
  function is not one.
  """
  @spec trace((-> result)) ::
          {result, %{binary: [tuple()], host_calls: non_neg_integer()}}
        when result: var
  def trace(fun) do
    collector = begin_trace()

    try do
      result = fun.()
      {result, end_trace(collector)}
    after
      # A raise in `fun` leaves the trace on; stop it and the collector.
      Collector.discard(collector)
    end
  end

  @doc """
  Drops the calls `Nx.Defn.Expr` makes while it traces: it folds ops on scalar
  constants on `Nx.BinaryBackend`. That is Nx's work, not the compute path.
  """
  @spec compute_calls([tuple()]) :: [tuple()]
  def compute_calls(calls), do: Enum.reject(calls, &match?({_, _, {Nx.Defn.Expr, _, _}}, &1))

  @doc "Starts tracing the calling process; give the result to `end_trace/2`."
  @spec begin_trace() :: pid()
  def begin_trace do
    Code.ensure_loaded!(Nx.BinaryBackend)
    install_patterns()

    collector =
      Collector.start(%{binary: MapSet.new(), host_calls: 0}, &collect/2, fn seen ->
        %{seen | binary: Enum.sort(seen.binary)}
      end)

    :erlang.trace(self(), false, [:silent])
    Collector.trace_self(collector)
    collector
  end

  # Trace patterns are global and slow to set: once per VM.
  defp install_patterns do
    if :persistent_term.get(@installed, nil) == nil do
      Code.ensure_loaded!(NxCL.Compiler.Executor)
      Code.ensure_loaded!(NxCL.Compiler.Executor.Host)
      :erlang.system_flag(:backtrace_depth, 100)

      for {name, arity} <- Nx.Backend.behaviour_info(:callbacks),
          name not in @data,
          function_exported?(Nx.BinaryBackend, name, arity) do
        :erlang.trace_pattern(
          {Nx.BinaryBackend, name, arity},
          [{:_, [], [{:message, {:caller}}]}],
          [:local]
        )
      end

      # A host call computes on `Nx.BinaryBackend` and can make millions of calls.
      # Note it at `run/3`, go silent after `read/2` (the callback follows), and
      # resume at `put_results/3`.
      :erlang.trace_pattern(
        {NxCL.Compiler.Executor.Host, :run, 3},
        [{:_, [], [{:message, {:caller}}]}],
        [:local]
      )

      :erlang.trace_pattern(
        {NxCL.Compiler.Executor.Host, :read, 2},
        [{:_, [], [{:silent, true}]}],
        [
          :local
        ]
      )

      :erlang.trace_pattern(
        {NxCL.Compiler.Executor, :put_results, 3},
        [{:_, [], [{:silent, false}]}],
        [:local]
      )

      :persistent_term.put(@installed, :installed)
    end
  end

  @doc """
  Stops tracing `pid` (default: the calling process) and returns what was seen since
  `begin_trace/0` (see `trace/1`).
  """
  @spec end_trace(pid(), pid()) :: %{binary: [tuple()], host_calls: non_neg_integer()}
  def end_trace(collector, pid \\ self()), do: Collector.stop(collector, pid)

  defp ran_function?(%{kind: :io_call, fun: nil}), do: false
  defp ran_function?(_entry), do: true

  defp collect({:trace, _pid, :call, {Nx.BinaryBackend, name, args}, caller}, seen),
    do: Map.update!(seen, :binary, &MapSet.put(&1, {name, length(args), caller}))

  defp collect(
         {:trace, _pid, :call, {NxCL.Compiler.Executor.Host, :run, [entry, _ctx, _buffers]},
          _caller},
         seen
       ),
       do: if(ran_function?(entry), do: Map.update!(seen, :host_calls, &(&1 + 1)), else: seen)

  # `NxCL.TestSupport.Trace` sets patterns on other executor functions.
  defp collect(_message, seen), do: seen

  @impl true
  def init(opts), do: opts

  @impl true
  def from_binary(tensor, binary, _opts), do: put_in(tensor.data, %__MODULE__{binary: binary})

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{binary: binary}, type: {_, bits}} = tensor, limit) do
    binary_part(binary, 0, div(bits * min(limit, Nx.size(tensor)), 8))
  end

  @impl true
  def inspect(%Nx.Tensor{data: %__MODULE__{binary: binary}} = tensor, opts) do
    Nx.Backend.inspect(tensor, binary, opts)
  end

  @impl true
  def backend_copy(%Nx.Tensor{} = tensor, backend, opts) do
    binary = to_binary(tensor, Nx.size(tensor))
    backend.from_binary(%{tensor | data: %{__struct__: backend}}, binary, opts)
  end

  @impl true
  def backend_transfer(tensor, backend, opts), do: backend_copy(tensor, backend, opts)

  @impl true
  def backend_deallocate(_tensor), do: :ok

  @impl true
  def constant(out, number, opts) do
    %Nx.Tensor{data: %Nx.BinaryBackend{state: binary}} =
      Nx.BinaryBackend.constant(out, number, opts)

    put_in(out.data, %__MODULE__{binary: binary})
  end

  @impl true
  def reshape(out, %Nx.Tensor{data: %__MODULE__{} = data}), do: put_in(out.data, data)

  @impl true
  def squeeze(out, %Nx.Tensor{data: %__MODULE__{} = data}, _axes), do: put_in(out.data, data)

  for {name, arity} <- Nx.Backend.behaviour_info(:callbacks), name not in @data do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(name)(unquote_splicing(args)) do
      if in_host_call?() do
        apply(Nx.BinaryBackend, unquote(name), Enum.map(unquote(args), &to_binary_data/1))
      else
        raise Fallback,
              "#{unquote(name)}/#{unquote(arity)} ran on the no-fallback backend: " <>
                "an intermediate fell back to the default backend instead of the device"
      end
    end
  end

  defp in_host_call? do
    {:current_stacktrace, frames} = Process.info(self(), :current_stacktrace)
    Enum.any?(frames, &match?({NxCL.Compiler.Executor.Host, _, _, _}, &1))
  end

  defp to_binary_data(%Nx.Tensor{data: %__MODULE__{binary: binary}} = tensor),
    do: %{tensor | data: %Nx.BinaryBackend{state: binary}}

  defp to_binary_data(list) when is_list(list), do: Enum.map(list, &to_binary_data/1)
  defp to_binary_data(other), do: other
end
