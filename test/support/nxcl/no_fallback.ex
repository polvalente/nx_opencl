defmodule NxCL.TestSupport.NoFallback do
  @moduledoc """
  A backend that raises on every compute callback, to show that a program runs on
  the device and not on the host CPU.

  `guard/1` runs a function with this backend as the default, so code on the
  compute path that builds an intermediate with the default backend raises
  `NxCL.TestSupport.NoFallback.Fallback` instead of computing on the CPU. Only
  data movement works: `from_binary`, `to_binary`, `constant`, `inspect`, the copy
  and free callbacks, and `reshape` and `squeeze`, which change only the shape of
  the host copy of a result (the executor reshapes what it reads back from the
  device).

  The one allowed crossing is a host call. A callback that `Nx.runtime_call/4` or
  `Nx.io_call/2` runs computes on the host by design, so a compute callback that
  runs inside `NxCL.Compiler.Executor.Host` goes to `Nx.BinaryBackend` instead of raising.

  `guard/1` also traces calls into `Nx.BinaryBackend` made by the running process
  (an op that names `Nx.BinaryBackend` itself, bypassing the default). Such a
  call is a fallback unless the function made a host call.
  """

  @behaviour Nx.Backend

  defstruct [:binary]

  defmodule Fallback do
    @moduledoc "Raised when a compute callback runs on the no-fallback backend."
    defexception [:message]
  end

  @collector {__MODULE__, :collector}

  # Callbacks that move data or change a shape; every other callback computes.
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
  `Fallback` when `Nx.BinaryBackend` computed during `fun` and no host call ran.
  """
  @spec guard((-> result)) :: result when result: var
  def guard(fun) do
    {result, %{binary: seen_calls, host_calls: calls}} =
      trace(fn -> Nx.with_default_backend(__MODULE__, fun) end)

    binary_calls = compute_calls(seen_calls)

    if binary_calls != [] and calls == 0 do
      raise Fallback,
            "Nx.BinaryBackend computed with no host call: " <> inspect(Enum.uniq(binary_calls))
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
      if Process.get(@collector), do: end_trace(collector)
    end
  end

  @doc """
  Drops the calls `Nx.Defn.Expr` makes while it traces: it folds an op on scalar
  constants at once, on `Nx.BinaryBackend`. That is Nx's own work before any
  compiler runs, not the compute path.
  """
  @spec compute_calls([tuple()]) :: [tuple()]
  def compute_calls(calls), do: Enum.reject(calls, &match?({_, _, {Nx.Defn.Expr, _, _}}, &1))

  @doc "Starts tracing the calling process; give the result to `end_trace/2`."
  @spec begin_trace() :: pid()
  def begin_trace do
    Code.ensure_loaded!(Nx.BinaryBackend)
    install_patterns()
    collector = spawn(fn -> collect(%{binary: MapSet.new(), host_calls: 0}) end)

    :erlang.trace(self(), false, [:silent])
    :erlang.trace(self(), true, [:call, {:tracer, collector}])
    Process.put(@collector, collector)
    collector
  end

  # Trace patterns are global and slow to set, so set them once per VM.
  defp install_patterns do
    if :persistent_term.get(@collector, nil) == nil do
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

      # A host call computes on `Nx.BinaryBackend` by design and can make millions of
      # calls. Note it at `run/3`, then go silent once `read/2` has run (the callback
      # follows) until `put_results/3`, which the executor calls next.
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

      :persistent_term.put(@collector, :installed)
    end
  end

  @doc """
  Stops tracing `pid` (default: the calling process) and returns what was seen since
  `begin_trace/0` (see `trace/1`).
  """
  @spec end_trace(pid(), pid()) :: %{binary: [tuple()], host_calls: non_neg_integer()}
  def end_trace(collector, pid \\ self()) do
    # A process that has exited has no trace left to stop.
    try do
      :erlang.trace(pid, false, [:call, :silent])
    rescue
      ArgumentError -> :ok
    end

    Process.delete(@collector)
    ref = make_ref()
    send(collector, {:done, self(), ref})

    receive do
      {^ref, calls} -> calls
    after
      5000 -> raise "trace collector did not answer"
    end
  end

  # `Host.run/3` returns an `:io_call` that has no function at once; nothing runs.
  defp ran_function?(%{kind: :io_call, fun: nil}), do: false
  defp ran_function?(_entry), do: true

  defp collect(seen) do
    receive do
      {:trace, _pid, :call, {Nx.BinaryBackend, name, args}, caller} ->
        collect(Map.update!(seen, :binary, &MapSet.put(&1, {name, length(args), caller})))

      {:trace, _pid, :call, {NxCL.Compiler.Executor.Host, :run, [entry, _ctx, _buffers]}, _caller} ->
        collect(
          if ran_function?(entry), do: Map.update!(seen, :host_calls, &(&1 + 1)), else: seen
        )

      # `NxCL.TestSupport.Trace` sets patterns on other executor functions.
      {:trace, _pid, :call, _mfa, _caller} ->
        collect(seen)

      {:done, from, ref} ->
        send(from, {ref, %{seen | binary: Enum.sort(seen.binary)}})
    end
  end

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

  # A scalar built from a number (`Nx.tensor/2` of a number); no tensor is computed.
  @impl true
  def constant(out, number, opts) do
    %Nx.Tensor{data: %Nx.BinaryBackend{state: binary}} =
      Nx.BinaryBackend.constant(out, number, opts)

    put_in(out.data, %__MODULE__{binary: binary})
  end

  # Shape-only: no element is read or computed.
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

  # A callback of `Nx.runtime_call/4` or `Nx.io_call/2` runs in `NxCL.Compiler.Executor.Host`.
  defp in_host_call? do
    {:current_stacktrace, frames} = Process.info(self(), :current_stacktrace)
    Enum.any?(frames, &match?({NxCL.Compiler.Executor.Host, _, _, _}, &1))
  end

  defp to_binary_data(%Nx.Tensor{data: %__MODULE__{binary: binary}} = tensor),
    do: %{tensor | data: %Nx.BinaryBackend{state: binary}}

  defp to_binary_data(list) when is_list(list), do: Enum.map(list, &to_binary_data/1)
  defp to_binary_data(other), do: other
end
