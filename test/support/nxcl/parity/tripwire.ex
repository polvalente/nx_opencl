defmodule NxCL.Parity.Tripwire do
  @moduledoc """
  The no-fallback tripwire: an `Nx.Backend` that raises on every compute callback.

  Installed as the default backend while `NxCL.Compiler` or `NxCL.Backend` runs a
  row (`guard/2`), so any code on the compute path that builds an intermediate
  with the default backend raises `NxCL.Parity.Tripwire.Fallback` instead of
  computing on the CPU. Only data movement works: `from_binary`, `to_binary`,
  `constant`, `inspect`, the copy and free callbacks, and `reshape` and `squeeze`, which change
  only the shape of the host copy of a result (the executor reshapes what it reads
  back from the device).

  The one allowed crossing is a host call. A callback that `Nx.runtime_call/4` or
  `Nx.io_call/2` runs computes on the host by design, so a compute callback that
  runs inside `NxCL.Executor.Host` goes to `Nx.BinaryBackend` instead of raising.
  A block host function (`qr`, `eigh`, `svd`, `lu`) runs under `Nx.BinaryBackend`
  on purpose (`NxCL.Lower.Block`) and never reaches this backend.

  `guard/2` also traces calls into `Nx.BinaryBackend` made by the running process
  (an op that names `Nx.BinaryBackend` itself, bypassing the default). Such a
  call is a fallback unless the row made a host call.

  Host calls are counted by kind: the block that ran on the host
  (`:block_lin_alg_qr`, `:block_lin_alg_eigh`, `:block_lin_alg_svd`,
  `:block_lin_alg_lu`, read from `NxCL.Lower.Block`) or `:callback` for a
  callback the user passed to `Nx.runtime_call/4`, `Nx.io_call/2` or a hook. An
  op built on a host block, such as `solve` on `lu`, counts under the block.
  """

  @behaviour Nx.Backend

  defstruct [:binary]

  defmodule Fallback do
    @moduledoc "Raised when a compute callback runs on the tripwire backend."
    defexception [:message]
  end

  @table __MODULE__
  @outer {__MODULE__, :outer}
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

  @doc "Creates the table that holds host call counts per row label."
  @spec start() :: :ok
  def start do
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :public, :set])
    end

    :ok
  end

  @doc """
  Runs `fun` with this backend as the default and returns its result. With
  `default: false` the default backend stays as it is: the eager run needs it,
  because the evaluator builds the output of `iota` and the like on it. Host calls
  the executor ran meanwhile are added to the count of their kind (see the module
  doc). Raises `Fallback` when `Nx.BinaryBackend` computed during `fun` and no host
  call ran.
  """
  @spec guard(term(), (-> result), keyword()) :: result when result: var
  def guard(label, fun, opts \\ []) do
    start()
    outer = Nx.default_backend()

    {result, %{binary: seen_calls, blocks: blocks, host_calls: calls}} =
      trace(fn ->
        Process.put(@outer, outer)

        try do
          if Keyword.get(opts, :default, true),
            do: Nx.with_default_backend(__MODULE__, fun),
            else: fun.()
        after
          Process.delete(@outer)
        end
      end)

    binary_calls = compute_calls(seen_calls)

    for struct <- blocks do
      kind = kind(struct)
      :ets.update_counter(@table, kind, 1, {kind, 0})
    end

    if binary_calls != [] and calls == 0 do
      raise Fallback,
            "#{inspect(label)}: Nx.BinaryBackend computed with no host call: " <>
              inspect(Enum.uniq(binary_calls))
    end

    result
  end

  # `Nx.Block.LinAlg.LU` -> `:block_lin_alg_lu`, as the matrix names the op.
  defp kind(:callback), do: :callback

  defp kind(struct) do
    "Elixir.Nx.Block." <> rest = Atom.to_string(struct)
    String.to_atom("block_" <> Enum.map_join(String.split(rest, "."), "_", &Macro.underscore/1))
  end

  @doc """
  Runs `fun` and returns `{result, seen}`. `seen.binary` lists each distinct
  `Nx.BinaryBackend` compute callbacks (as `{name, arity, caller}`) that the calling
  process ran meanwhile; data movement callbacks are not listed. `seen.blocks` lists
  what ran as a host call: a block struct, or `:callback` for a user callback.
  `seen.host_calls` counts the host steps that ran a function; an `:io_call` with no
  function is not one.
  """
  @spec trace((-> result)) ::
          {result,
           %{
             binary: [tuple()],
             blocks: [module() | :callback],
             host_calls: non_neg_integer()
           }}
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

  @doc """
  Runs `fun` with the default backend and the trace that were in effect before
  `guard/2`. `NxCL.Parity.Tripwire.Compiler` wraps the user function with it: the
  function builds the expression, and any concrete tensor it computes on is the
  row's own literal, not the compute path.
  """
  @spec paused((-> result)) :: result when result: var
  def paused(fun) do
    case {Process.get(@outer), Process.get(@collector)} do
      {nil, _} ->
        fun.()

      {outer, collector} ->
        :erlang.trace(self(), false, [:call])

        try do
          Nx.with_default_backend(outer, fun)
        after
          :erlang.trace(self(), true, [:call, {:tracer, collector}])
        end
    end
  end

  @doc "Starts tracing the calling process; give the result to `end_trace/2`."
  @spec begin_trace() :: pid()
  def begin_trace do
    start()
    Code.ensure_loaded!(Nx.BinaryBackend)
    install_patterns()
    collector = spawn(fn -> collect(%{binary: MapSet.new(), blocks: [], host_calls: 0}) end)

    :erlang.trace(self(), false, [:silent])
    :erlang.trace(self(), true, [:call, {:tracer, collector}])
    Process.put(@collector, collector)
    collector
  end

  # Trace patterns are global and slow to set, so set them once per VM.
  defp install_patterns do
    if :persistent_term.get(@collector, nil) == nil do
      Code.ensure_loaded!(NxCL.Executor)
      Code.ensure_loaded!(NxCL.Executor.Host)
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
        {NxCL.Executor.Host, :run, 3},
        [{:_, [], [{:message, {:caller}}]}],
        [:local]
      )

      :erlang.trace_pattern({NxCL.Executor.Host, :read, 2}, [{:_, [], [{:silent, true}]}], [
        :local
      ])

      :erlang.trace_pattern(
        {NxCL.Executor, :put_results, 3},
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
  @spec end_trace(pid(), pid()) :: %{
          binary: [tuple()],
          blocks: [module() | :callback],
          host_calls: non_neg_integer()
        }
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

  # The block struct a host entry runs, or `:callback` for a user callback. The closure
  # `NxCL.Lower.Block.host_entry/4` builds holds the block struct in its environment.
  defp block_of(%{fun: fun}) when is_function(fun) do
    with {:module, NxCL.Lower.Block} <- Function.info(fun, :module),
         {:env, env} <- Function.info(fun, :env),
         %{__struct__: struct} <- Enum.find(env, &match?(%{__struct__: _}, &1)) do
      struct
    else
      _ -> :callback
    end
  end

  defp block_of(_entry), do: :callback

  # `Host.run/3` returns an `:io_call` that has no function at once; nothing runs.
  defp ran_function?(%{kind: :io_call, fun: nil}), do: false
  defp ran_function?(_entry), do: true

  defp collect(seen) do
    receive do
      {:trace, _pid, :call, {Nx.BinaryBackend, name, args}, caller} ->
        collect(Map.update!(seen, :binary, &MapSet.put(&1, {name, length(args), caller})))

      {:trace, _pid, :call, {NxCL.Executor.Host, :run, [entry, _ctx, _buffers]}, _caller} ->
        seen = Map.update!(seen, :blocks, &[block_of(entry) | &1])

        collect(
          if ran_function?(entry), do: Map.update!(seen, :host_calls, &(&1 + 1)), else: seen
        )

      # `NxCL.TestSupport.Trace` sets patterns on other executor functions.
      {:trace, _pid, :call, _mfa, _caller} ->
        collect(seen)

      {:done, from, ref} ->
        seen = %{seen | binary: Enum.sort(seen.binary), blocks: Enum.reverse(seen.blocks)}
        send(from, {ref, seen})
    end
  end

  @doc "Host call counts per row label, since the last `reset/0`."
  @spec host_calls() :: %{term() => pos_integer()}
  def host_calls do
    start()
    Map.new(:ets.tab2list(@table))
  end

  @doc "Clears the counts."
  @spec reset() :: :ok
  def reset do
    start()
    :ets.delete_all_objects(@table)
    :ok
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
              "#{unquote(name)}/#{unquote(arity)} ran on the tripwire backend: " <>
                "an intermediate fell back to the default backend instead of the device"
      end
    end
  end

  # A callback of `Nx.runtime_call/4` or `Nx.io_call/2` runs in `NxCL.Executor.Host`.
  defp in_host_call? do
    {:current_stacktrace, frames} = Process.info(self(), :current_stacktrace)
    Enum.any?(frames, &match?({NxCL.Executor.Host, _, _, _}, &1))
  end

  defp to_binary_data(%Nx.Tensor{data: %__MODULE__{binary: binary}} = tensor),
    do: %{tensor | data: %Nx.BinaryBackend{state: binary}}

  defp to_binary_data(list) when is_list(list), do: Enum.map(list, &to_binary_data/1)
  defp to_binary_data(other), do: other
end

defmodule NxCL.Parity.Tripwire.Compiler do
  @moduledoc """
  Runs another compiler (`inner:`) with the tracing of `NxCL.Parity.Tripwire`
  paused while the user function builds its expression.
  """
  @behaviour Nx.Defn.Compiler

  alias NxCL.Parity.Tripwire

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    {inner, opts} = Keyword.pop!(opts, :inner)
    inner.__jit__(key, vars, &Tripwire.paused(fn -> fun.(&1) end), args_list, opts)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    {inner, opts} = Keyword.pop!(opts, :inner)
    inner.__compile__(key, vars, &Tripwire.paused(fn -> fun.(&1) end), opts)
  end

  @impl true
  def __partitions_options__(opts) do
    {inner, opts} = Keyword.pop!(opts, :inner)
    inner.__partitions_options__(opts)
  end

  @impl true
  def __to_backend__(opts) do
    {inner, opts} = Keyword.pop!(opts, :inner)
    inner.__to_backend__(opts)
  end
end
