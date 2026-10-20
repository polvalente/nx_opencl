defmodule NxCL.TestSupport.Trace do
  @moduledoc """
  Counts the executor's control steps with Erlang call tracing, so `lib/` has
  no counters. `run/1` runs a function in the calling process and returns
  `{result, counts}`:

    * `:predicate_reads` - calls of `NxCL.Compiler.Executor.truthy?/3`, each a
      scalar predicate read back (a `cond` clause or a dynamic `while` condition).
    * `:segment_runs` - calls of `NxCL.Compiler.Executor.eval!/3`, each a
      straight-line segment run natively.
    * `:scope_runs` - calls of `NxCL.Compiler.Executor.run_sub/3`: every `while`
      body run, predicate scope and `cond` branch. A dynamic loop's body runs are
      `scope_runs - predicate_reads`.
    * `:host_calls` - calls of `NxCL.Compiler.Executor.Host.run/3` that run a
      function (an `:io_call` with no function runs nothing).
    * `:callback_inputs`, `:callback_input_bytes` - the tensors a callback takes
      (one per leaf of `in_template`) and their bytes.
    * `:callback_outputs` - the tensors a `:runtime_call` returns (one per leaf of
      its output template).

  The three callback counts come from the host-call entry, not from observed
  reads and uploads. Tracing `NxCL.Native.buffer_read/2` fails next to
  `NxCL.TestSupport.NoFallback`: its pattern on `Host.read/2` silences the
  traced process.

  `run/1` raises if a trace pattern matches no function, so a rename of a traced
  private function fails the test instead of leaving a count at zero. A test that
  expects zero should also assert a count that must be above zero (for example
  `scope_runs`). The executor must run in the caller, and the test must be
  `async: false`. Do not nest `run/1` or use it inside
  `NxCL.TestSupport.NoFallback.trace/1`: a process has one tracer.
  """

  @empty %{
    predicate_reads: 0,
    segment_runs: 0,
    scope_runs: 0,
    host_calls: 0,
    callback_inputs: 0,
    callback_input_bytes: 0,
    callback_outputs: 0
  }

  @doc "Runs `fun` and returns `{result, counts}`."
  @spec run((-> result)) :: {result, %{atom() => non_neg_integer()}} when result: var
  def run(fun) do
    install_patterns()
    tracer = spawn(fn -> collect([]) end)
    :erlang.trace(self(), true, [:call, {:tracer, tracer}])

    try do
      result = fun.()
      {result, stop(tracer)}
    after
      :erlang.trace(self(), false, [:call])
    end
  end

  # A process does not trace itself, so a second process collects the messages.
  defp collect(messages) do
    receive do
      {:trace, _pid, :call, {mod, _fun, _args}, _caller} = message
      when mod in [NxCL.Compiler.Executor, NxCL.Compiler.Executor.Host] ->
        collect([message | messages])

      # Other patterns (`NxCL.TestSupport.NoFallback` sets some) can fire; skip them.
      {:trace, _pid, _kind, _mfa, _caller} ->
        collect(messages)

      {:done, from, ref} ->
        send(from, {ref, Enum.reverse(messages)})
    end
  end

  defp stop(tracer) do
    :erlang.trace(self(), false, [:call])
    ref = make_ref()
    send(tracer, {:done, self(), ref})

    receive do
      {^ref, messages} -> Enum.reduce(messages, @empty, &count(&2, &1))
    after
      5000 -> raise "trace collector did not answer"
    end
  end

  defp count(counts, {:trace, _pid, :call, {NxCL.Compiler.Executor, :truthy?, _args}, _caller}),
    do: bump(counts, :predicate_reads)

  defp count(counts, {:trace, _pid, :call, {NxCL.Compiler.Executor, :eval!, _args}, _caller}),
    do: bump(counts, :segment_runs)

  defp count(counts, {:trace, _pid, :call, {NxCL.Compiler.Executor, :run_sub, _args}, _caller}),
    do: bump(counts, :scope_runs)

  defp count(
         counts,
         {:trace, _pid, :call, {NxCL.Compiler.Executor.Host, :run, [entry | _]}, _caller}
       ),
       do: host_run(counts, entry)

  defp count(counts, _other), do: counts

  defp host_run(counts, %{kind: :io_call, fun: nil}), do: counts

  defp host_run(counts, entry) do
    leaves = Nx.Defn.Composite.flatten_list([entry.in_template])
    uploads = if entry.kind == :runtime_call, do: leaf_count(entry.out_template), else: 0

    counts
    |> bump(:host_calls)
    |> bump(:callback_inputs, length(leaves))
    |> bump(:callback_input_bytes, leaves |> Enum.map(&Nx.byte_size/1) |> Enum.sum())
    |> bump(:callback_outputs, uploads)
  end

  defp leaf_count(template), do: length(Nx.Defn.Composite.flatten_list([template]))

  defp bump(counts, key, n \\ 1), do: Map.update!(counts, key, &(&1 + n))

  # Set once per VM. The `Host.run/3` pattern matches the one
  # `NxCL.TestSupport.NoFallback` sets, because a pattern on a function replaces the
  # earlier one. Each pattern must match exactly one function.
  defp install_patterns() do
    if :persistent_term.get({__MODULE__, :installed}, false) == false do
      Code.ensure_loaded!(NxCL.Compiler.Executor)
      Code.ensure_loaded!(NxCL.Compiler.Executor.Host)
      spec = [{:_, [], [{:message, {:caller}}]}]

      for {mod, fun, arity} <- [
            {NxCL.Compiler.Executor, :truthy?, 3},
            {NxCL.Compiler.Executor, :eval!, 3},
            {NxCL.Compiler.Executor, :run_sub, 3},
            {NxCL.Compiler.Executor.Host, :run, 3}
          ] do
        case :erlang.trace_pattern({mod, fun, arity}, spec, [:local]) do
          1 -> :ok
          n -> raise "trace pattern #{inspect(mod)}.#{fun}/#{arity} matched #{n} functions, not 1"
        end
      end

      :persistent_term.put({__MODULE__, :installed}, true)
    end
  end
end
