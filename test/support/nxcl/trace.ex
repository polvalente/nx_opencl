defmodule NxCL.TestSupport.Trace do
  @moduledoc """
  Counts the executor's control steps from outside, with Erlang call tracing, so
  `lib/` carries no counters.

  `run/1` runs a function in the calling process and returns `{result, counts}`:

    * `:predicate_reads` - calls of `NxCL.Executor.truthy?/3`, each of which reads
      one scalar predicate back from the device (a `cond` clause or a dynamic
      `while` condition).
    * `:segment_runs` - calls of `NxCL.Executor.eval!/3`, each of which runs one
      straight-line segment natively.
    * `:scope_runs` - calls of `NxCL.Executor.run_sub/3`: every `while` body run,
      predicate scope and `cond` branch. A dynamic loop's body runs are
      `scope_runs - predicate_reads`; with no `cond` and a static trip count they
      are `scope_runs`.
    * `:host_calls` - calls of `NxCL.Executor.Host.run/3` that run a function (an
      `:io_call` with no function runs nothing).
    * `:host_reads` and `:host_read_bytes` - the tensors `Host.run/3` reads back
      for a function: one per leaf of the entry's `in_template`, and their bytes.
    * `:host_uploads` - the tensors `Host.run/3` uploads: one per leaf of a
      `:runtime_call` output template. An `:io_call` uploads none.

  The NIF functions (`buffer_read`, `eval_program`) cannot be call-traced, so the
  counts come from the Elixir functions around them. Tracing follows one process:
  the executor must run in the caller (it does), and a test that uses this module
  must be `async: false`. Do not nest `run/1` or use it inside
  `NxCL.Parity.Tripwire.trace/1`: a process has one tracer.
  """

  @empty %{
    predicate_reads: 0,
    segment_runs: 0,
    scope_runs: 0,
    host_calls: 0,
    host_reads: 0,
    host_read_bytes: 0,
    host_uploads: 0
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
      when mod in [NxCL.Executor, NxCL.Executor.Host] ->
        collect([message | messages])

      # Other patterns (`NxCL.Parity.Tripwire` sets some) can fire; skip them.
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

  defp count(counts, {:trace, _pid, :call, {NxCL.Executor, :truthy?, _args}, _caller}),
    do: bump(counts, :predicate_reads)

  defp count(counts, {:trace, _pid, :call, {NxCL.Executor, :eval!, _args}, _caller}),
    do: bump(counts, :segment_runs)

  defp count(counts, {:trace, _pid, :call, {NxCL.Executor, :run_sub, _args}, _caller}),
    do: bump(counts, :scope_runs)

  defp count(counts, {:trace, _pid, :call, {NxCL.Executor.Host, :run, [entry | _]}, _caller}),
    do: host_run(counts, entry)

  defp count(counts, _other), do: counts

  defp host_run(counts, %{kind: :io_call, fun: nil}), do: counts

  defp host_run(counts, entry) do
    leaves = Nx.Defn.Composite.flatten_list([entry.in_template])
    uploads = if entry.kind == :runtime_call, do: leaf_count(entry.out_template), else: 0

    counts
    |> bump(:host_calls)
    |> bump(:host_reads, length(leaves))
    |> bump(:host_read_bytes, leaves |> Enum.map(&Nx.byte_size/1) |> Enum.sum())
    |> bump(:host_uploads, uploads)
  end

  defp leaf_count(template), do: length(Nx.Defn.Composite.flatten_list([template]))

  defp bump(counts, key, n \\ 1), do: Map.update!(counts, key, &(&1 + n))

  # Patterns are set once per VM. Each appends the caller to the trace message. The
  # `Host.run/3` pattern matches the one `NxCL.Parity.Tripwire` sets, because a
  # pattern on a function replaces the earlier one.
  defp install_patterns() do
    if :persistent_term.get({__MODULE__, :installed}, false) == false do
      Code.ensure_loaded!(NxCL.Executor)
      Code.ensure_loaded!(NxCL.Executor.Host)
      spec = [{:_, [], [{:message, {:caller}}]}]

      for {mod, fun, arity} <- [
            {NxCL.Executor, :truthy?, 3},
            {NxCL.Executor, :eval!, 3},
            {NxCL.Executor, :run_sub, 3},
            {NxCL.Executor.Host, :run, 3}
          ] do
        :erlang.trace_pattern({mod, fun, arity}, spec, [:local])
      end

      :persistent_term.put({__MODULE__, :installed}, true)
    end
  end
end
