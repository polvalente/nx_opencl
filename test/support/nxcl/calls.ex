defmodule NxCL.TestSupport.Calls do
  @moduledoc """
  Counts calls of public functions from outside, with Erlang call tracing, so
  `lib/` carries no counters.

  `count/2` runs a function in the calling process and returns `{result, counts}`,
  where `counts` maps each `{module, function, arity}` asked for to the number of
  times the process called it. Tracing follows one process, so the code under test
  must run in the caller. A process has one tracer: do not nest `count/2` or use it
  inside `NxCL.TestSupport.Trace.run/1`.
  """

  @doc "Runs `fun` and returns `{result, counts}` for `mfas`."
  @spec count([mfa()], (-> result)) :: {result, %{mfa() => non_neg_integer()}} when result: var
  def count(mfas, fun) do
    for {mod, _fun, _arity} = mfa <- mfas do
      Code.ensure_loaded!(mod)
      # Exactly one function matches, or the count would be zero for the wrong reason.
      1 = :erlang.trace_pattern(mfa, true, [])
    end

    tracer = spawn_link(fn -> collect(Map.new(mfas, &{&1, 0})) end)
    :erlang.trace(self(), true, [:call, {:tracer, tracer}])

    try do
      result = fun.()
      {result, stop(tracer)}
    after
      :erlang.trace(self(), false, [:call])
      for mfa <- mfas, do: :erlang.trace_pattern(mfa, false, [])
    end
  end

  # A process does not trace itself, so a second process collects the messages.
  defp collect(counts) do
    receive do
      {:trace, _pid, :call, {mod, fun, args}} ->
        collect(Map.update!(counts, {mod, fun, length(args)}, &(&1 + 1)))

      {:done, from, ref} ->
        send(from, {ref, counts})
    end
  end

  defp stop(tracer) do
    :erlang.trace(self(), false, [:call])
    # Wait until the VM has handed every trace message to the collector.
    delivered = :erlang.trace_delivered(self())

    receive do
      {:trace_delivered, _pid, ^delivered} -> :ok
    after
      5000 -> raise "trace messages were not delivered"
    end

    ref = make_ref()
    send(tracer, {:done, self(), ref})

    receive do
      {^ref, counts} -> counts
    after
      5000 -> raise "trace collector did not answer"
    end
  end
end
