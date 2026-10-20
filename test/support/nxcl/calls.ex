defmodule NxCL.TestSupport.Calls do
  @moduledoc """
  Counts calls of public functions with Erlang call tracing, so `lib/` has no
  counters. `count/2` runs a function in the calling process and returns
  `{result, counts}`, mapping each `{module, function, arity}` asked for to the
  number of calls. The code under test must run in the caller. A process has one
  tracer: do not nest `count/2` or use it inside `NxCL.TestSupport.Trace.run/1`.
  """

  alias NxCL.TestSupport.Collector

  @doc "Runs `fun` and returns `{result, counts}` for `mfas`."
  @spec count([mfa()], (-> result)) :: {result, %{mfa() => non_neg_integer()}} when result: var
  def count(mfas, fun) do
    for {mod, _fun, _arity} = mfa <- mfas do
      Code.ensure_loaded!(mod)
      # Exactly one function must match, or a zero count proves nothing.
      1 = :erlang.trace_pattern(mfa, true, [])
    end

    collector = Collector.start(Map.new(mfas, &{&1, 0}), &tally/2)
    Collector.trace_self(collector)

    try do
      result = fun.()
      {result, Collector.stop(collector)}
    after
      Collector.discard(collector)
      for mfa <- mfas, do: :erlang.trace_pattern(mfa, false, [])
    end
  end

  defp tally({:trace, _pid, :call, {mod, fun, args}}, counts),
    do: Map.update!(counts, {mod, fun, length(args)}, &(&1 + 1))

  # Patterns that other tracers set for the whole VM can fire as well (they send the
  # caller too); only the functions asked for count.
  defp tally(_message, counts), do: counts
end
