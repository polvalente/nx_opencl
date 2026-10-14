defmodule NxCL.Executor.Stats do
  @moduledoc """
  Counters of the executor's control steps, for tests and measurement.

    * `:predicate_reads` - scalar predicates read back from the device (`cond`
      clauses and dynamic `while` conditions).
    * `:segment_runs` - straight-line segments run natively.
    * `:loop_iterations` - `while` body runs, static or dynamic.
  """

  @keys [:predicate_reads, :segment_runs, :loop_iterations]

  @doc "Adds one to a counter."
  @spec bump(atom()) :: :ok
  def bump(key), do: :counters.add(counters(), index(key), 1)

  @doc "The counters as a map."
  @spec get() :: %{atom() => non_neg_integer()}
  def get() do
    ref = counters()
    Map.new(@keys, &{&1, :counters.get(ref, index(&1))})
  end

  @doc "Sets every counter to zero."
  @spec reset() :: :ok
  def reset() do
    ref = counters()
    Enum.each(@keys, &:counters.put(ref, index(&1), 0))
  end

  defp index(key), do: Enum.find_index(@keys, &(&1 == key)) + 1

  defp counters() do
    case :persistent_term.get({__MODULE__, :counters}, nil) do
      nil ->
        ref = :counters.new(length(@keys), [:atomics])
        :persistent_term.put({__MODULE__, :counters}, ref)
        ref

      ref ->
        ref
    end
  end
end
