defmodule NxCL.Executor.Stats do
  @moduledoc """
  Counters of the executor's control steps, for tests and measurement.

    * `:predicate_reads` - scalar predicates read back from the device (`cond`
      clauses and dynamic `while` conditions).
    * `:segment_runs` - straight-line segments run natively.
    * `:loop_iterations` - `while` body runs, static or dynamic.
    * `:host_calls` - host callbacks run.
    * `:host_reads` - tensors read back from the device for a callback. A tensor
      a callback does not take is never counted.
    * `:host_read_bytes` - bytes of those tensors.
    * `:host_uploads` - tensors a callback returned and the executor uploaded. A
      side-effect callback uploads none: its outputs are its input buffers.
  """

  @keys [
    :predicate_reads,
    :segment_runs,
    :loop_iterations,
    :host_calls,
    :host_reads,
    :host_read_bytes,
    :host_uploads
  ]

  @doc "Adds `n` (default one) to a counter."
  @spec bump(atom(), integer()) :: :ok
  def bump(key, n \\ 1), do: :counters.add(counters(), index(key), n)

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
