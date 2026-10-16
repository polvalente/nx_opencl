defmodule NxCL.CacheStats do
  @moduledoc """
  Counters of the compile caches, for tests and measurement.

    * `:lower` - calls of `NxCL.Lower.lower/2`.
    * `:closure_hit`, `:closure_miss` - `NxCL.CompileCache` lookups.
    * `:structural_hit`, `:structural_miss` - lookups of a compiled program
      resource in `NxCL.ProgramCache`.
    * `:plan_hit`, `:plan_miss` - lookups of an executor plan.
  """

  @doc false
  @spec table() :: atom()
  def table(), do: __MODULE__

  @doc "Adds one to a counter."
  @spec bump(atom()) :: :ok
  def bump(key) do
    :ets.update_counter(__MODULE__, key, 1, {key, 0})
    :ok
  end

  @doc "The counters as a map. A counter never bumped is zero."
  @spec get() :: %{atom() => non_neg_integer()}
  def get() do
    zero =
      Map.new(
        ~w(lower closure_hit closure_miss structural_hit structural_miss plan_hit plan_miss)a,
        &{&1, 0}
      )

    Map.merge(zero, Map.new(:ets.tab2list(__MODULE__)))
  end

  @doc "Sets every counter to zero."
  @spec reset() :: :ok
  def reset() do
    :ets.delete_all_objects(__MODULE__)
    :ok
  end
end
