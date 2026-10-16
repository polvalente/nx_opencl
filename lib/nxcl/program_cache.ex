defmodule NxCL.ProgramCache do
  @moduledoc """
  Bounded cache of compiled program resources and executor plans, shared by
  every call site. A key is an id-free signature of the lowered program (see
  `NxCL.Program.signature/1`) plus the device, so two functions that lower to
  the same program share one resource.

  The table is public. Readers and writers use ETS directly; this process only
  owns the tables. When two processes build the same key at once, the first
  insert wins and the other uses the winner's value.

  Dropping an entry only drops the cache's hold on the resource. A call that
  runs it keeps its own hold, and the native side frees the program, its kernels
  and its buffers when the last holder lets go.

  The bound is `:max_items`, default 1024, FIFO:

      config :nx_opencl, NxCL.ProgramCache, max_items: 1024
  """

  use GenServer

  alias NxCL.CacheStats

  @default_max 1024

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    opts = [:named_table, :public, :set, read_concurrency: true, write_concurrency: true]
    :ets.new(__MODULE__, opts)
    :ets.insert(__MODULE__, {:__index__, 0})
    :ets.new(CacheStats.table(), opts)
    {:ok, nil}
  end

  @doc """
  The value under `key` (a hit), or what `build` returns (a miss), stored first
  come first kept. `kind` is `:structural` or `:plan`; it picks the counters.
  """
  @spec fetch_or_build(:structural | :plan, term(), (-> term())) :: term()
  def fetch_or_build(kind, key, build) do
    case :ets.lookup(__MODULE__, key) do
      [{^key, value, _index}] ->
        CacheStats.bump(stat(kind, :hit))
        value

      [] ->
        CacheStats.bump(stat(kind, :miss))
        store(key, build.())
    end
  end

  @doc "Drops every entry. A running call keeps what it uses."
  @spec clear() :: :ok
  def clear() do
    :ets.select_delete(__MODULE__, [{{:_, :_, :_}, [], [true]}])
    :ok
  end

  @doc "Number of entries."
  @spec size() :: non_neg_integer()
  def size(), do: :ets.info(__MODULE__, :size) - 1

  defp store(key, value) do
    index = :ets.update_counter(__MODULE__, :__index__, 1)

    if :ets.insert_new(__MODULE__, {key, value, index}) do
      evict(index, max_items())
      value
    else
      case :ets.lookup(__MODULE__, key) do
        [{^key, winner, _index}] -> winner
        [] -> value
      end
    end
  end

  defp evict(index, max) when index > max do
    cutoff = index - max
    :ets.select_delete(__MODULE__, [{{:_, :_, :"$1"}, [{:"=<", :"$1", cutoff}], [true]}])
  end

  defp evict(_index, _max), do: :ok

  defp max_items() do
    case :nx_opencl |> Application.get_env(__MODULE__, []) |> Keyword.get(:max_items) do
      nil ->
        @default_max

      n when is_integer(n) and n > 0 ->
        n

      other ->
        raise ArgumentError,
              "config :nx_opencl, #{inspect(__MODULE__)}, max_items must be a positive " <>
                "integer, got: #{inspect(other)}"
    end
  end

  defp stat(:structural, :hit), do: :structural_hit
  defp stat(:structural, :miss), do: :structural_miss
  defp stat(:plan, :hit), do: :plan_hit
  defp stat(:plan, :miss), do: :plan_miss
end
