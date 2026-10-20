defmodule NxCL.Compiler.ProgramCache do
  @moduledoc """
  Bounded cache of compiled programs and executor plans. A compiled program
  is keyed by the device and `NxCL.Compiler.Program.signature/1`, so functions
  that lower alike share one entry. A plan holds no device state and is keyed by
  the signature alone.

  Callers use the public ETS table directly. If two processes build the same
  key at once, the first insert wins. Dropping an entry does not free a program
  a running call still holds; the native side frees it at the last release.

  The bound is `:max_items` (default 1024), oldest first out:

      config :nx_opencl, NxCL.Compiler.ProgramCache, max_items: 1024
  """

  use GenServer

  @default_max 1024

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    opts = [:named_table, :public, :set, read_concurrency: true, write_concurrency: true]
    :ets.new(__MODULE__, opts)
    :ets.insert(__MODULE__, {:__index__, 0})
    {:ok, nil}
  end

  @doc """
  The value under `key`, or what `build` returns on a miss.
  """
  @spec fetch_or_build(term(), (-> term())) :: term()
  def fetch_or_build(key, build) do
    case :ets.lookup(__MODULE__, key) do
      [{^key, value, _index}] ->
        value

      [] ->
        store(key, build.())
    end
  end

  @doc """
  Drops every entry. A running call keeps what it uses.
  """
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
end
