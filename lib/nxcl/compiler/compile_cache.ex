defmodule NxCL.Compiler.CompileCache do
  @moduledoc """
  Bounded cache of compiled closures. `NxCL.Compiler.__compile__/4` stores one
  closure per `{key, input templates, device}`, so a repeat `Nx.Defn.jit/2` call
  of the same function skips the trace and the lowering.

  Both limits default to `:infinity`:

      config :nx_opencl, NxCL.Compiler.CompileCache,
        max_items: 1024,
        ttl: :timer.minutes(30)

  Overflow evicts the oldest insert first. An entry that outlives `:ttl` counts
  as a miss on read, and the process sweeps it out every half `:ttl`.

  Reads go straight to the ETS table; only writes and the sweep run in this
  process.
  """

  use GenServer

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    opts = Keyword.validate!(opts, [:ttl, :max_items, name: __MODULE__])
    {name, opts} = Keyword.pop!(opts, :name)
    GenServer.start_link(__MODULE__, {name, opts}, name: name)
  end

  @impl true
  def init({name, opts}) do
    :ets.new(name, [:named_table, :protected, :set, read_concurrency: true])
    state = %{table: name, counter: 0, ttl: opts[:ttl], max_items: opts[:max_items]}
    _ = max_items(state)
    {:ok, schedule_sweep(state)}
  end

  @doc "The cached value for `key`, or `{:error, :cache_miss}`."
  @spec fetch(term(), atom()) :: {:ok, term()} | {:error, :cache_miss}
  def fetch(key, server \\ __MODULE__) do
    case :ets.lookup(server, key) do
      [{^key, value, expires, _index}] ->
        if expires == :infinity or expires > now(),
          do: {:ok, value},
          else: {:error, :cache_miss}

      [] ->
        {:error, :cache_miss}
    end
  end

  @doc "Stores `value` under `key` and returns it."
  @spec put(term(), term(), atom()) :: term()
  def put(key, value, server \\ __MODULE__), do: GenServer.call(server, {:put, key, value})

  @doc "Number of entries, expired ones included until the next sweep."
  @spec size(atom()) :: non_neg_integer()
  def size(server \\ __MODULE__), do: :ets.info(server, :size)

  @impl true
  def handle_call({:put, key, value}, _from, state) do
    index = state.counter + 1
    :ets.insert(state.table, {key, value, expires(state), index})
    evict_overflow(state.table, max_items(state), index)
    {:reply, value, %{state | counter: index}}
  end

  @impl true
  def handle_info(:sweep, state) do
    :ets.select_delete(state.table, [
      {{:_, :_, :"$1", :_}, [{:is_integer, :"$1"}, {:"=<", :"$1", now()}], [true]}
    ])

    {:noreply, schedule_sweep(state)}
  end

  defp expires(state) do
    case ttl(state) do
      :infinity -> :infinity
      ttl -> now() + ttl
    end
  end

  defp schedule_sweep(state) do
    case ttl(state) do
      :infinity -> :ok
      ttl -> Process.send_after(self(), :sweep, max(div(ttl, 2), 1))
    end

    state
  end

  defp evict_overflow(_table, :infinity, _index), do: :ok

  defp evict_overflow(table, max, index) when index > max do
    cutoff = index - max
    :ets.select_delete(table, [{{:_, :_, :_, :"$1"}, [{:"=<", :"$1", cutoff}], [true]}])
  end

  defp evict_overflow(_table, _max, _index), do: :ok

  defp now(), do: System.monotonic_time(:millisecond)

  defp ttl(%{ttl: ttl}) when ttl != nil, do: bound!(ttl, :ttl)
  defp ttl(_state), do: bound!(config(:ttl), :ttl)

  defp max_items(%{max_items: n}) when n != nil, do: bound!(n, :max_items)
  defp max_items(_state), do: bound!(config(:max_items), :max_items)

  defp config(key) do
    :nx_opencl |> Application.get_env(__MODULE__, []) |> Keyword.get(key, :infinity)
  end

  defp bound!(:infinity, _key), do: :infinity
  defp bound!(n, _key) when is_integer(n) and n > 0, do: n

  defp bound!(other, key) do
    raise ArgumentError,
          "config :nx_opencl, #{inspect(__MODULE__)}, #{key} must be :infinity or a " <>
            "positive integer, got: #{inspect(other)}"
  end
end
