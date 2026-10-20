defmodule NxCL.Compiler.CompileCache do
  @moduledoc """
  Bounded cache of compiled closures, one per function, input templates and run
  options (`:device`, `:command_queue`).

  `:max_items` defaults to 1024 and `:ttl` to `:infinity`. Either takes a
  positive integer or `:infinity`:

      config :nx_opencl, NxCL.Compiler.CompileCache,
        max_items: 4096,
        ttl: :timer.minutes(30)

  Overflow evicts the oldest insert first. An entry older than `:ttl` is a
  miss on read, and a sweep removes it every half `:ttl`. Reads use the ETS
  table directly; writes and the sweep run in this process. The sweep timer
  starts with the cache, so set `:ttl` in config before the application starts.

  ## Memory

  A closure holds the tensors its function captured, and those stay on the
  device while the entry lives. With the default 1024 entries and no `:ttl`, a
  program that captures large tensors can hold a lot of device memory. Lower
  `:max_items`, or set a `:ttl`, to release them sooner.
  """

  use GenServer

  @default_max_items 1024

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    :ets.new(__MODULE__, [:named_table, :protected, :set, read_concurrency: true])
    # Validate both bounds at start.
    _ = {max_items(), ttl()}
    {:ok, schedule_sweep(%{counter: 0})}
  end

  @doc "The cached value for `key`, or `{:error, :cache_miss}`."
  @spec fetch(term()) :: {:ok, term()} | {:error, :cache_miss}
  def fetch(key) do
    case :ets.lookup(__MODULE__, key) do
      [{^key, value, expires, _index}] ->
        if expires == :infinity or expires > now(),
          do: {:ok, value},
          else: {:error, :cache_miss}

      [] ->
        {:error, :cache_miss}
    end
  end

  @doc "Stores `value` under `key` and returns it."
  @spec put(term(), term()) :: term()
  def put(key, value), do: GenServer.call(__MODULE__, {:put, key, value})

  @impl true
  def handle_call({:put, key, value}, _from, state) do
    index = state.counter + 1
    :ets.insert(__MODULE__, {key, value, expires(), index})
    evict_overflow(max_items(), index)
    {:reply, value, %{state | counter: index}}
  end

  @impl true
  def handle_info(:sweep, state) do
    :ets.select_delete(__MODULE__, [
      {{:_, :_, :"$1", :_}, [{:is_integer, :"$1"}, {:"=<", :"$1", now()}], [true]}
    ])

    {:noreply, schedule_sweep(state)}
  end

  defp expires() do
    case ttl() do
      :infinity -> :infinity
      ttl -> now() + ttl
    end
  end

  defp schedule_sweep(state) do
    case ttl() do
      :infinity -> :ok
      ttl -> Process.send_after(self(), :sweep, max(div(ttl, 2), 1))
    end

    state
  end

  defp evict_overflow(:infinity, _index), do: :ok

  defp evict_overflow(max, index) when index > max do
    cutoff = index - max
    :ets.select_delete(__MODULE__, [{{:_, :_, :_, :"$1"}, [{:"=<", :"$1", cutoff}], [true]}])
  end

  defp evict_overflow(_max, _index), do: :ok

  defp now(), do: System.monotonic_time(:millisecond)

  defp ttl(), do: bound!(config(:ttl, :infinity), :ttl)
  defp max_items(), do: bound!(config(:max_items, @default_max_items), :max_items)

  defp config(key, default) do
    :nx_opencl |> Application.get_env(__MODULE__, []) |> Keyword.get(key, default)
  end

  defp bound!(:infinity, _key), do: :infinity
  defp bound!(n, _key) when is_integer(n) and n > 0, do: n

  defp bound!(other, key) do
    raise ArgumentError,
          "config :nx_opencl, #{inspect(__MODULE__)}, #{key} must be :infinity or a " <>
            "positive integer, got: #{inspect(other)}"
  end
end
