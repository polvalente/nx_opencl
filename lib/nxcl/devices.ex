defmodule NxCL.Devices do
  @moduledoc """
  The device registry. It opens each device context once and shares it across
  processes.

  `device:` is an index (counting every device, GPUs first) or `{platform,
  index}` (counting one platform's devices). The default is 0.

  The registry must not crash alone: its table holds the only map from a context
  to its device, and the caches hold programs built for those contexts.
  The application supervisor restarts the caches with it.

  A context has a default queue and any number of extra queues (`new_queue/1`).
  A program runs on the queue `command_queue:` names, or the default. Queues of
  one context share its kernel cache; each runs one program at a time.
  """

  use GenServer

  @type key :: {non_neg_integer() | nil, non_neg_integer()}
  @type t :: %{ctx: reference(), key: key(), queue: reference() | nil}

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil) do
    :ets.new(__MODULE__, [:named_table, :protected, :set, read_concurrency: true])
    {:ok, nil}
  end

  @doc """
  The device and queue `opts` name (`:device`, `:command_queue`), or `{:error, reason}`
  when the device cannot open. Raises `ArgumentError` when `:device` is not an
  index or a `{platform, index}` pair.
  """
  @spec fetch(keyword()) :: {:ok, t()} | {:error, String.t()}
  def fetch(opts \\ []) do
    key = key(opts)
    queue = Keyword.get(opts, :command_queue)

    with {:ok, ctx} <- context(key), do: {:ok, %{ctx: ctx, key: key, queue: queue}}
  end

  @doc "Like `fetch/1`; raises `NxCL.CompilerError` on an error."
  @spec fetch!(keyword()) :: t()
  def fetch!(opts \\ []) do
    case fetch(opts) do
      {:ok, device} ->
        device

      {:error, reason} ->
        raise NxCL.CompilerError, message: "cannot open the OpenCL device: #{reason}"
    end
  end

  @doc """
  Opens a command queue on the context `opts` name. Pass it as `command_queue:`.
  """
  @spec new_queue(keyword()) :: {:ok, reference()} | {:error, String.t()}
  def new_queue(opts \\ []) do
    with {:ok, %{ctx: ctx}} <- fetch(Keyword.delete(opts, :command_queue)) do
      case NxCL.Native.queue_create(ctx) do
        queue when is_reference(queue) -> {:ok, queue}
        {:error, reason} -> {:error, reason}
      end
    end
  end

  @doc "The registry entry of an open context, or `nil`."
  @spec for_ctx(reference()) :: t() | nil
  def for_ctx(ctx) do
    case :ets.match(__MODULE__, {:"$1", ctx}) do
      [[key] | _] -> %{ctx: ctx, key: key, queue: nil}
      [] -> nil
    end
  end

  @doc false
  @spec key(keyword()) :: key()
  def key(opts) do
    case Keyword.get(opts, :device, 0) do
      index when is_integer(index) and index >= 0 ->
        {nil, index}

      {platform, index} = key
      when (is_nil(platform) or (is_integer(platform) and platform >= 0)) and
             is_integer(index) and index >= 0 ->
        key

      other ->
        raise ArgumentError,
              "expected :device to be an index or a {platform, index} pair, got: #{inspect(other)}"
    end
  end

  defp context(key) do
    case :ets.lookup(__MODULE__, key) do
      [{^key, ctx}] -> {:ok, ctx}
      [] -> GenServer.call(__MODULE__, {:open, key}, :infinity)
    end
  end

  @impl true
  def handle_call({:open, key}, _from, state) do
    reply =
      case :ets.lookup(__MODULE__, key) do
        [{^key, ctx}] -> {:ok, ctx}
        [] -> open(key)
      end

    {:reply, reply, state}
  end

  defp open({platform, index} = key) do
    result =
      if platform,
        do: NxCL.Native.device_ctx_create_on(platform, index),
        else: NxCL.Native.device_ctx_create(index)

    case result do
      ctx when is_reference(ctx) ->
        :ets.insert(__MODULE__, {key, ctx})
        {:ok, ctx}

      {:error, reason} ->
        {:error, to_string(reason)}
    end
  end
end
