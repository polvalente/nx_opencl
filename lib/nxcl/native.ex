defmodule NxCL.Native do
  @moduledoc false

  # Derive Rust target triple from Nerves environment variables.
  # Nerves sets TARGET_ARCH, TARGET_OS, TARGET_ABI during cross-compilation.
  # RUSTLER_TARGET overrides if set explicitly.
  rust_target =
    cond do
      target = System.get_env("RUSTLER_TARGET") ->
        target

      (arch = System.get_env("TARGET_ARCH")) && System.get_env("TARGET_OS") ->
        abi = System.get_env("TARGET_ABI") || "gnu"

        case {arch, abi} do
          {"aarch64", abi} -> "aarch64-unknown-linux-#{abi}"
          {"arm", "gnueabihf"} -> "armv7-unknown-linux-gnueabihf"
          {"arm", abi} -> "armv7-unknown-linux-#{abi}"
          {"x86_64", abi} -> "x86_64-unknown-linux-#{abi}"
          {arch, abi} -> "#{arch}-unknown-linux-#{abi}"
        end

      true ->
        nil
    end

  # When cross-compiling, tell Cargo to use Nerves' CC as the linker.
  linker_env =
    case {rust_target, System.get_env("CC")} do
      {nil, _} ->
        []

      {_, nil} ->
        []

      {target, cc} ->
        triple_env = target |> String.upcase() |> String.replace("-", "_")
        [{"CARGO_TARGET_#{triple_env}_LINKER", cc}]
    end

  use Rustler,
    otp_app: :nx_opencl,
    crate: "nxcl_nif",
    target: rust_target,
    env: linker_env

  @spec device_ctx_create(non_neg_integer()) :: reference() | {:error, String.t()}
  def device_ctx_create(_device_id), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Like `device_ctx_create/1`, with the index counted inside one platform."
  @spec device_ctx_create_on(non_neg_integer(), non_neg_integer()) ::
          reference() | {:error, String.t()}
  def device_ctx_create_on(_platform, _device_id), do: :erlang.nif_error(:nif_not_loaded)

  @doc "A new command queue on the context, for `eval_program_on/3`."
  @spec queue_create(reference()) :: reference() | {:error, String.t()}
  def queue_create(_ctx), do: :erlang.nif_error(:nif_not_loaded)

  @spec device_info(reference()) :: [{String.t(), String.t()}]
  def device_info(_ctx), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_create(reference(), non_neg_integer()) :: reference()
  def buffer_create(_ctx, _size), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_write(reference(), binary()) :: reference()
  def buffer_write(_ctx, _data), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_read(reference(), non_neg_integer()) :: binary()
  def buffer_read(_buf, _size), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Reads several buffers with one wait on the device. Each item is `{buffer, size}`;
  the result lists the binaries in the same order.
  """
  @spec buffer_read_many(reference(), [{reference(), non_neg_integer()}]) :: [binary()]
  def buffer_read_many(_ctx, _items), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Drops the handle's hold on the device memory. The memory goes when no other
  handle shares it. Reading a freed handle raises. Returns `false` when the
  handle was already free.
  """
  @spec buffer_free(reference()) :: boolean()
  def buffer_free(_buf), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Decodes and validates a program wire term and builds its kernels. See
  `NxCL.Executor` for the term layout.
  """
  @spec compile_program(reference(), tuple()) :: {:ok, reference()} | {:error, String.t()}
  def compile_program(_ctx, _wire), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Runs a compiled program on a list of buffers (inputs, captures, constants in
  slot order; `nil` for an unread input). On the default queue it returns before the
  device is done and the first read of an output waits; a program that can report an
  error code waits at the end.
  """
  @spec eval_program(reference(), [reference() | nil]) ::
          {:ok, [reference()]} | {:error, String.t()}
  def eval_program(_program, _buffers), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Like `eval_program/2`, on a queue from `queue_create/1`. The queue must belong
  to the program's device context.
  """
  @spec eval_program_on(reference(), [reference() | nil], reference()) ::
          {:ok, [reference()]} | {:error, String.t()}
  def eval_program_on(_program, _buffers, _queue), do: :erlang.nif_error(:nif_not_loaded)
end
