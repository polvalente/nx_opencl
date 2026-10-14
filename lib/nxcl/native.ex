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

  @spec device_info(reference()) :: [{String.t(), String.t()}]
  def device_info(_ctx), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_create(reference(), non_neg_integer()) :: reference()
  def buffer_create(_ctx, _size), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_write(reference(), binary()) :: reference()
  def buffer_write(_ctx, _data), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_read(reference(), non_neg_integer()) :: binary()
  def buffer_read(_buf, _size), do: :erlang.nif_error(:nif_not_loaded)

  @spec elementwise_binary_op(
          reference(),
          String.t(),
          reference(),
          reference(),
          non_neg_integer()
        ) ::
          reference()
  def elementwise_binary_op(_ctx, _op, _a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec elementwise_unary_op(reference(), String.t(), reference(), non_neg_integer()) ::
          reference()
  def elementwise_unary_op(_ctx, _op, _a, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec matmul_op(
          reference(),
          reference(),
          reference(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: reference()
  def matmul_op(_ctx, _a, _b, _m, _n, _k), do: :erlang.nif_error(:nif_not_loaded)

  @spec reduce_op(reference(), String.t(), reference(), non_neg_integer()) :: reference()
  def reduce_op(_ctx, _op, _input, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec softmax_op(reference(), reference(), non_neg_integer(), non_neg_integer()) :: reference()
  def softmax_op(_ctx, _input, _rows, _cols), do: :erlang.nif_error(:nif_not_loaded)
end
