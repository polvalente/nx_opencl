defmodule NxCL.Compiler.Lower.Contract do
  @moduledoc false
  # Lowers `dot` and `conv`. A `dot` becomes transposes and reshapes to
  # `[b, m, k]` and `[b, k, n]`, one `batched_gemm`, and a reshape. A `conv`
  # becomes transposes to the layout the native `conv` reads, the instruction,
  # and one transpose of its `[o, n, spatial...]` result. The native side expands
  # `conv` to im2col plus `batched_gemm`. Operand refs already have the node's type.

  import NxCL.Compiler.Lower.Shared, only: [identity?: 1, push: 2]

  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Lower.Shared
  alias NxCL.Compiler.Ops

  @doc """
  Lowers `dot` over `{left, right}` refs. `sides` holds `{shape, contract_axes,
  batch_axes}` per operand.
  """
  @spec dot(Shared.builder(), Shared.tensor(), {Instruction.ref(), Instruction.ref()}, tuple()) ::
          {Shared.builder(), Instruction.ref()}
  def dot(state, node, {rl, rr}, {{ls, cl, bl}, {rs, cr, br}}) do
    free_l = axes_without(ls, cl ++ bl)
    free_r = axes_without(rs, cr ++ br)

    b = size(ls, bl)
    m = size(ls, free_l)
    k = size(ls, cl)
    n = size(rs, free_r)

    {state, left_ref} = to_matrix(state, rl, ls, bl ++ free_l ++ cl, {b, m, k}, node.type)
    {state, right_ref} = to_matrix(state, rr, rs, br ++ cr ++ free_r, {b, k, n}, node.type)

    {state, gemm} =
      push(state, Ops.batched_gemm(left_ref, right_ref, node.type, {b, m, n}, k))

    reshape(state, gemm, node.type, {b, m, n}, node.shape)
  end

  @doc """
  Lowers `conv` over `{{input, input_shape}, {kernel, kernel_shape}}`. `opts` are
  the Expr options.
  """
  @spec conv(Shared.builder(), Shared.tensor(), tuple(), keyword()) ::
          {Shared.builder(), Instruction.ref()}
  def conv(state, node, {{rx, xs}, {rw, ws}}, opts) do
    {state, input_ref, _} = permute(state, rx, xs, opts[:input_permutation], node.type)
    {state, kernel_ref, _} = permute(state, rw, ws, opts[:kernel_permutation], node.type)

    out_perm = opts[:output_permutation]

    options = %{
      strides: opts[:strides],
      padding: opts[:padding],
      kernel_dilation: opts[:kernel_dilation],
      input_dilation: opts[:input_dilation],
      feature_groups: opts[:feature_group_size],
      batch_groups: opts[:batch_group_size]
    }

    # The native result has the first two axes of `[n, o, spatial...]` swapped.
    [n, o | spatial] = node.shape |> Tuple.to_list() |> permute_list(out_perm)
    raw_shape = List.to_tuple([o, n | spatial])

    {state, raw} =
      push(state, Ops.conv(input_ref, kernel_ref, node.type, raw_shape, options))

    axes = out_perm |> invert() |> Enum.map(&swap_leading/1)

    if identity?(axes) do
      {state, raw}
    else
      push(state, Ops.transpose(raw, node.type, node.shape, axes))
    end
  end

  # Axis `a` of the `[n, o, ...]` layout is axis `swap_leading(a)` of `[o, n, ...]`.
  defp swap_leading(0), do: 1
  defp swap_leading(1), do: 0
  defp swap_leading(axis), do: axis

  defp invert(perm) do
    perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  # `list` permuted by `axes`: entry `i` is `list[axes[i]]`.
  defp permute_list(list, axes), do: Enum.map(axes, &Enum.at(list, &1))

  defp axes_without(shape, taken) do
    for axis <- 0..(tuple_size(shape) - 1)//1, axis not in taken, do: axis
  end

  defp size(shape, axes), do: Enum.reduce(axes, 1, &(elem(shape, &1) * &2))

  defp to_matrix(state, ref, shape, axes, target, type) do
    {state, ref, permuted} = permute(state, ref, shape, axes, type)
    {state, ref} = reshape(state, ref, type, permuted, target)
    {state, ref}
  end

  defp permute(state, ref, shape, axes, type) do
    permuted = shape |> Tuple.to_list() |> permute_list(axes) |> List.to_tuple()

    if identity?(axes) do
      {state, ref, shape}
    else
      {state, ref} = push(state, Ops.transpose(ref, type, permuted, axes))
      {state, ref, permuted}
    end
  end

  defp reshape(state, ref, _type, shape, shape), do: {state, ref}

  defp reshape(state, ref, type, _from, to) do
    push(state, Ops.alias_op(:reshape, ref, type, to))
  end
end
