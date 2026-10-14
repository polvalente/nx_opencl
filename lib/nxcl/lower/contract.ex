defmodule NxCL.Lower.Contract do
  @moduledoc """
  Lowers `dot` and `conv` to instructions.

  A `dot` becomes transposes and reshapes of both operands to `[b, m, k]` and
  `[b, k, n]`, one `batched_gemm`, and a reshape of the result. Rank 1 and
  scalar-contract cases take the same path. A `conv` becomes transposes to the
  layout the native `conv` instruction reads (input `[n, c, spatial...]`, kernel
  `[o, i, kernel spatial...]`), the instruction, and one transpose of its
  `[o, n, spatial...]` result to the layout the caller asked for. The native side
  expands `conv` to im2col plus `batched_gemm`; Elixir builds no kernel source.

  Both functions take the lowering state and `push`, a function
  `(state, instruction) -> {state, ref}` that appends an instruction. They return
  `{state, ref}` for the node's result. The operand refs already have the node's type.
  """

  alias NxCL.Program.Builder

  @type push :: (map(), NxCL.Program.Instruction.t() ->
                   {map(), NxCL.Program.Instruction.ref()})

  @doc """
  Lowers `dot` over operand refs `{left, right}`. `sides` holds
  `{shape, contract_axes, batch_axes}` for each operand.
  """
  @spec dot(map(), push(), map(), tuple(), tuple()) ::
          {map(), NxCL.Program.Instruction.ref()}
  def dot(state, push, node, {rl, rr}, {{ls, cl, bl}, {rs, cr, br}}) do
    free_l = axes_without(ls, cl ++ bl)
    free_r = axes_without(rs, cr ++ br)

    b = size(ls, bl)
    m = size(ls, free_l)
    k = size(ls, cl)
    n = size(rs, free_r)

    {state, left_ref} = to_matrix(state, push, rl, ls, bl ++ free_l ++ cl, {b, m, k}, node.type)
    {state, right_ref} = to_matrix(state, push, rr, rs, br ++ cr ++ free_r, {b, k, n}, node.type)

    {state, gemm} =
      push.(state, Builder.batched_gemm(left_ref, right_ref, node.type, {b, m, n}, k))

    reshape(state, push, gemm, node.type, {b, m, n}, node.shape)
  end

  @doc """
  Lowers `conv` over `{{input, input_shape}, {kernel, kernel_shape}}`. `opts` are the
  Expr options (`Nx.conv/3` after normalization).
  """
  @spec conv(map(), push(), map(), tuple(), keyword()) ::
          {map(), NxCL.Program.Instruction.ref()}
  def conv(state, push, node, {{rx, xs}, {rw, ws}}, opts) do
    {state, input_ref, _} = permute(state, push, rx, xs, opts[:input_permutation], node.type)
    {state, kernel_ref, _} = permute(state, push, rw, ws, opts[:kernel_permutation], node.type)

    out_perm = opts[:output_permutation]

    options = %{
      strides: opts[:strides],
      padding: opts[:padding],
      kernel_dilation: opts[:kernel_dilation],
      input_dilation: opts[:input_dilation],
      feature_groups: opts[:feature_group_size],
      batch_groups: opts[:batch_group_size]
    }

    # The output is `[n, o, spatial...]` permuted by the inverse of `out_perm`; the
    # native result is the same with the first two axes swapped.
    [n, o | spatial] = node.shape |> Tuple.to_list() |> permute_list(out_perm)
    raw_shape = List.to_tuple([o, n | spatial])

    {state, raw} =
      push.(state, Builder.conv(input_ref, kernel_ref, node.type, raw_shape, options))

    axes = out_perm |> invert() |> Enum.map(&swap_leading/1)

    if axes == Enum.to_list(0..(length(axes) - 1)//1) do
      {state, raw}
    else
      push.(state, Builder.transpose(raw, node.type, node.shape, axes))
    end
  end

  # -- Helpers ------------------------------------------------------------

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

  # Transposes `ref` by `axes` (skipped for the identity) and reshapes to `target`.
  defp to_matrix(state, push, ref, shape, axes, target, type) do
    {state, ref, permuted} = permute(state, push, ref, shape, axes, type)
    {state, ref} = reshape(state, push, ref, type, permuted, target)
    {state, ref}
  end

  defp permute(state, push, ref, shape, axes, type) do
    permuted = shape |> Tuple.to_list() |> permute_list(axes) |> List.to_tuple()

    if axes == Enum.to_list(0..(tuple_size(shape) - 1)//1) do
      {state, ref, shape}
    else
      {state, ref} = push.(state, Builder.transpose(ref, type, permuted, axes))
      {state, ref, permuted}
    end
  end

  defp reshape(state, _push, ref, _type, shape, shape), do: {state, ref}

  defp reshape(state, push, ref, type, _from, to) do
    push.(state, Builder.alias_op(:reshape, ref, type, to))
  end
end
