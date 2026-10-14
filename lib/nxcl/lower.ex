defmodule NxCL.Lower do
  @moduledoc """
  Lowers an `Nx.Defn.Expr` graph to an `NxCL.Program`.

  The pass walks the graph in post-order (`NxCL.Defn.Tree`) and emits one
  instruction per op node. `parameter`, `constant` and `tensor` nodes emit no
  instruction; they become `{:input, i}`, `{:const, i}` and `{:capture, i}`
  refs. `metadata` nodes pass their operand's ref through.

  An op the pass cannot lower raises `ArgumentError` with the prefix
  `"does not yet lower op"`. It never falls back to another path.
  """

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Defn.Tree
  alias NxCL.Program
  alias NxCL.Program.Builder

  @unary_ops Builder.unary_ops()
  @binary_ops Builder.binary_ops()

  @doc """
  Lowers `output`, the result of tracing a function with `num_inputs`
  flattened arguments.

  The input list always has `num_inputs` entries, so a function that ignores
  an argument keeps the positions of the others.
  """
  @spec lower(Nx.Container.t(), non_neg_integer() | nil) :: Program.t()
  def lower(output, num_inputs \\ nil) do
    ordered = Tree.post_order(output)

    state =
      Enum.reduce(
        ordered,
        %{inputs: %{}, captures: [], constants: [], instrs: [], refs: %{}},
        &expand/2
      )

    max_pos = state.inputs |> Map.keys() |> Enum.max(fn -> -1 end)
    arity = max(num_inputs || 0, max_pos + 1)
    inputs = for pos <- 0..(arity - 1)//1, do: Map.get(state.inputs, pos)

    outputs =
      [output]
      |> Composite.flatten_list()
      |> Enum.map(&Map.fetch!(state.refs, &1.data.id))

    instructions = Enum.reverse(state.instrs)

    %Program{
      inputs: inputs,
      captures: Enum.reverse(state.captures),
      constants: Enum.reverse(state.constants),
      instructions: instructions,
      outputs: outputs,
      output_template: Composite.traverse(output, &Nx.to_template/1),
      ref_counts: ref_counts(instructions, outputs)
    }
  end

  defp ref_counts(instructions, outputs) do
    instructions
    |> Enum.flat_map(& &1.operands)
    |> Enum.concat(outputs)
    |> Enum.frequencies()
  end

  # -- Leaves -------------------------------------------------------------

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}} = node, state) do
    state = %{state | inputs: Map.put(state.inputs, pos, {node.type, node.shape})}
    put_ref(state, id, {:input, pos})
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :constant, args: [value]}} = node, state) do
    if node.shape != {} do
      unsupported(:broadcast, "(constant of shape #{inspect(node.shape)})")
    end

    ref = {:const, length(state.constants)}
    put_ref(%{state | constants: [{value, node.type} | state.constants]}, id, ref)
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :tensor, args: [tensor]}}, state) do
    ref = {:capture, length(state.captures)}
    put_ref(%{state | captures: [tensor | state.captures]}, id, ref)
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :metadata, args: [inner, _meta]}}, state) do
    if is_tuple(inner), do: unsupported(:metadata, "(tuple operand)")
    put_ref(state, id, Map.fetch!(state.refs, inner.data.id))
  end

  # -- Ops ----------------------------------------------------------------

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x]}} = node, state)
       when op in @unary_ops do
    check_shapes(node, [x], state)
    emit(state, node, Builder.unary(op, ref(state, x), node.type, node.shape))
  end

  # `Nx.logical_not/1` is a block; the elementwise kernel does the same job.
  defp expand(
         %T{data: %Nx.Defn.Expr{op: :block, args: [%Nx.Block.LogicalNot{}, [x], _default, _fun]}} =
           node,
         state
       ) do
    check_shapes(node, [x], state)
    emit(state, node, Builder.unary(:logical_not, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [l, r]}} = node, state)
       when op in @binary_ops do
    check_shapes(node, [l, r], state)
    emit(state, node, Builder.binary(op, ref(state, l), ref(state, r), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :as_type, args: [x]}} = node, state) do
    check_shapes(node, [x], state)
    emit(state, node, Builder.as_type(ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :bitcast, args: [x]}} = node, state) do
    check_shapes(node, [x], state)
    emit(state, node, Builder.bitcast(ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :select, args: [p, t, f]}} = node, state) do
    check_shapes(node, [p, t, f], state)

    emit(
      state,
      node,
      Builder.select(ref(state, p), ref(state, t), ref(state, f), node.type, node.shape)
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :clip, args: [x, lo, hi]}} = node, state) do
    check_shapes(node, [x, lo, hi], state)

    emit(
      state,
      node,
      Builder.clip(ref(state, x), ref(state, lo), ref(state, hi), node.type, node.shape)
    )
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :dot, args: [l, [1], [], r, [0], []]}} = node,
         state
       )
       when tuple_size(l.shape) == 2 and tuple_size(r.shape) == 2 do
    for t <- [l, r, node], t.type != {:f, 32} do
      unsupported(:dot, "(f32 only until stage 08, got #{inspect(t.type)})")
    end

    k = elem(l.shape, 1)
    emit(state, node, Builder.dot(ref(state, l), ref(state, r), node.type, node.shape, k))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :dot}}, _state) do
    unsupported(:dot, "(only 2D matmul, contracting axis 1 with axis 0)")
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reshape, args: [x]}} = node, state) do
    emit(state, node, Builder.alias_op(:reshape, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :squeeze, args: [x, _axes]}} = node, state) do
    emit(state, node, Builder.alias_op(:squeeze, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op}}, _state), do: unsupported(op, nil)

  # -- Helpers ------------------------------------------------------------

  defp emit(state, %T{data: %Nx.Defn.Expr{id: id}}, instruction) do
    ref = {:result, length(state.instrs)}
    put_ref(%{state | instrs: [instruction | state.instrs]}, id, ref)
  end

  defp put_ref(state, id, ref), do: %{state | refs: Map.put(state.refs, id, ref)}

  defp ref(state, %T{data: %Nx.Defn.Expr{id: id}}), do: Map.fetch!(state.refs, id)

  # The runtime does not broadcast yet. Every operand has the output shape,
  # except a scalar constant, which the kernel takes by value.
  defp check_shapes(%T{data: %Nx.Defn.Expr{op: op}, shape: shape}, operands, state) do
    if Enum.any?(operands, &(&1.shape != shape and not scalar_const?(state, &1))) do
      unsupported(op, "(operands of different shapes need broadcast)")
    end
  end

  defp scalar_const?(state, %T{shape: {}} = operand),
    do: match?({:const, _}, ref(state, operand))

  defp scalar_const?(_state, _operand), do: false

  defp unsupported(op, detail) do
    suffix = if detail, do: " " <> detail, else: ""
    raise ArgumentError, "does not yet lower op #{inspect(op)}#{suffix}"
  end
end
