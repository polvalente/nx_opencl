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
       when op in [:negate, :exp, :log, :tanh, :sigmoid, :abs] do
    check_shapes(node, [x])
    emit(state, node, Builder.unary(op, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [l, r]}} = node, state)
       when op in [:add, :subtract, :multiply, :divide] do
    check_shapes(node, [l, r])
    emit(state, node, Builder.binary(op, ref(state, l), ref(state, r), node.type, node.shape))
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :dot, args: [l, [1], [], r, [0], []]}} = node,
         state
       )
       when tuple_size(l.shape) == 2 and tuple_size(r.shape) == 2 do
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

  # The runtime does not broadcast yet, so every operand needs the output shape.
  defp check_shapes(%T{data: %Nx.Defn.Expr{op: op}, shape: shape}, operands) do
    if Enum.any?(operands, &(&1.shape != shape)) do
      unsupported(op, "(operands of different shapes need broadcast)")
    end
  end

  defp unsupported(op, detail) do
    suffix = if detail, do: " " <> detail, else: ""
    raise ArgumentError, "does not yet lower op #{inspect(op)}#{suffix}"
  end
end
