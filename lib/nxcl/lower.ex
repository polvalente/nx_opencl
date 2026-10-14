defmodule NxCL.Lower do
  @moduledoc """
  Lowers an `Nx.Defn.Expr` graph to an `NxCL.Program`.

  The pass walks the graph in post-order (`NxCL.Defn.Tree`) and emits one
  instruction per op node. `parameter`, scalar `constant` and `tensor` nodes emit
  no instruction; they become `{:input, i}`, `{:const, i}` and `{:capture, i}`
  refs. `metadata` nodes pass their operand's ref through, and `elem` picks a
  ref out of a tuple. A constant of a shape emits one `broadcast`. An operand
  that lacks the output shape of an elementwise op gets a `broadcast` too.

  An op the pass cannot lower raises `ArgumentError` with the prefix
  `"does not yet lower op"`. It never falls back to another path.
  """

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Defn.Tree
  alias NxCL.Program
  alias NxCL.Program.Builder

  @reduce_ops [:sum, :product, :reduce_max, :reduce_min, :all, :any]
  @arg_ops [:argmax, :argmin]
  @window_ops [:window_sum, :window_product, :window_max, :window_min]
  @scatter_ops [:window_scatter_max, :window_scatter_min]
  @scan_blocks %{
    Nx.Block.CumulativeSum => :cumulative_sum,
    Nx.Block.CumulativeProduct => :cumulative_product,
    Nx.Block.CumulativeMin => :cumulative_min,
    Nx.Block.CumulativeMax => :cumulative_max
  }

  # Binary ops a `reduce` body may hold: the result has the operand type.
  @step_ops ~w(add subtract multiply divide quotient remainder pow atan2 max min bitwise_and
               bitwise_or bitwise_xor left_shift right_shift)a

  @unary_ops Builder.unary_ops()
  @binary_ops Builder.binary_ops()

  # Ops Nx rejects on complex tensors (probed against Nx 1.0 on `c64`). Nx raises
  # while tracing, so these raise here only for a graph built another way.
  @complex_rejected ~w(atan2 bitwise_and bitwise_not bitwise_or bitwise_xor bitcast ceil clip
                       count_leading_zeros erf erf_inv erfc floor greater greater_equal
                       left_shift less less_equal max min population_count quotient remainder
                       right_shift round)a

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
    {state, ref} = put_const(state, value, node.type)

    if node.shape == {} do
      put_ref(state, id, ref)
    else
      # A constant of a shape is a scalar constant filling that shape.
      emit(state, node, Builder.broadcast(ref, node.type, node.shape, []))
    end
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :tensor, args: [tensor]}}, state) do
    ref = {:capture, length(state.captures)}
    put_ref(%{state | captures: [tensor | state.captures]}, id, ref)
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :metadata, args: [inner, _meta]}}, state) do
    if is_tuple(inner) do
      refs = inner |> Tuple.to_list() |> Enum.map(&ref(state, &1))
      put_ref(state, id, {:tuple, refs})
    else
      put_ref(state, id, ref(state, inner))
    end
  end

  # A `fun` node holds the body of a `reduce`; the node that owns it reads the body.
  defp expand(%T{data: %Nx.Defn.Expr{op: :fun}}, state), do: state

  # `elem` picks one ref out of a tuple node; it emits no instruction.
  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :elem, args: [tuple, pos]}}, state) do
    case Map.fetch!(state.refs, tuple.data.id) do
      {:tuple, refs} -> put_ref(state, id, Enum.at(refs, pos))
      _other -> unsupported(:elem, "(operand is not a tuple)")
    end
  end

  # -- Ops ----------------------------------------------------------------

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x]}} = node, state)
       when op in @unary_ops do
    check_complex(op, node, [x])
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Builder.unary(op, rx, node.type, node.shape))
  end

  # `Nx.logical_not/1` is a block; the elementwise kernel does the same job.
  defp expand(
         %T{data: %Nx.Defn.Expr{op: :block, args: [%Nx.Block.LogicalNot{}, [x], _default, _fun]}} =
           node,
         state
       ) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Builder.unary(:logical_not, rx, node.type, node.shape))
  end

  # `Nx.phase/1` is a block whose default is `atan2(imag, real)`; one kernel does it.
  defp expand(
         %T{data: %Nx.Defn.Expr{op: :block, args: [%Nx.Block.Phase{}, [x], _default, _fun]}} =
           node,
         state
       ) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Builder.unary(:phase, rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [l, r]}} = node, state)
       when op in @binary_ops do
    check_complex(op, node, [l, r])
    {state, [rl, rr]} = conform(state, node, [l, r])
    emit(state, node, Builder.binary(op, rl, rr, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :as_type, args: [x]}} = node, state) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Builder.as_type(rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :bitcast, args: [x]}} = node, state) do
    check_complex(:bitcast, node, [x])
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Builder.bitcast(rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :select, args: [p, t, f]}} = node, state) do
    {state, [rp, rt, rf]} = conform(state, node, [p, t, f])
    emit(state, node, Builder.select(rp, rt, rf, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :clip, args: [x, lo, hi]}} = node, state) do
    check_complex(:clip, node, [x, lo, hi])
    {state, [rx, rlo, rhi]} = conform(state, node, [x, lo, hi])
    emit(state, node, Builder.clip(rx, rlo, rhi, node.type, node.shape))
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

  defp expand(%T{data: %Nx.Defn.Expr{op: :broadcast, args: [x, _shape, axes]}} = node, state) do
    emit(state, node, Builder.broadcast(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :transpose, args: [x, axes]}} = node, state) do
    emit(state, node, Builder.transpose(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reverse, args: [x, axes]}} = node, state) do
    emit(state, node, Builder.reverse(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :slice, args: [x, starts, lengths, strides]}} = node,
         state
       ) do
    {state, starts} = start_refs(state, starts)

    emit(
      state,
      node,
      Builder.slice(ref(state, x), starts, node.type, node.shape, lengths, strides)
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :put_slice, args: [x, starts, slice]}} = node, state) do
    {state, rx} = cast(state, x, node.type)
    {state, rs} = cast(state, slice, node.type)
    {state, starts} = start_refs(state, starts)
    emit(state, node, Builder.put_slice(rx, rs, starts, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :pad, args: [x, value, config]}} = node, state) do
    {state, rx} = cast(state, x, node.type)
    {state, rv} = cast(state, value, node.type)
    emit(state, node, Builder.pad(rx, rv, node.type, node.shape, config))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [tensors, axis]}} = node, state)
       when op in [:concatenate, :stack] do
    {state, refs} = cast_all(state, tensors, node.type)
    emit(state, node, Builder.join(op, refs, node.type, node.shape, axis))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :iota, args: [axis]}} = node, state) do
    emit(state, node, Builder.iota(node.type, node.shape, axis))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :eye}} = node, state) do
    emit(state, node, Builder.eye(node.type, node.shape))
  end

  # -- Reductions, scans, windows, sort ------------------------------------

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in @reduce_ops do
    emit(
      state,
      node,
      Builder.reduce(op, ref(state, x), node.type, node.shape, axes(x, opts[:axes]))
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in @arg_ops do
    axes = axes(x, opts[:axis] && [opts[:axis]])

    emit(
      state,
      node,
      Builder.arg_reduce(op, ref(state, x), node.type, node.shape, opts[:tie_break], axes)
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reduce, args: [x, init, opts, fun]}} = node, state) do
    steps = reduce_steps(fun, node.type, :reduce)
    axes = axes(x, opts[:axes])

    cond do
      not monoid?(steps) ->
        # Any other body folds in order, starting from the initial accumulator.
        instr =
          Builder.reduce_steps(
            [ref(state, x), ref(state, init)],
            node.type,
            node.shape,
            steps,
            axes
          )

        emit(state, node, instr)

      identity_init?(steps, init) ->
        emit(
          state,
          node,
          Builder.reduce_steps([ref(state, x)], node.type, node.shape, steps, axes)
        )

      true ->
        # Fold the initial accumulator in: the result is `fun(total, init)`.
        {state, total} =
          push(state, Builder.reduce_steps([ref(state, x)], node.type, node.shape, steps, axes))

        {state, init_ref} = broadcast_to(state, init, node.shape)
        fold_steps(state, node, steps, total, init_ref)
    end
  end

  defp expand(
         %T{
           data: %Nx.Defn.Expr{
             op: :block,
             args: [%{__struct__: mod, axis: axis, reverse: reverse}, [x], _default, _fun]
           }
         } = node,
         state
       )
       when is_map_key(@scan_blocks, mod) do
    op = Map.fetch!(@scan_blocks, mod)
    emit(state, node, Builder.scan(op, ref(state, x), node.type, node.shape, axis, reverse))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, dims, opts]}} = node, state)
       when op in @window_ops do
    emit(
      state,
      node,
      Builder.window(
        op,
        ref(state, x),
        node.type,
        node.shape,
        dims,
        opts[:strides],
        opts[:window_dilations],
        opts[:padding]
      )
    )
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :window_reduce, args: [x, init, dims, opts, fun]}} = node,
         state
       ) do
    steps = reduce_steps(fun, node.type, :window_reduce)

    emit(
      state,
      node,
      Builder.window_reduce(
        ref(state, x),
        ref(state, init),
        node.type,
        node.shape,
        steps,
        {dims, opts[:strides], opts[:window_dilations], opts[:padding]}
      )
    )
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: op, args: [x, source, init, dims, opts]}} = node,
         state
       )
       when op in @scatter_ops do
    emit(
      state,
      node,
      Builder.window_scatter(
        op,
        ref(state, x),
        ref(state, source),
        ref(state, init),
        node.type,
        node.shape,
        {dims, opts[:strides], opts[:padding]}
      )
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in [:sort, :argsort] do
    emit(
      state,
      node,
      Builder.sort(op, ref(state, x), node.type, node.shape, opts[:axis] || 0, opts[:direction])
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op}}, _state), do: unsupported(op, nil)

  # -- Helpers ------------------------------------------------------------

  defp emit(state, %T{data: %Nx.Defn.Expr{id: id}}, instruction) do
    {state, ref} = push(state, instruction)
    put_ref(state, id, ref)
  end

  # Adds an instruction that no node owns, such as a broadcast or a cast.
  defp push(state, instruction) do
    ref = {:result, length(state.instrs)}
    {%{state | instrs: [instruction | state.instrs]}, ref}
  end

  defp put_const(state, value, type) do
    ref = {:const, length(state.constants)}
    {%{state | constants: [{value, type} | state.constants]}, ref}
  end

  defp put_ref(state, id, ref), do: %{state | refs: Map.put(state.refs, id, ref)}

  defp ref(state, %T{data: %Nx.Defn.Expr{id: id}}), do: Map.fetch!(state.refs, id)

  defp check_complex(op, node, operands) do
    if op in @complex_rejected and Enum.any?([node | operands], &Nx.Type.complex?(&1.type)) do
      raise ArgumentError, "#{inspect(op)} does not support complex tensors"
    end
  end

  # Every operand of an elementwise op has the output shape, except a scalar
  # constant, which the kernel takes by value. Any other operand gets a
  # broadcast instruction, aligned to the trailing axes as `Nx.broadcast` does.
  defp conform(state, %T{shape: shape}, operands) do
    {state, refs} =
      Enum.reduce(operands, {state, []}, fn x, {state, refs} ->
        {state, ref} = broadcast_to(state, x, shape)
        {state, [ref | refs]}
      end)

    {state, Enum.reverse(refs)}
  end

  defp broadcast_to(state, %T{shape: shape} = x, shape), do: {state, ref(state, x)}

  defp broadcast_to(state, %T{} = x, shape) do
    if scalar_const?(state, x) do
      {state, ref(state, x)}
    else
      rank = tuple_size(shape)
      axes = Enum.to_list((rank - tuple_size(x.shape))..(rank - 1)//1)
      push(state, Builder.broadcast(ref(state, x), x.type, shape, axes))
    end
  end

  # Casts an operand to the output type with an `as_type` instruction.
  defp cast(state, %T{type: type} = x, type), do: {state, ref(state, x)}

  defp cast(state, %T{} = x, type),
    do: push(state, Builder.as_type(ref(state, x), type, x.shape))

  defp cast_all(state, tensors, type) do
    {state, refs} =
      Enum.reduce(tensors, {state, []}, fn x, {state, refs} ->
        {state, ref} = cast(state, x, type)
        {state, [ref | refs]}
      end)

    {state, Enum.reverse(refs)}
  end

  # A start index is a scalar tensor or an integer; an integer becomes an `s64`
  # constant. The device clamps every start.
  defp start_refs(state, starts) do
    {state, refs} =
      Enum.reduce(starts, {state, []}, fn
        start, {state, refs} when is_integer(start) ->
          {state, ref} = put_const(state, start, {:s, 64})
          {state, [ref | refs]}

        %T{} = start, {state, refs} ->
          {state, [ref(state, start) | refs]}
      end)

    {state, Enum.reverse(refs)}
  end

  defp scalar_const?(state, %T{shape: {}} = operand),
    do: match?({:const, _}, ref(state, operand))

  defp scalar_const?(_state, _operand), do: false

  # The reduced axes, sorted; `nil` means every axis.
  defp axes(%T{shape: shape}, nil), do: Enum.to_list(0..(tuple_size(shape) - 1)//1)
  defp axes(_x, axes), do: Enum.sort(axes)

  # Lowers the body of a `reduce` or `window_reduce` function to steps over
  # two variables (the parameters) and the result of each earlier step. A body
  # with any other node raises.
  defp reduce_steps(%T{data: %Nx.Defn.Expr{op: :fun, args: [params, body, _mfa]}}, type, what) do
    if length(params) != 2 do
      unsupported(what, "(the function must take two arguments)")
    end

    {vars, steps} =
      body
      |> Tree.post_order()
      |> Enum.reduce({%{}, []}, fn
        %T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}}, {vars, steps} ->
          {Map.put(vars, id, pos), steps}

        %T{data: %Nx.Defn.Expr{id: id, op: op, args: [l, r]}} = node, {vars, steps}
        when op in @step_ops ->
          if Enum.any?([node, l, r], &(&1.type != type)) do
            unsupported(op, "(mixed types in a #{what} body)")
          end

          a = Map.get(vars, l.data.id) || unsupported(op, "(an operand of a #{what} body)")
          b = Map.get(vars, r.data.id) || unsupported(op, "(an operand of a #{what} body)")
          {Map.put(vars, id, 2 + length(steps)), [{op, a, b} | steps]}

        %T{data: %Nx.Defn.Expr{op: op}}, _acc ->
          unsupported(op, "(in a #{what} body)")
      end)

    steps = Enum.reverse(steps)

    if steps == [] or Map.fetch!(vars, body.data.id) != 1 + length(steps) do
      unsupported(what, "(the body must end in a binary op)")
    end

    steps
  end

  # A body of one `add`, `multiply`, `max` or `min` over the two parameters is
  # associative and commutative, so the device may fold it in a tree.
  defp monoid?([{op, a, b}]) when op in [:add, :multiply, :max, :min],
    do: {a, b} in [{0, 1}, {1, 0}]

  defp monoid?(_steps), do: false

  # A body that is one `add` or `multiply` with its identity as the initial
  # accumulator is `sum` or `product`; nothing is left to fold in.
  defp identity_init?([{op, a, b}], %T{data: %Nx.Defn.Expr{op: :constant, args: [v]}})
       when {a, b} in [{0, 1}, {1, 0}] do
    (op == :add and v == 0) or (op == :multiply and v == 1)
  end

  defp identity_init?(_steps, _init), do: false

  # Emits `fun(total, init)` as elementwise instructions; the last one is the node's.
  defp fold_steps(state, node, steps, total, init) do
    last = length(steps) - 1

    {state, _refs} =
      steps
      |> Enum.with_index()
      |> Enum.reduce({state, [total, init]}, fn {{op, a, b}, k}, {state, refs} ->
        instr = Builder.binary(op, Enum.at(refs, a), Enum.at(refs, b), node.type, node.shape)

        if k == last do
          {emit(state, node, instr), refs}
        else
          {state, ref} = push(state, instr)
          {state, refs ++ [ref]}
        end
      end)

    state
  end

  defp unsupported(op, detail) do
    suffix = if detail, do: " " <> detail, else: ""
    raise ArgumentError, "does not yet lower op #{inspect(op)}#{suffix}"
  end
end
