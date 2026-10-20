defmodule NxCL.Compiler.Lower do
  @moduledoc """
  Lowers an `Nx.Defn.Expr` graph to an `NxCL.Compiler.Program`.

  It walks the graph in post-order and emits one instruction per op node.
  `parameter`, scalar `constant` and `tensor` nodes emit none; they become
  `{:input, i}`, `{:const, i}` and `{:capture, i}` refs. An operand that lacks
  the output shape of an elementwise op gets a `broadcast`.

  A `cond` or `while` becomes one instruction with nested bodies. An `io_call`
  or `runtime_call` becomes one `:host_call` instruction. An op it cannot lower
  raises `ArgumentError` ("does not yet lower op ...").

  ## Scopes

  The program is the root scope. Each body of a `cond` or `while` is a scope of
  its own, lowered by `Lower.Control`, an internal module. A scope has its own
  instruction list and its own refs, and it starts with `{:carry, i}` refs for
  the values the `cond` or `while` passes in. It reads its parent through those
  refs only. A scope that reaches a parent node or parameter raises
  `ArgumentError`, because the parent's refs do not exist in the body. Literal
  tensors and constants may appear in both scopes. They are shared tables of the
  program, not refs of a scope.
  """

  import NxCL.Compiler.Lower.Shared,
    only: [axes: 2, check_complex: 3, fit_axis: 6, monoid?: 1, push: 2, put_const: 3]

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Lower.Block
  alias NxCL.Compiler.Lower.Contract
  alias NxCL.Compiler.Lower.Control
  alias NxCL.Compiler.Lower.HostCall
  alias NxCL.Compiler.Lower.Shared
  alias NxCL.Compiler.Ops
  alias NxCL.Compiler.Program
  alias NxCL.Compiler.Tree

  @reduce_ops [:sum, :product, :reduce_max, :reduce_min, :all, :any]
  @arg_ops [:argmax, :argmin]
  @window_ops [:window_sum, :window_product, :window_max, :window_min]
  @scatter_ops [:window_scatter_max, :window_scatter_min]
  @scan_blocks Shared.scan_blocks()

  @unary_ops Ops.unary_ops()
  @binary_ops Ops.binary_ops()

  @typedoc "What a node id maps to: a ref, or the refs of a tuple-typed node."
  @type node_ref :: Instruction.ref() | {:tuple, [node_ref()]}

  @typedoc """
  The lowering state. `:captures`, `:constants`, `:instrs` and `:host_calls`
  are built in reverse. `:avail` and `:outer` hold the node ids of the current
  and parent scopes; `:depth` counts nested bodies.
  """
  @type state :: %{
          inputs: %{non_neg_integer() => {Nx.Type.t(), tuple()}},
          captures: [Nx.Tensor.t()],
          constants: [{term(), Nx.Type.t()}],
          instrs: [Instruction.t()],
          host_calls: [Program.host_call()],
          refs: %{reference() => node_ref()},
          avail: %{reference() => term()},
          outer: %{reference() => term()},
          depth: non_neg_integer(),
          scope: :root | :sub
        }

  @doc """
  Lowers `output`, the trace of a function with `num_inputs` flattened
  arguments. The input list has `num_inputs` entries, so an ignored argument
  keeps the positions of the others.
  """
  @spec lower(Nx.Container.t(), non_neg_integer() | nil) :: Program.t()
  def lower(output, num_inputs \\ nil) do
    state = %{
      inputs: %{},
      captures: [],
      constants: [],
      instrs: [],
      host_calls: [],
      refs: %{},
      avail: Nx.Defn.Tree.scope_ids(output),
      outer: %{},
      depth: 0,
      scope: :root
    }

    ordered = Tree.post_order(output, &Control.scope_deps(&1, state))
    state = Enum.reduce(ordered, state, &expand/2)

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
      host_calls: Enum.reverse(state.host_calls),
      outputs: outputs,
      output_template: Composite.traverse(output, &Nx.to_template/1),
      ref_counts: Shared.ref_counts(instructions, outputs)
    }
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :parameter}}, %{scope: :sub}) do
    raise ArgumentError,
          "an inner scope reads a parameter of its parent scope; " <>
            "pass the value in as an argument of the while or cond"
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}} = node, state) do
    state = %{state | inputs: Map.put(state.inputs, pos, {node.type, node.shape})}
    put_ref(state, id, {:input, pos})
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :constant, args: [value]}} = node, state) do
    {state, ref} = put_const(state, value, node.type)

    if node.shape == {} do
      put_ref(state, id, ref)
    else
      emit(state, node, Ops.broadcast(ref, node.type, node.shape, []))
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

  # The `reduce` that owns a `fun` reads its body.
  defp expand(%T{data: %Nx.Defn.Expr{op: :fun}}, state), do: state

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :elem, args: [tuple, pos]}}, state) do
    case Map.fetch!(state.refs, tuple.data.id) do
      {:tuple, refs} -> put_ref(state, id, Enum.at(refs, pos))
      _other -> unsupported(:elem, "(operand is not a tuple)")
    end
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x]}} = node, state)
       when op in @unary_ops do
    check_complex(op, node, [x])
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Ops.unary(op, rx, node.type, node.shape))
  end

  # `Nx.logical_not/1` is a block; the elementwise kernel does it.
  defp expand(
         %T{data: %Nx.Defn.Expr{op: :block, args: [%Nx.Block.LogicalNot{}, [x], _default, _fun]}} =
           node,
         state
       ) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Ops.unary(:logical_not, rx, node.type, node.shape))
  end

  # `Nx.phase/1` is a block; one kernel replaces its default `atan2(imag, real)`.
  defp expand(
         %T{data: %Nx.Defn.Expr{op: :block, args: [%Nx.Block.Phase{}, [x], _default, _fun]}} =
           node,
         state
       ) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Ops.unary(:phase, rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [l, r]}} = node, state)
       when op in @binary_ops do
    check_complex(op, node, [l, r])
    {state, [rl, rr]} = conform(state, node, [l, r])
    emit(state, node, Ops.binary(op, rl, rr, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :as_type, args: [x]}} = node, state) do
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Ops.as_type(rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :bitcast, args: [x]}} = node, state) do
    check_complex(:bitcast, node, [x])
    {state, [rx]} = conform(state, node, [x])
    emit(state, node, Ops.bitcast(rx, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :select, args: [p, t, f]}} = node, state) do
    {state, [rp, rt, rf]} = conform(state, node, [p, t, f])
    emit(state, node, Ops.select(rp, rt, rf, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :clip, args: [x, lo, hi]}} = node, state) do
    check_complex(:clip, node, [x, lo, hi])
    {state, [rx, rlo, rhi]} = conform(state, node, [x, lo, hi])
    emit(state, node, Ops.clip(rx, rlo, rhi, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :dot, args: args}} = node, state) do
    [l, contract_l, batch_l, r, contract_r, batch_r] = args
    {state, rl} = cast(state, l, node.type)
    {state, rr} = cast(state, r, node.type)

    sides = {{l.shape, contract_l, batch_l}, {r.shape, contract_r, batch_r}}
    {state, ref} = Contract.dot(state, node, {rl, rr}, sides)
    put_ref(state, id, ref)
  end

  defp expand(%T{data: %Nx.Defn.Expr{id: id, op: :conv, args: [x, w, opts]}} = node, state) do
    {state, rx} = cast(state, x, node.type)
    {state, rw} = cast(state, w, node.type)
    {state, ref} = Contract.conv(state, node, {{rx, x.shape}, {rw, w.shape}}, opts)
    put_ref(state, id, ref)
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reshape, args: [x]}} = node, state) do
    emit(state, node, Ops.alias_op(:reshape, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :squeeze, args: [x, _axes]}} = node, state) do
    emit(state, node, Ops.alias_op(:squeeze, ref(state, x), node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :broadcast, args: [x, _shape, axes]}} = node, state) do
    emit(state, node, Ops.broadcast(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :transpose, args: [x, axes]}} = node, state) do
    emit(state, node, Ops.transpose(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reverse, args: [x, axes]}} = node, state) do
    emit(state, node, Ops.reverse(ref(state, x), node.type, node.shape, axes))
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :slice, args: [x, starts, lengths, strides]}} = node,
         state
       ) do
    {state, starts} = start_refs(state, starts)

    emit(
      state,
      node,
      Ops.slice(ref(state, x), starts, node.type, node.shape, lengths, strides)
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :put_slice, args: [x, starts, slice]}} = node, state) do
    {state, rx} = cast(state, x, node.type)
    {state, rs} = cast(state, slice, node.type)
    {state, starts} = start_refs(state, starts)
    emit(state, node, Ops.put_slice(rx, rs, starts, node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :pad, args: [x, value, config]}} = node, state) do
    {state, rx} = cast(state, x, node.type)
    {state, rv} = cast(state, value, node.type)
    emit(state, node, Ops.pad(rx, rv, node.type, node.shape, config))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [tensors, axis]}} = node, state)
       when op in [:concatenate, :stack] do
    {state, refs} = cast_all(state, tensors, node.type)
    emit(state, node, Ops.join(op, refs, node.type, node.shape, axis))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :iota, args: [axis]}} = node, state) do
    emit(state, node, Ops.iota(node.type, node.shape, axis))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :eye}} = node, state) do
    emit(state, node, Ops.eye(node.type, node.shape))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in @reduce_ops do
    emit(
      state,
      node,
      Ops.reduce(op, ref(state, x), node.type, node.shape, axes(x, opts[:axes]))
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in @arg_ops do
    axes = axes(x, opts[:axis] && [opts[:axis]])

    emit(
      state,
      node,
      Ops.arg_reduce(op, ref(state, x), node.type, node.shape, opts[:tie_break], axes)
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :reduce, args: [x, init, opts, fun]}} = node, state) do
    steps = reduce_steps(fun, node.type, :reduce)
    axes = axes(x, opts[:axes])

    cond do
      not monoid?(steps) ->
        # The body is not a monoid: fold in order from the initial accumulator.
        instr =
          Ops.reduce_steps(
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
          Ops.reduce_steps([ref(state, x)], node.type, node.shape, steps, axes)
        )

      true ->
        # The result is `fun(total, init)`.
        {state, total} =
          push(state, Ops.reduce_steps([ref(state, x)], node.type, node.shape, steps, axes))

        {state, init_ref} = broadcast_to(state, init, node.shape)

        {state, ref} =
          Shared.fold_steps(state, steps, [total, init_ref], node.type, node.shape)

        put_ref(state, node.data.id, ref)
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
    emit(state, node, Ops.scan(op, ref(state, x), node.type, node.shape, axis, reverse))
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, dims, opts]}} = node, state)
       when op in @window_ops do
    emit(
      state,
      node,
      Ops.window(
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
      Ops.window_reduce(
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
      Ops.window_scatter(
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
      Ops.sort(op, ref(state, x), node.type, node.shape, opts[:axis] || 0, opts[:direction])
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: :gather, args: [x, indices, opts]}} = node, state) do
    emit(
      state,
      node,
      Ops.gather(ref(state, x), ref(state, indices), node.type, node.shape, opts[:axes])
    )
  end

  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, indices, updates, opts]}} = node, state)
       when op in [:indexed_add, :indexed_put] do
    {state, rx} = cast(state, x, node.type)
    {state, ru} = cast(state, updates, node.type)
    refs = {rx, ref(state, indices), ru}
    emit(state, node, Ops.indexed(op, refs, node.type, node.shape, opts[:axes]))
  end

  defp expand(
         %T{data: %Nx.Defn.Expr{op: :triangular_solve, args: [a, b, opts]}} = node,
         state
       ) do
    {state, ra} = cast(state, a, node.type)
    {state, rb} = cast(state, b, node.type)
    emit(state, node, Ops.triangular_solve(ra, rb, node.type, node.shape, opts))
  end

  defp expand(
         %T{
           data: %Nx.Defn.Expr{
             op: :block,
             args: [%Nx.Block.LinAlg.Cholesky{}, [x], _default, _fun]
           }
         } = node,
         state
       ) do
    {state, rx} = cast(state, x, node.type)
    emit(state, node, Ops.cholesky(rx, node.type, node.shape))
  end

  # Nx pads or crops the axis to `:length` first.
  defp expand(%T{data: %Nx.Defn.Expr{op: op, args: [x, opts]}} = node, state)
       when op in [:fft, :ifft] do
    {state, rx} = cast(state, x, node.type)
    axis = opts[:axis]
    {state, rx} = fit_axis(state, rx, x.shape, node.type, node.shape, axis)
    emit(state, node, Ops.fft(op, rx, node.type, node.shape, axis))
  end

  # A block with no native clause above lowers through its default.
  defp expand(%T{data: %Nx.Defn.Expr{op: :block}} = node, state),
    do: Block.lower(state, node, &expand/2)

  defp expand(%T{data: %Nx.Defn.Expr{op: :cond}} = node, state),
    do: Control.cond_node(state, node, &expand/2)

  defp expand(%T{data: %Nx.Defn.Expr{op: :while}} = node, state),
    do: Control.while_node(state, node, &expand/2)

  defp expand(%T{data: %Nx.Defn.Expr{op: op}} = node, state) when op in [:io_call, :runtime_call],
    do: HostCall.node(state, node)

  defp expand(%T{data: %Nx.Defn.Expr{op: op}}, _state), do: unsupported(op, nil)

  defp emit(state, %T{data: %Nx.Defn.Expr{id: id}}, instruction) do
    {state, ref} = push(state, instruction)
    put_ref(state, id, ref)
  end

  defp put_ref(state, id, ref), do: %{state | refs: Map.put(state.refs, id, ref)}

  defp ref(state, %T{data: %Nx.Defn.Expr{id: id}}), do: Map.fetch!(state.refs, id)

  # Every operand of an elementwise op gets the output shape, except a scalar
  # constant, which the kernel takes by value. Axes align to the trailing ones.
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
      push(state, Ops.broadcast(ref(state, x), x.type, shape, axes))
    end
  end

  defp cast(state, %T{type: type} = x, type), do: {state, ref(state, x)}

  defp cast(state, %T{} = x, type),
    do: push(state, Ops.as_type(ref(state, x), type, x.shape))

  defp cast_all(state, tensors, type) do
    {state, refs} =
      Enum.reduce(tensors, {state, []}, fn x, {state, refs} ->
        {state, ref} = cast(state, x, type)
        {state, [ref | refs]}
      end)

    {state, Enum.reverse(refs)}
  end

  # An integer start becomes an `s64` constant. The device clamps every start.
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

  # The steps of the `fun` node of a `reduce` or `window_reduce`.
  defp reduce_steps(%T{data: %Nx.Defn.Expr{op: :fun, args: [params, body, _mfa]}}, type, what),
    do: Shared.reduce_steps(params, body, type, what)

  # One `add` or `multiply` with its identity as the initial value: nothing to fold in.
  defp identity_init?([{op, a, b}], %T{data: %Nx.Defn.Expr{op: :constant, args: [v]}})
       when {a, b} in [{0, 1}, {1, 0}] do
    (op == :add and v == 0) or (op == :multiply and v == 1)
  end

  defp identity_init?(_steps, _init), do: false

  defp unsupported(op, detail) do
    suffix = if detail, do: " " <> detail, else: ""
    raise ArgumentError, "does not yet lower op #{inspect(op)}#{suffix}"
  end
end
