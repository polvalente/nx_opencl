defmodule NxCL.Compiler.Evaluator do
  @moduledoc false
  # One-op programs for `NxCL.Backend`.
  #
  # Each backend callback builds a small `NxCL.Compiler.Program` from its arguments with the
  # `NxCL.Compiler.Ops` functions that `NxCL.Compiler.Lower` uses, then runs it through
  # `NxCL.Compiler.Executor`. The tensors are the program's inputs, already on the device,
  # so nothing crosses to the host. A call may emit more than one instruction (a
  # broadcast or a cast before the op), as the lowering does.
  #
  # The executor caches the decoded program by its wire term, which holds the op,
  # the operand types and shapes, and the attrs. A repeat call skips the kernel
  # build.
  #
  # An op the lowering cannot take raises `ArgumentError` ("does not yet lower op
  # ..."). Nothing runs on `Nx.BinaryBackend`.

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Evaluator.Steps
  alias NxCL.Compiler.Executor
  alias NxCL.Compiler.Lower
  alias NxCL.Compiler.Lower.Block
  alias NxCL.Compiler.Lower.Contract
  alias NxCL.Compiler.Ops
  alias NxCL.Compiler.Program

  @doc """
  Builds and runs a program. `out` is the result template (a tensor or a
  container of them), `tensors` the operands. `build` gets the builder state and
  the operands as `{ref, tensor}` pairs and returns `{state, ref | [ref]}`.
  """
  @spec run(Nx.Container.t(), [T.t()], (map(), [{term(), T.t()}] -> {map(), term()})) ::
          Nx.Container.t()
  def run(out, tensors, build) do
    tensors = Enum.map(tensors, &device/1)
    operands = tensors |> Enum.with_index() |> Enum.map(fn {t, i} -> {{:input, i}, t} end)

    {st, out_refs} = build.(%{instrs: [], constants: [], host_calls: []}, operands)
    program = program(st, tensors, out, List.wrap(out_refs))
    opts = device_opts(tensors)
    buffers = Executor.run_buffers(program, Enum.map(tensors, &{:buffer, &1.data.ref}), opts)
    ctx = NxCL.Devices.fetch!(opts).ctx

    {result, []} =
      Composite.traverse(out, buffers, fn t, [buf | rest] ->
        {%{t | data: %NxCL.Backend{ref: buf, ctx: ctx}}, rest}
      end)

    result
  end

  defp program(st, tensors, out, out_refs) do
    instructions = Enum.reverse(st.instrs)

    %Program{
      inputs: Enum.map(tensors, &{&1.type, &1.shape}),
      constants: Enum.reverse(st.constants),
      instructions: instructions,
      host_calls: Enum.reverse(st.host_calls),
      outputs: out_refs,
      output_template: Composite.traverse(out, &Nx.to_template/1),
      ref_counts: Lower.ref_counts(instructions, out_refs)
    }
  end

  # A tensor from another backend goes to the device first.
  # The device of the first operand that sits on one; the default device otherwise.
  defp device_opts(tensors) do
    with %T{data: %NxCL.Backend{ctx: ctx}} <- List.first(tensors),
         %{key: key} <- NxCL.Devices.for_ctx(ctx) do
      [device: key]
    else
      _ -> []
    end
  end

  defp device(%T{data: %NxCL.Backend{}} = t), do: t
  defp device(%T{} = t), do: NxCL.Backend.from_binary(t, Nx.to_binary(t), [])

  # -- Operand helpers ----------------------------------------------------

  # Broadcasts an operand to `shape` with the trailing axes aligned, as
  # `Nx.broadcast` does, unless it already has the shape.
  defp conform(st, {ref, %T{shape: shape}}, shape), do: {st, ref}

  defp conform(st, {ref, %T{shape: from, type: type}}, shape) do
    rank = tuple_size(shape)
    axes = Enum.to_list((rank - tuple_size(from))..(rank - 1)//1)
    Lower.push(st, Ops.broadcast(ref, type, shape, axes))
  end

  defp conform_all(st, operands, shape), do: map_state(st, operands, &conform(&1, &2, shape))

  # Casts an operand to `type` with an `as_type` instruction.
  defp cast(st, {ref, %T{type: type}}, type), do: {st, ref}
  defp cast(st, {ref, %T{} = x}, type), do: Lower.push(st, Ops.as_type(ref, type, x.shape))

  defp cast_all(st, operands, type), do: map_state(st, operands, &cast(&1, &2, type))

  defp map_state(st, items, fun) do
    {st, refs} =
      Enum.reduce(items, {st, []}, fn item, {st, refs} ->
        {st, ref} = fun.(st, item)
        {st, [ref | refs]}
      end)

    {st, Enum.reverse(refs)}
  end

  defp refs(operands), do: Enum.map(operands, &elem(&1, 0))

  # A start index is a scalar tensor or an integer; an integer goes up as an
  # `s64` scalar so every start index is an operand and the program caches by shape.
  defp start_tensor(%T{} = t), do: t

  defp start_tensor(i) when is_integer(i),
    do: NxCL.Backend.from_binary(Nx.template({}, {:s, 64}), <<i::signed-64-native>>, [])

  # -- Elementwise ----------------------------------------------------------

  @spec unary(atom(), T.t(), T.t()) :: T.t()
  def unary(op, out, tensor) do
    Lower.check_complex(op, out, [tensor])

    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      Lower.push(st, Ops.unary(op, rx, out.type, out.shape))
    end)
  end

  @spec binary(atom(), T.t(), T.t(), T.t()) :: T.t()
  def binary(op, out, left, right) do
    Lower.check_complex(op, out, [left, right])

    run(out, [left, right], fn st, [l, r] ->
      {st, [rl, rr]} = conform_all(st, [l, r], out.shape)
      Lower.push(st, Ops.binary(op, rl, rr, out.type, out.shape))
    end)
  end

  @spec as_type(T.t(), T.t()) :: T.t()
  def as_type(out, tensor) do
    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      Lower.push(st, Ops.as_type(rx, out.type, out.shape))
    end)
  end

  @spec bitcast(T.t(), T.t()) :: T.t()
  def bitcast(out, tensor) do
    Lower.check_complex(:bitcast, out, [tensor])

    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      Lower.push(st, Ops.bitcast(rx, out.type, out.shape))
    end)
  end

  @spec select(T.t(), T.t(), T.t(), T.t()) :: T.t()
  def select(out, pred, on_true, on_false) do
    run(out, [pred, on_true, on_false], fn st, operands ->
      {st, [rp, rt, rf]} = conform_all(st, operands, out.shape)
      Lower.push(st, Ops.select(rp, rt, rf, out.type, out.shape))
    end)
  end

  @spec clip(T.t(), T.t(), T.t(), T.t()) :: T.t()
  def clip(out, tensor, min, max) do
    Lower.check_complex(:clip, out, [tensor, min, max])

    run(out, [tensor, min, max], fn st, operands ->
      {st, [rx, low, high]} = conform_all(st, operands, out.shape)
      Lower.push(st, Ops.clip(rx, low, high, out.type, out.shape))
    end)
  end

  # -- Dot and conv -----------------------------------------------------------

  @spec dot(T.t(), T.t(), [integer()], [integer()], T.t(), [integer()], [integer()]) :: T.t()
  def dot(out, left, contract_l, batch_l, right, contract_r, batch_r) do
    run(out, [left, right], fn st, [l, r] ->
      {st, rl} = cast(st, l, out.type)
      {st, rr} = cast(st, r, out.type)
      sides = {{left.shape, contract_l, batch_l}, {right.shape, contract_r, batch_r}}
      Contract.dot(st, &Lower.push/2, out, {rl, rr}, sides)
    end)
  end

  @spec conv(T.t(), T.t(), T.t(), keyword()) :: T.t()
  def conv(out, tensor, kernel, opts) do
    run(out, [tensor, kernel], fn st, [x, w] ->
      {st, rx} = cast(st, x, out.type)
      {st, rw} = cast(st, w, out.type)
      Contract.conv(st, &Lower.push/2, out, {{rx, tensor.shape}, {rw, kernel.shape}}, opts)
    end)
  end

  # -- Shape ------------------------------------------------------------------

  @spec alias_op(:reshape | :squeeze, T.t(), T.t()) :: T.t()
  def alias_op(op, out, tensor) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.alias_op(op, ref, out.type, out.shape))
    end)
  end

  @spec broadcast(T.t(), T.t(), [non_neg_integer()]) :: T.t()
  def broadcast(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.broadcast(ref, out.type, out.shape, axes))
    end)
  end

  @spec transpose(T.t(), T.t(), [non_neg_integer()]) :: T.t()
  def transpose(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.transpose(ref, out.type, out.shape, axes))
    end)
  end

  @spec reverse(T.t(), T.t(), [non_neg_integer()]) :: T.t()
  def reverse(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.reverse(ref, out.type, out.shape, axes))
    end)
  end

  @spec slice(T.t(), T.t(), list(), [non_neg_integer()], [pos_integer()]) :: T.t()
  def slice(out, tensor, starts, lengths, strides) do
    run(out, [tensor | Enum.map(starts, &start_tensor/1)], fn st, [{ref, _} | starts] ->
      Lower.push(
        st,
        Ops.slice(ref, refs(starts), out.type, out.shape, lengths, strides)
      )
    end)
  end

  @spec put_slice(T.t(), T.t(), list(), T.t()) :: T.t()
  def put_slice(out, tensor, starts, slice) do
    run(out, [tensor, slice | Enum.map(starts, &start_tensor/1)], fn st, [x, s | starts] ->
      {st, rx} = cast(st, x, out.type)
      {st, rs} = cast(st, s, out.type)
      Lower.push(st, Ops.put_slice(rx, rs, refs(starts), out.type, out.shape))
    end)
  end

  @spec pad(T.t(), T.t(), T.t(), list()) :: T.t()
  def pad(out, tensor, value, config) do
    run(out, [tensor, value], fn st, [x, v] ->
      {st, rx} = cast(st, x, out.type)
      {st, rv} = cast(st, v, out.type)
      Lower.push(st, Ops.pad(rx, rv, out.type, out.shape, config))
    end)
  end

  @spec join(:concatenate | :stack, T.t(), [T.t()], non_neg_integer()) :: T.t()
  def join(op, out, tensors, axis) do
    run(out, tensors, fn st, operands ->
      {st, refs} = cast_all(st, operands, out.type)
      Lower.push(st, Ops.join(op, refs, out.type, out.shape, axis))
    end)
  end

  @spec iota(T.t(), non_neg_integer() | nil) :: T.t()
  def iota(out, axis) do
    run(out, [], fn st, [] -> Lower.push(st, Ops.iota(out.type, out.shape, axis)) end)
  end

  @spec eye(T.t()) :: T.t()
  def eye(out) do
    run(out, [], fn st, [] -> Lower.push(st, Ops.eye(out.type, out.shape)) end)
  end

  # The number goes up as one scalar; a shape broadcasts it on the device. The
  # program holds no value, so one program serves every number.
  @spec constant(T.t(), number() | Complex.t() | atom()) :: T.t()
  def constant(%T{shape: {}} = out, number), do: scalar(out, number)

  def constant(out, number) do
    run(out, [scalar(%{out | shape: {}, names: []}, number)], fn st, [{ref, _}] ->
      Lower.push(st, Ops.broadcast(ref, out.type, out.shape, []))
    end)
  end

  defp scalar(out, number) do
    binary =
      %{out | shape: {}, names: []}
      |> then(&Nx.tensor(number, type: &1.type, backend: Nx.BinaryBackend))
      |> Nx.to_binary()

    NxCL.Backend.from_binary(out, binary, [])
  end

  # -- Reductions, scans, windows, sort -----------------------------------------

  @spec reduce(atom(), T.t(), T.t(), keyword()) :: T.t()
  def reduce(op, out, tensor, opts) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(
        st,
        Ops.reduce(op, ref, out.type, out.shape, Lower.axes(tensor, opts[:axes]))
      )
    end)
  end

  @spec arg_reduce(atom(), T.t(), T.t(), keyword()) :: T.t()
  def arg_reduce(op, out, tensor, opts) do
    axes = Lower.axes(tensor, opts[:axis] && [opts[:axis]])

    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.arg_reduce(op, ref, out.type, out.shape, opts[:tie_break], axes))
    end)
  end

  # A generic `reduce`: the function body is read as steps, as in the lowering.
  @spec reduce_fun(T.t(), T.t(), T.t(), keyword(), function()) :: T.t()
  def reduce_fun(out, tensor, acc, opts, fun) do
    steps = Steps.trace(fun, out.type, :reduce)
    axes = Lower.axes(tensor, opts[:axes])

    run(out, [tensor, acc], fn st, [{x, _}, {init, _} = acc_operand] ->
      if Lower.monoid?(steps) do
        # The initial accumulator folds in afterwards: `fun(total, init)`.
        {st, total} =
          Lower.push(st, Ops.reduce_steps([x], out.type, out.shape, steps, axes))

        {st, init} = conform(st, acc_operand, out.shape)
        fold(st, steps, [total, init], out)
      else
        Lower.push(st, Ops.reduce_steps([x, init], out.type, out.shape, steps, axes))
      end
    end)
  end

  # Emits `fun(total, init)` as elementwise instructions; the last is the result.
  defp fold(st, steps, refs, out) do
    {st, refs} =
      Enum.reduce(steps, {st, refs}, fn {op, a, b}, {st, refs} ->
        instr = Ops.binary(op, Enum.at(refs, a), Enum.at(refs, b), out.type, out.shape)
        {st, ref} = Lower.push(st, instr)
        {st, refs ++ [ref]}
      end)

    {st, List.last(refs)}
  end

  @spec scan(atom(), T.t(), T.t(), non_neg_integer(), boolean()) :: T.t()
  def scan(op, out, tensor, axis, reverse) do
    run(out, [tensor], fn st, [{ref, _}] ->
      Lower.push(st, Ops.scan(op, ref, out.type, out.shape, axis, reverse))
    end)
  end

  @spec window(atom(), T.t(), T.t(), tuple(), keyword()) :: T.t()
  def window(op, out, tensor, dims, opts) do
    run(out, [tensor], fn st, [{ref, _}] ->
      instr =
        Ops.window(
          op,
          ref,
          out.type,
          out.shape,
          dims,
          opts[:strides],
          opts[:window_dilations],
          opts[:padding]
        )

      Lower.push(st, instr)
    end)
  end

  @spec window_reduce(T.t(), T.t(), T.t(), tuple(), keyword(), function()) :: T.t()
  def window_reduce(out, tensor, acc, dims, opts, fun) do
    steps = Steps.trace(fun, out.type, :window_reduce)
    window = {dims, opts[:strides], opts[:window_dilations], opts[:padding]}

    run(out, [tensor, acc], fn st, [{x, _}, {init, _}] ->
      Lower.push(st, Ops.window_reduce(x, init, out.type, out.shape, steps, window))
    end)
  end

  @spec window_scatter(atom(), T.t(), T.t(), T.t(), T.t(), tuple(), keyword()) :: T.t()
  def window_scatter(op, out, tensor, source, init, dims, opts) do
    run(out, [tensor, source, init], fn st, [{x, _}, {s, _}, {i, _}] ->
      window = {dims, opts[:strides], opts[:padding]}
      Lower.push(st, Ops.window_scatter(op, x, s, i, out.type, out.shape, window))
    end)
  end

  @spec sort(:sort | :argsort, T.t(), T.t(), keyword()) :: T.t()
  def sort(op, out, tensor, opts) do
    run(out, [tensor], fn st, [{ref, _}] ->
      instr = Ops.sort(op, ref, out.type, out.shape, opts[:axis] || 0, opts[:direction])
      Lower.push(st, instr)
    end)
  end

  # -- Indexing ---------------------------------------------------------------

  @spec gather(T.t(), T.t(), T.t(), keyword()) :: T.t()
  def gather(out, tensor, indices, opts) do
    run(out, [tensor, indices], fn st, [{x, _}, {i, _}] ->
      Lower.push(st, Ops.gather(x, i, out.type, out.shape, opts[:axes]))
    end)
  end

  @spec indexed(:indexed_add | :indexed_put, T.t(), T.t(), T.t(), T.t(), keyword()) :: T.t()
  def indexed(op, out, tensor, indices, updates, opts) do
    run(out, [tensor, indices, updates], fn st, [x, {i, _}, u] ->
      {st, rx} = cast(st, x, out.type)
      {st, ru} = cast(st, u, out.type)
      Lower.push(st, Ops.indexed(op, {rx, i, ru}, out.type, out.shape, opts[:axes]))
    end)
  end

  # -- Linear algebra and FFT ---------------------------------------------------

  @spec triangular_solve(T.t(), T.t(), T.t(), keyword()) :: T.t()
  def triangular_solve(out, a, b, opts) do
    run(out, [a, b], fn st, [oa, ob] ->
      {st, ra} = cast(st, oa, out.type)
      {st, rb} = cast(st, ob, out.type)
      Lower.push(st, Ops.triangular_solve(ra, rb, out.type, out.shape, opts))
    end)
  end

  # Nx pads or crops the axis to `:length`; so does the lowering, then one transform.
  @spec fft(:fft | :ifft, T.t(), T.t(), keyword()) :: T.t()
  def fft(op, out, tensor, opts) do
    axis = opts[:axis]

    run(out, [tensor], fn st, [x] ->
      {st, rx} = cast(st, x, out.type)
      {st, rx} = Lower.fit_axis(st, rx, tensor.shape, out.type, out.shape, axis)
      Lower.push(st, Ops.fft(op, rx, out.type, out.shape, axis))
    end)
  end

  # -- Blocks -------------------------------------------------------------------

  @doc """
  Runs an `Nx.Block`. The decision table of `NxCL.Compiler.Lower.Block` picks the way: a
  native instruction or the block's default.
  """
  @spec block(struct(), Nx.Container.t(), [T.t()], function()) :: Nx.Container.t()
  def block(struct, out, args, fun) do
    case Block.decision(struct) do
      :native -> native_block(struct, out, args)
      :default -> with_device_default(fn -> apply(fun, [struct | args]) end)
    end
  end

  # The default may build constants and loop state (a `while` init, an `iota`). They
  # must live on the device with the arguments, not on the caller's default backend.
  defp with_device_default(fun), do: Nx.with_default_backend(NxCL.Backend, fun)

  defp native_block(%Nx.Block.LogicalNot{}, out, [x]), do: unary_block(:logical_not, out, x)
  defp native_block(%Nx.Block.Phase{}, out, [x]), do: unary_block(:phase, out, x)

  defp native_block(%Nx.Block.LinAlg.Cholesky{}, out, [x]) do
    run(out, [x], fn st, [x] ->
      {st, rx} = cast(st, x, out.type)
      Lower.push(st, Ops.cholesky(rx, out.type, out.shape))
    end)
  end

  defp native_block(%{__struct__: mod, axis: axis, reverse: reverse}, out, [x]) do
    case Map.fetch(Lower.scan_blocks(), mod) do
      {:ok, op} -> scan(op, out, x, axis, reverse)
      :error -> raise ArgumentError, "does not yet lower op #{inspect(mod)} (a native block)"
    end
  end

  defp unary_block(op, out, x) do
    run(out, [x], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      Lower.push(st, Ops.unary(op, rx, out.type, out.shape))
    end)
  end
end
