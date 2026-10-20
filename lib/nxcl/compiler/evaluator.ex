defmodule NxCL.Compiler.Evaluator do
  @moduledoc false
  # One-op programs for `NxCL.Backend`. Each backend callback builds a small
  # program with `NxCL.Compiler.Ops` and runs it through `NxCL.Compiler.Executor`.
  # The tensors are inputs already on the device. A call may emit a broadcast or
  # a cast before the op. An op the lowering cannot take raises `ArgumentError`.

  import NxCL.Compiler.Lower.Shared,
    only: [axes: 2, check_complex: 3, fit_axis: 6, monoid?: 1, push: 2]

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Evaluator.Steps
  alias NxCL.Compiler.Executor
  alias NxCL.Compiler.Lower.Block
  alias NxCL.Compiler.Lower.Contract
  alias NxCL.Compiler.Lower.Shared
  alias NxCL.Compiler.Ops
  alias NxCL.Compiler.Program
  alias NxCL.Devices

  @doc """
  Builds and runs a program. `out` is the result template, `tensors` the
  operands. `build` gets the builder state and the operands as `{ref, tensor}`
  pairs and returns `{state, ref | [ref]}`.

  The program runs on the operands' device; operands on two contexts raise
  `ArgumentError`. With no operand on a device, `opts[:device]` decides
  (default 0).
  """
  @spec run(
          Nx.Container.t(),
          [Shared.tensor()],
          (Shared.builder(), [{term(), Shared.tensor()}] -> {Shared.builder(), term()}),
          keyword()
        ) :: Nx.Container.t()
  def run(out, tensors, build, opts \\ []) do
    device_opts = device_opts(tensors, opts)
    tensors = Enum.map(tensors, &device(&1, device_opts))
    operands = tensors |> Enum.with_index() |> Enum.map(fn {t, i} -> {{:input, i}, t} end)

    {st, out_refs} = build.(%{instrs: [], constants: []}, operands)
    program = program(st, tensors, out, List.wrap(out_refs))

    buffers =
      Executor.run_buffers(program, Enum.map(tensors, &{:buffer, &1.data.ref}), device_opts)

    ctx = Devices.fetch!(device_opts).ctx

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
      outputs: out_refs,
      output_template: Composite.traverse(out, &Nx.to_template/1),
      ref_counts: Shared.ref_counts(instructions, out_refs)
    }
  end

  @doc """
  The `[device: key]` option for a run over `tensors`: their device, which must
  be one context, else `opts[:device]` (default 0). Raises `ArgumentError` on a
  mismatch.
  """
  @spec device_opts([Shared.tensor()], keyword()) :: [device: Devices.key()]
  def device_opts(tensors, opts \\ []) do
    ctxs = for %T{data: %NxCL.Backend{ctx: ctx}} <- tensors, uniq: true, do: ctx

    case ctxs do
      [] -> [device: Devices.key(opts)]
      [ctx] -> [device: device_key!(ctx, opts)]
      _many -> raise ArgumentError, mixed_contexts_message(ctxs)
    end
  end

  defp device_key!(ctx, opts) do
    case Devices.for_ctx(ctx) do
      %{key: key} ->
        if Keyword.has_key?(opts, :device) and Devices.key(opts) != key do
          raise ArgumentError,
                "the operands live on device #{inspect(key)}, not on the requested " <>
                  "device #{inspect(Devices.key(opts))}; move them with Nx.backend_transfer/3"
        end

        key

      nil ->
        raise ArgumentError, "a tensor lives on an OpenCL context that NxCL.Devices does not know"
    end
  end

  defp mixed_contexts_message(ctxs) do
    keys = for ctx <- ctxs, do: with(%{key: key} <- Devices.for_ctx(ctx), do: key)

    "the operands live on different OpenCL contexts (devices #{inspect(keys)}); " <>
      "move them to one device with Nx.backend_transfer/3 first"
  end

  defp device(%T{data: %NxCL.Backend{}} = t, _device_opts), do: t

  defp device(%T{} = t, device_opts),
    do: NxCL.Backend.from_binary(t, Nx.to_binary(t), device_opts)

  defp conform(st, {ref, %T{shape: shape}}, shape), do: {st, ref}

  defp conform(st, {ref, %T{shape: from, type: type}}, shape) do
    rank = tuple_size(shape)
    axes = Enum.to_list((rank - tuple_size(from))..(rank - 1)//1)
    push(st, Ops.broadcast(ref, type, shape, axes))
  end

  defp conform_all(st, operands, shape), do: map_state(st, operands, &conform(&1, &2, shape))

  defp cast(st, {ref, %T{type: type}}, type), do: {st, ref}
  defp cast(st, {ref, %T{} = x}, type), do: push(st, Ops.as_type(ref, type, x.shape))

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

  # An integer start becomes an `s64` operand, so the program caches by shape.
  defp start_tensor(%T{} = t), do: t

  defp start_tensor(i) when is_integer(i),
    do: Nx.tensor(i, type: {:s, 64}, backend: Nx.BinaryBackend)

  @spec unary(atom(), Shared.tensor(), Shared.tensor()) :: Shared.tensor()
  def unary(op, out, tensor) do
    check_complex(op, out, [tensor])

    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      push(st, Ops.unary(op, rx, out.type, out.shape))
    end)
  end

  @spec binary(atom(), Shared.tensor(), Shared.tensor(), Shared.tensor()) :: Shared.tensor()
  def binary(op, out, left, right) do
    check_complex(op, out, [left, right])

    run(out, [left, right], fn st, [l, r] ->
      {st, [rl, rr]} = conform_all(st, [l, r], out.shape)
      push(st, Ops.binary(op, rl, rr, out.type, out.shape))
    end)
  end

  @spec as_type(Shared.tensor(), Shared.tensor()) :: Shared.tensor()
  def as_type(out, tensor) do
    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      push(st, Ops.as_type(rx, out.type, out.shape))
    end)
  end

  @spec bitcast(Shared.tensor(), Shared.tensor()) :: Shared.tensor()
  def bitcast(out, tensor) do
    check_complex(:bitcast, out, [tensor])

    run(out, [tensor], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      push(st, Ops.bitcast(rx, out.type, out.shape))
    end)
  end

  @spec select(Shared.tensor(), Shared.tensor(), Shared.tensor(), Shared.tensor()) ::
          Shared.tensor()
  def select(out, pred, on_true, on_false) do
    run(out, [pred, on_true, on_false], fn st, operands ->
      {st, [rp, rt, rf]} = conform_all(st, operands, out.shape)
      push(st, Ops.select(rp, rt, rf, out.type, out.shape))
    end)
  end

  @spec clip(Shared.tensor(), Shared.tensor(), Shared.tensor(), Shared.tensor()) ::
          Shared.tensor()
  def clip(out, tensor, min, max) do
    check_complex(:clip, out, [tensor, min, max])

    run(out, [tensor, min, max], fn st, operands ->
      {st, [rx, low, high]} = conform_all(st, operands, out.shape)
      push(st, Ops.clip(rx, low, high, out.type, out.shape))
    end)
  end

  @spec dot(
          Shared.tensor(),
          Shared.tensor(),
          [integer()],
          [integer()],
          Shared.tensor(),
          [integer()],
          [integer()]
        ) :: Shared.tensor()
  def dot(out, left, contract_l, batch_l, right, contract_r, batch_r) do
    run(out, [left, right], fn st, [l, r] ->
      {st, rl} = cast(st, l, out.type)
      {st, rr} = cast(st, r, out.type)
      sides = {{left.shape, contract_l, batch_l}, {right.shape, contract_r, batch_r}}
      Contract.dot(st, out, {rl, rr}, sides)
    end)
  end

  @spec conv(Shared.tensor(), Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def conv(out, tensor, kernel, opts) do
    run(out, [tensor, kernel], fn st, [x, w] ->
      {st, rx} = cast(st, x, out.type)
      {st, rw} = cast(st, w, out.type)
      Contract.conv(st, out, {{rx, tensor.shape}, {rw, kernel.shape}}, opts)
    end)
  end

  @spec alias_op(:reshape | :squeeze, Shared.tensor(), Shared.tensor()) :: Shared.tensor()
  def alias_op(op, out, tensor) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.alias_op(op, ref, out.type, out.shape))
    end)
  end

  @spec broadcast(Shared.tensor(), Shared.tensor(), [non_neg_integer()]) :: Shared.tensor()
  def broadcast(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.broadcast(ref, out.type, out.shape, axes))
    end)
  end

  @spec transpose(Shared.tensor(), Shared.tensor(), [non_neg_integer()]) :: Shared.tensor()
  def transpose(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.transpose(ref, out.type, out.shape, axes))
    end)
  end

  @spec reverse(Shared.tensor(), Shared.tensor(), [non_neg_integer()]) :: Shared.tensor()
  def reverse(out, tensor, axes) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.reverse(ref, out.type, out.shape, axes))
    end)
  end

  @spec slice(Shared.tensor(), Shared.tensor(), list(), [non_neg_integer()], [pos_integer()]) ::
          Shared.tensor()
  def slice(out, tensor, starts, lengths, strides) do
    run(out, [tensor | Enum.map(starts, &start_tensor/1)], fn st, [{ref, _} | starts] ->
      push(
        st,
        Ops.slice(ref, refs(starts), out.type, out.shape, lengths, strides)
      )
    end)
  end

  @spec put_slice(Shared.tensor(), Shared.tensor(), list(), Shared.tensor()) :: Shared.tensor()
  def put_slice(out, tensor, starts, slice) do
    run(out, [tensor, slice | Enum.map(starts, &start_tensor/1)], fn st, [x, s | starts] ->
      {st, rx} = cast(st, x, out.type)
      {st, rs} = cast(st, s, out.type)
      push(st, Ops.put_slice(rx, rs, refs(starts), out.type, out.shape))
    end)
  end

  @spec pad(Shared.tensor(), Shared.tensor(), Shared.tensor(), list()) :: Shared.tensor()
  def pad(out, tensor, value, config) do
    run(out, [tensor, value], fn st, [x, v] ->
      {st, rx} = cast(st, x, out.type)
      {st, rv} = cast(st, v, out.type)
      push(st, Ops.pad(rx, rv, out.type, out.shape, config))
    end)
  end

  @spec join(:concatenate | :stack, Shared.tensor(), [Shared.tensor()], non_neg_integer()) ::
          Shared.tensor()
  def join(op, out, tensors, axis) do
    run(out, tensors, fn st, operands ->
      {st, refs} = cast_all(st, operands, out.type)
      push(st, Ops.join(op, refs, out.type, out.shape, axis))
    end)
  end

  @spec iota(Shared.tensor(), non_neg_integer() | nil, keyword()) :: Shared.tensor()
  def iota(out, axis, opts \\ []) do
    run(out, [], fn st, [] -> push(st, Ops.iota(out.type, out.shape, axis)) end, opts)
  end

  @spec eye(Shared.tensor(), keyword()) :: Shared.tensor()
  def eye(out, opts \\ []) do
    run(out, [], fn st, [] -> push(st, Ops.eye(out.type, out.shape)) end, opts)
  end

  # The program holds no value, so one program serves every number.
  @spec constant(Shared.tensor(), number() | Complex.t() | atom(), keyword()) :: Shared.tensor()
  def constant(out, number, opts \\ [])

  def constant(%T{shape: {}} = out, number, opts) do
    scalar = scalar(out, number)
    NxCL.Backend.from_binary(out, Nx.to_binary(scalar), device_opts([], opts))
  end

  def constant(out, number, opts) do
    run(
      out,
      [scalar(out, number)],
      fn st, [{ref, _}] -> push(st, Ops.broadcast(ref, out.type, out.shape, [])) end,
      opts
    )
  end

  defp scalar(out, number), do: Nx.tensor(number, type: out.type, backend: Nx.BinaryBackend)

  @spec reduce(atom(), Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def reduce(op, out, tensor, opts) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(
        st,
        Ops.reduce(op, ref, out.type, out.shape, axes(tensor, opts[:axes]))
      )
    end)
  end

  @spec arg_reduce(atom(), Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def arg_reduce(op, out, tensor, opts) do
    axes = axes(tensor, opts[:axis] && [opts[:axis]])

    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.arg_reduce(op, ref, out.type, out.shape, opts[:tie_break], axes))
    end)
  end

  @spec reduce_fun(Shared.tensor(), Shared.tensor(), Shared.tensor(), keyword(), function()) ::
          Shared.tensor()
  def reduce_fun(out, tensor, acc, opts, fun) do
    steps = Steps.trace(fun, out.type, :reduce)
    axes = axes(tensor, opts[:axes])

    # The accumulator is a device tensor, not a literal, so the identity-init
    # shortcut of `NxCL.Compiler.Lower` does not apply: a monoid body always
    # folds the accumulator in after the reduction.
    run(out, [tensor, acc], fn st, [{x, _}, {init, _} = acc_operand] ->
      if monoid?(steps) do
        # The result is `fun(total, init)`.
        {st, total} =
          push(st, Ops.reduce_steps([x], out.type, out.shape, steps, axes))

        {st, init} = conform(st, acc_operand, out.shape)
        Shared.fold_steps(st, steps, [total, init], out.type, out.shape)
      else
        push(st, Ops.reduce_steps([x, init], out.type, out.shape, steps, axes))
      end
    end)
  end

  @spec scan(atom(), Shared.tensor(), Shared.tensor(), non_neg_integer(), boolean()) ::
          Shared.tensor()
  def scan(op, out, tensor, axis, reverse) do
    run(out, [tensor], fn st, [{ref, _}] ->
      push(st, Ops.scan(op, ref, out.type, out.shape, axis, reverse))
    end)
  end

  @spec window(atom(), Shared.tensor(), Shared.tensor(), tuple(), keyword()) :: Shared.tensor()
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

      push(st, instr)
    end)
  end

  @spec window_reduce(
          Shared.tensor(),
          Shared.tensor(),
          Shared.tensor(),
          tuple(),
          keyword(),
          function()
        ) :: Shared.tensor()
  def window_reduce(out, tensor, acc, dims, opts, fun) do
    steps = Steps.trace(fun, out.type, :window_reduce)
    window = {dims, opts[:strides], opts[:window_dilations], opts[:padding]}

    run(out, [tensor, acc], fn st, [{x, _}, {init, _}] ->
      push(st, Ops.window_reduce(x, init, out.type, out.shape, steps, window))
    end)
  end

  @spec window_scatter(
          atom(),
          Shared.tensor(),
          Shared.tensor(),
          Shared.tensor(),
          Shared.tensor(),
          tuple(),
          keyword()
        ) :: Shared.tensor()
  def window_scatter(op, out, tensor, source, init, dims, opts) do
    run(out, [tensor, source, init], fn st, [{x, _}, {s, _}, {i, _}] ->
      window = {dims, opts[:strides], opts[:padding]}
      push(st, Ops.window_scatter(op, x, s, i, out.type, out.shape, window))
    end)
  end

  @spec sort(:sort | :argsort, Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def sort(op, out, tensor, opts) do
    run(out, [tensor], fn st, [{ref, _}] ->
      instr = Ops.sort(op, ref, out.type, out.shape, opts[:axis] || 0, opts[:direction])
      push(st, instr)
    end)
  end

  @spec gather(Shared.tensor(), Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def gather(out, tensor, indices, opts) do
    run(out, [tensor, indices], fn st, [{x, _}, {i, _}] ->
      push(st, Ops.gather(x, i, out.type, out.shape, opts[:axes]))
    end)
  end

  @spec indexed(
          :indexed_add | :indexed_put,
          Shared.tensor(),
          Shared.tensor(),
          Shared.tensor(),
          Shared.tensor(),
          keyword()
        ) :: Shared.tensor()
  def indexed(op, out, tensor, indices, updates, opts) do
    run(out, [tensor, indices, updates], fn st, [x, {i, _}, u] ->
      {st, rx} = cast(st, x, out.type)
      {st, ru} = cast(st, u, out.type)
      push(st, Ops.indexed(op, {rx, i, ru}, out.type, out.shape, opts[:axes]))
    end)
  end

  @spec triangular_solve(Shared.tensor(), Shared.tensor(), Shared.tensor(), keyword()) ::
          Shared.tensor()
  def triangular_solve(out, a, b, opts) do
    run(out, [a, b], fn st, [oa, ob] ->
      {st, ra} = cast(st, oa, out.type)
      {st, rb} = cast(st, ob, out.type)
      push(st, Ops.triangular_solve(ra, rb, out.type, out.shape, opts))
    end)
  end

  # Nx pads or crops the axis to `:length` first.
  @spec fft(:fft | :ifft, Shared.tensor(), Shared.tensor(), keyword()) :: Shared.tensor()
  def fft(op, out, tensor, opts) do
    axis = opts[:axis]

    run(out, [tensor], fn st, [x] ->
      {st, rx} = cast(st, x, out.type)
      {st, rx} = fit_axis(st, rx, tensor.shape, out.type, out.shape, axis)
      push(st, Ops.fft(op, rx, out.type, out.shape, axis))
    end)
  end

  @doc """
  Runs an `Nx.Block` as a native instruction or through its default.
  """
  @spec block(struct(), Nx.Container.t(), [Shared.tensor()], function()) :: Nx.Container.t()
  def block(struct, out, args, fun) do
    case Block.decision(struct) do
      :native -> native_block(struct, out, args)
      :default -> with_device_default(device_opts(args), fn -> apply(fun, [struct | args]) end)
    end
  end

  # The default may build constants and loop state; they must live on the
  # device with the arguments.
  defp with_device_default(device_opts, fun),
    do: Nx.with_default_backend({NxCL.Backend, device_opts}, fun)

  defp native_block(%Nx.Block.LogicalNot{}, out, [x]), do: unary_block(:logical_not, out, x)
  defp native_block(%Nx.Block.Phase{}, out, [x]), do: unary_block(:phase, out, x)

  defp native_block(%Nx.Block.LinAlg.Cholesky{}, out, [x]) do
    run(out, [x], fn st, [x] ->
      {st, rx} = cast(st, x, out.type)
      push(st, Ops.cholesky(rx, out.type, out.shape))
    end)
  end

  defp native_block(%{__struct__: mod, axis: axis, reverse: reverse}, out, [x]) do
    case Map.fetch(Shared.scan_blocks(), mod) do
      {:ok, op} -> scan(op, out, x, axis, reverse)
      :error -> raise ArgumentError, "does not yet lower op #{inspect(mod)} (a native block)"
    end
  end

  defp unary_block(op, out, x) do
    run(out, [x], fn st, [x] ->
      {st, [rx]} = conform_all(st, [x], out.shape)
      push(st, Ops.unary(op, rx, out.type, out.shape))
    end)
  end
end
