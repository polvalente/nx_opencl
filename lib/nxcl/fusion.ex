defmodule NxCL.Fusion do
  @moduledoc """
  Groups elementwise instructions of a `NxCL.Program` into `:fused` instructions.

  A group holds connected elementwise instructions (including `as_type`, `select`
  and `clip`) of one output shape. It has one result, the last step. A value
  that something outside the group reads stays the result of its own group and
  later groups read it as an input. A `broadcast` of a group input is not a
  step: the group reads the input through a broadcast index mapping, and a
  broadcast of a scalar constant is the constant itself. The pass never joins
  instructions across a `cond`, `while` or `host_call`, and never changes the
  instructions inside one (it fuses each sub-program on its own).

  A `:fused` instruction carries the attrs `[maps, steps]`:

    * `maps` has one entry per operand: `:direct` (the operand has the output
      shape) or `{:broadcast, axes}` (operand axis `j` fills output axis `axes[j]`).
    * `steps` is a list of `{op, srcs, out_type, attrs}`. A source is `{:in, k}`,
      the k-th operand, or `{:step, j}`, an earlier step. The last step is the
      result. Scalar constants are operands; the native side writes them into
      the kernel as literals.

  The pass emits no OpenCL C. The native side builds the kernel
  (`codegen/fused.rs`).

  Options:

    * `:fusion` - `false` returns the program unchanged.
    * `:max_args` - most kernel arguments a group may need. Default: the
      device's limit.
    * `:split` - `false` sends an oversized group as it is, so the native
      limit error shows. For tests.
  """

  alias NxCL.Program
  alias NxCL.Program.Builder
  alias NxCL.Program.Instruction
  alias NxCL.Program.SubProgram

  @control [:cond, :while, :host_call]
  @fusible Builder.unary_ops() ++ Builder.binary_ops() ++ [:as_type, :select, :clip]

  @doc "Ops a group may hold."
  @spec fusible_ops() :: [atom()]
  def fusible_ops(), do: @fusible

  @doc "Fuses `program`. See the module doc for the options."
  @spec fuse(Program.t(), keyword()) :: Program.t()
  def fuse(%Program{} = program, opts \\ []) do
    if Keyword.get(opts, :fusion, true) do
      env = %{max_args: max_args(opts), split: Keyword.get(opts, :split, true)}
      {instructions, outputs} = fuse_scope(program.instructions, program.outputs, env)

      %{
        program
        | instructions: instructions,
          outputs: outputs,
          ref_counts: NxCL.Lower.ref_counts(instructions, outputs)
      }
    else
      program
    end
  end

  @doc "The kernel argument limit of the device, as Rust reports it."
  @spec device_max_args() :: pos_integer()
  def device_max_args() do
    info = NxCL.Native.device_info(NxCL.Executor.ctx())
    info |> List.keyfind("max_kernel_args", 0) |> elem(1) |> String.to_integer()
  end

  defp max_args(opts), do: Keyword.get_lazy(opts, :max_args, &device_max_args/0)

  # -- One scope ---------------------------------------------------------------

  defp fuse_scope(instructions, outputs, env) do
    instructions = Enum.map(instructions, &fuse_subprograms(&1, env))
    instrs = List.to_tuple(instructions)
    ctx = context(instructions, outputs, instrs, env)

    groups =
      (tuple_size(instrs) - 1)..0//-1
      |> Enum.reduce(%{groups: %{}, of: %{}}, &visit(&1, &2, ctx))
      |> Map.fetch!(:groups)
      |> Map.values()
      |> Enum.filter(&(length(&1.members) >= 2))

    if groups == [] do
      {instructions, outputs}
    else
      rebuild(instructions, outputs, groups, ctx)
    end
  end

  defp fuse_subprograms(%Instruction{subprograms: []} = instr, _env), do: instr

  defp fuse_subprograms(%Instruction{subprograms: subs} = instr, env) do
    subs =
      Enum.map(subs, fn %SubProgram{} = sub ->
        {instructions, outputs} = fuse_scope(sub.instructions, sub.outputs, env)
        %{sub | instructions: instructions, outputs: outputs}
      end)

    %{instr | subprograms: subs}
  end

  defp context(instructions, outputs, instrs, env) do
    consumers =
      instructions
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {instr, c}, acc ->
        Enum.reduce(instr.operands, acc, fn
          {:result, m}, acc -> Map.update(acc, m, [c], &[c | &1])
          _ref, acc -> acc
        end)
      end)

    # Index of the last control instruction before each position, or -1.
    {floors, _} =
      instructions
      |> Enum.with_index()
      |> Enum.map_reduce(-1, fn {instr, i}, last ->
        {last, if(instr.op in @control, do: i, else: last)}
      end)

    %{
      instrs: instrs,
      consumers: consumers,
      output_set: for({:result, m} <- outputs, into: MapSet.new(), do: m),
      floors: List.to_tuple(floors),
      env: env
    }
  end

  # -- Grouping ----------------------------------------------------------------

  # Walks back from the last instruction. An elementwise instruction joins the
  # group that reads it when every reader is in that group; otherwise it
  # roots a group of its own.
  defp visit(i, state, ctx) do
    instr = elem(ctx.instrs, i)

    if instr.op in @fusible do
      case join_target(i, instr, state, ctx) do
        {:join, gid, inputs} -> join(state, i, gid, inputs)
        :none -> start(state, i, instr, ctx)
      end
    else
      state
    end
  end

  defp join_target(i, instr, state, ctx) do
    readers = Map.get(ctx.consumers, i, [])

    with true <- readers != [] and not MapSet.member?(ctx.output_set, i),
         [gid] <- readers |> Enum.map(&Map.get(state.of, &1)) |> Enum.uniq(),
         true <- gid != nil,
         group = Map.fetch!(state.groups, gid),
         true <- group.shape == instr.out_shape,
         true <- elem(ctx.floors, group.root) < i,
         inputs = swap_inputs(group.inputs, i, instr, group.shape, ctx),
         true <- fits?(inputs, tuple_size(group.shape), ctx) do
      {:join, gid, inputs}
    else
      _ -> :none
    end
  end

  defp swap_inputs(inputs, i, instr, shape, ctx) do
    new = Enum.map(instr.operands, &input_key(&1, shape, ctx.instrs))
    inputs |> MapSet.delete({:result, i}) |> MapSet.union(MapSet.new(new))
  end

  defp join(state, i, gid, inputs) do
    group = Map.fetch!(state.groups, gid)
    group = %{group | members: [i | group.members], inputs: inputs}
    %{state | groups: Map.put(state.groups, gid, group), of: Map.put(state.of, i, gid)}
  end

  defp start(state, i, instr, ctx) do
    inputs =
      instr.operands
      |> Enum.map(&input_key(&1, instr.out_shape, ctx.instrs))
      |> MapSet.new()

    group = %{id: i, root: i, shape: instr.out_shape, members: [i], inputs: inputs}
    %{state | groups: Map.put(state.groups, i, group), of: Map.put(state.of, i, i)}
  end

  # What a group reads for operand `ref`: a broadcast of the group's shape is
  # read through its mapping, or is the constant itself.
  defp input_key({:result, m} = ref, shape, instrs) do
    case elem(instrs, m) do
      %Instruction{op: :broadcast, out_shape: ^shape, operands: [{:const, _} = src]} ->
        src

      %Instruction{op: :broadcast, out_shape: ^shape, operands: [src], attrs: axes} ->
        {:bcast, src, axes}

      _ ->
        ref
    end
  end

  defp input_key(ref, _shape, _instrs), do: ref

  defp fits?(_inputs, _rank, %{env: %{split: false}}), do: true

  defp fits?(inputs, rank, %{env: %{max_args: max}}), do: arg_count(inputs, rank) <= max

  # Mirrors `fused::arg_count` in `codegen/fused.rs`.
  defp arg_count(inputs, rank) do
    buffers = Enum.count(inputs, &(not match?({:const, _}, &1)))
    bcasts = Enum.count(inputs, &match?({:bcast, _, _}, &1))
    index = if bcasts > 0, do: rank * (1 + bcasts), else: 0
    buffers + 1 + index
  end

  # -- Rebuilding --------------------------------------------------------------

  defp rebuild(instructions, outputs, groups, ctx) do
    fused_at = Map.new(groups, &{&1.root, build(&1, ctx)})
    absorbed = groups |> Enum.flat_map(&List.delete(&1.members, &1.root)) |> MapSet.new()

    kept =
      instructions
      |> Enum.with_index()
      |> Enum.reject(fn {_instr, i} -> MapSet.member?(absorbed, i) end)
      |> Enum.map(fn {instr, i} -> {i, Map.get(fused_at, i, instr)} end)

    kept = drop_dead_broadcasts(kept, outputs)
    index = kept |> Enum.with_index() |> Map.new(fn {{old, _}, new} -> {old, new} end)
    remap = &remap(&1, index)

    {Enum.map(kept, fn {_old, instr} -> %{instr | operands: Enum.map(instr.operands, remap)} end),
     Enum.map(outputs, remap)}
  end

  # A broadcast every reader of which fused away has no readers left.
  defp drop_dead_broadcasts(kept, outputs) do
    used =
      for {_old, instr} <- kept, {:result, m} <- instr.operands, into: MapSet.new(), do: m

    used =
      Enum.reduce(outputs, used, fn
        {:result, m}, acc -> MapSet.put(acc, m)
        _, acc -> acc
      end)

    Enum.reject(kept, fn {old, instr} -> instr.op == :broadcast and old not in used end)
  end

  defp remap({:result, m}, index), do: {:result, Map.fetch!(index, m)}
  defp remap({:result, m, k}, index), do: {:result, Map.fetch!(index, m), k}
  defp remap(ref, _index), do: ref

  defp build(%{members: members, root: root, shape: shape}, ctx) do
    members = Enum.sort(members)
    steps_at = members |> Enum.with_index() |> Map.new()

    {steps, {keys, _}} =
      Enum.map_reduce(members, {[], %{}}, fn m, acc ->
        instr = elem(ctx.instrs, m)
        {srcs, acc} = Enum.map_reduce(instr.operands, acc, &source(&1, &2, steps_at, shape, ctx))
        {{instr.op, srcs, instr.out_type, instr.attrs}, acc}
      end)

    keys = Enum.reverse(keys)
    root_instr = elem(ctx.instrs, root)

    %Instruction{
      op: :fused,
      operands: Enum.map(keys, &key_operand/1),
      out_type: root_instr.out_type,
      out_shape: shape,
      attrs: [Enum.map(keys, &key_map/1), steps]
    }
  end

  # A step reads an earlier member, or a group input (added on first use).
  defp source({:result, r} = ref, acc, steps_at, shape, ctx) do
    case steps_at do
      %{^r => step} -> {{:step, step}, acc}
      _ -> input_source(ref, acc, shape, ctx)
    end
  end

  defp source(ref, acc, _steps_at, shape, ctx), do: input_source(ref, acc, shape, ctx)

  defp input_source(ref, acc, shape, ctx) do
    {pos, acc} = input_pos(acc, input_key(ref, shape, ctx.instrs))
    {{:in, pos}, acc}
  end

  defp input_pos({keys, seen} = acc, key) do
    case seen do
      %{^key => pos} -> {pos, acc}
      _ -> {map_size(seen), {[key | keys], Map.put(seen, key, map_size(seen))}}
    end
  end

  defp key_operand({:bcast, src, _axes}), do: src
  defp key_operand(ref), do: ref

  defp key_map({:bcast, _src, axes}), do: {:broadcast, axes}
  defp key_map(_ref), do: :direct
end
