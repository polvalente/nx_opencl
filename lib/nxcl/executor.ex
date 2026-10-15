defmodule NxCL.Executor do
  @moduledoc false
  # Runs an `NxCL.Program` on the device: upload the inputs, run the program
  # natively, read the outputs, rebuild the container.
  #
  # A program with no `cond`, `while` or `host_call` goes to the native side whole.
  # Otherwise `NxCL.Executor.Plan` cuts it at the control instructions: each
  # straight-line segment runs natively as its own program, and this module runs
  # the `cond`, `while` and `host_call` steps itself (`NxCL.Executor.Host`).
  # Buffers stay on the device between steps, in a table keyed by ref. A step drops the buffers it was the last to read.
  #
  # Wire term sent to `NxCL.Native.compile_program/2`:
  #
  #     {[input], [capture], [const], [instr], [out_ref]}
  #     input = {type, shape} | nil        (nil: the function never reads it)
  #     capture = {type, shape}
  #     const = {type, value}              (value: a number, :nan, :infinity, :neg_infinity,
  #                                         or a {re, im} pair for a complex type)
  #     instr = {op, [ref], out_type, out_shape, [attr]}
  #     ref = {:input | :capture | :const | :result, index}
  #
  # A `batched_gemm` carries `[b, m, n, k]` as attrs. Buffers go to `eval_program/2` in the
  # order inputs, captures, constants; constants go as `nil`, because Rust holds
  # their values. The native side generates the kernel for each instruction from
  # the op, the types and the shapes.

  alias Nx.Defn.Composite
  alias NxCL.Executor.Host
  alias NxCL.Executor.Plan
  alias NxCL.Executor.Stats
  alias NxCL.Program
  alias NxCL.Program.Instruction

  @doc """
  Runs `program` on `args`, the flat list of lazy argument functions that
  `Nx.Defn.Compiler` passes. Returns the output container.
  """
  @spec run(Program.t(), [(-> Nx.Tensor.t())]) :: Nx.Container.t()
  def run(%Program{} = program, args) do
    program |> run_buffers(args) |> then(&read_outputs(program, &1))
  end

  @doc """
  Runs `program` and returns one device buffer per output leaf, nothing read
  back. An argument is a lazy function, or `{:buffer, ref}` for a tensor that is
  already on the device (the eager backend passes these).
  """
  @spec run_buffers(Program.t(), [(-> Nx.Tensor.t()) | {:buffer, reference()}]) :: [reference()]
  def run_buffers(%Program{} = program, args) do
    ctx = ctx()

    case plan_for(ctx, program) do
      {:simple, native} -> run_simple(ctx, program, native, args)
      %Plan{} = plan -> run_plan(ctx, plan, program, args)
    end
  end

  defp run_simple(ctx, program, native, args) do
    inputs =
      program.inputs
      |> Enum.with_index()
      |> Enum.map(fn
        {nil, _i} -> nil
        {{_type, _shape}, i} -> to_buffer(ctx, Enum.at(args, i))
      end)

    captures = Enum.map(program.captures, &upload(ctx, &1))

    consts = Enum.map(program.constants, fn _ -> nil end)

    case NxCL.Native.eval_program(native, inputs ++ captures ++ consts) do
      {:ok, outputs} -> outputs
      {:error, reason} -> raise NxCL.CompilerError, message: "device run failed: #{reason}"
    end
  end

  defp read_outputs(program, outputs) do
    leaves = Composite.flatten_list([program.output_template])

    tensors =
      outputs
      |> Enum.zip(leaves)
      |> Enum.map(fn {buf, template} ->
        buf
        |> NxCL.Native.buffer_read(Nx.byte_size(template))
        |> Nx.from_binary(template.type)
        |> Nx.reshape(template.shape, names: template.names)
      end)

    {container, []} =
      Composite.traverse(program.output_template, tensors, fn _template, [t | rest] ->
        {t, rest}
      end)

    container
  end

  @doc "The shared device context."
  @spec ctx() :: reference()
  def ctx() do
    case :persistent_term.get(:nxcl_device_ctx, nil) do
      nil ->
        ctx = NxCL.Native.device_ctx_create(0)
        :persistent_term.put(:nxcl_device_ctx, ctx)
        ctx

      ctx ->
        ctx
    end
  end

  @doc """
  Encodes a program as the term `NxCL.Native.compile_program/2` reads. Exposed
  for tests of the native decoder.
  """
  @spec encode(Program.t()) :: tuple()
  def encode(%Program{} = program) do
    {Enum.map(program.inputs, &encode_slot/1), Enum.map(program.captures, &{&1.type, &1.shape}),
     Enum.map(program.constants, &encode_const/1),
     Enum.map(program.instructions, &encode_instruction/1), program.outputs}
  end

  # The decoded program stays in `persistent_term`, so a repeat call skips decoding
  # and kernel lookups. Stage 15 replaces this with a bounded cache.
  defp compiled(ctx, program) do
    wire = encode(program)
    cached(ctx, {__MODULE__, wire}, wire)
  end

  defp cached(ctx, key, wire) do
    case :persistent_term.get(key, nil) do
      nil ->
        case NxCL.Native.compile_program(ctx, wire) do
          {:ok, native} ->
            :persistent_term.put(key, native)
            native

          {:error, reason} ->
            raise NxCL.CompilerError, message: "invalid program: #{reason}"
        end

      native ->
        native
    end
  end

  # A program with control flow caches its plan under a key that covers every
  # sub-program. Each segment of the plan compiles the first time it runs.
  defp plan_for(ctx, program) do
    if Plan.control?(program) do
      key = {__MODULE__, :plan, signature(program)}

      case :persistent_term.get(key, nil) do
        nil ->
          plan = Plan.build(program)
          :persistent_term.put(key, plan)
          plan

        plan ->
          plan
      end
    else
      {:simple, compiled(ctx, program)}
    end
  end

  defp signature(%Program{} = program) do
    {program.inputs, Enum.map(program.captures, &{&1.type, &1.shape}), program.constants,
     Enum.map(program.instructions, &instruction_signature/1), program.outputs}
  end

  defp instruction_signature(%Instruction{} = instr) do
    subs =
      Enum.map(instr.subprograms, fn sub ->
        {Enum.map(sub.instructions, &instruction_signature/1), sub.outputs}
      end)

    {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs, subs}
  end

  # -- Segmented run -----------------------------------------------------------

  defp run_plan(ctx, plan, program, args) do
    captures =
      for {t, i} <- Enum.with_index(program.captures),
          into: %{},
          do: {{:capture, i}, upload(ctx, t)}

    shared = Map.merge(captures, const_buffers(ctx, plan))
    env = %{ctx: ctx, shared: shared, host_calls: program.host_calls}

    table =
      program.inputs
      |> Enum.with_index()
      |> Enum.reduce(shared, fn
        {nil, _i}, table ->
          table

        {{_type, _shape}, i}, table ->
          Map.put(table, {:input, i}, to_buffer(ctx, Enum.at(args, i)))
      end)

    run_scope(plan.scope, table, env)
  end

  # Consts that a control step or an output reads as a buffer, made once per run.
  defp const_buffers(_ctx, %Plan{const_keys: []}), do: %{}

  defp const_buffers(ctx, %Plan{const_keys: keys, const_wire: wire}) do
    native = cached(ctx, {__MODULE__, :consts, wire}, wire)
    buffers = eval!(native, List.duplicate(nil, length(keys)))
    keys |> Enum.zip(buffers) |> Map.new()
  end

  defp run_scope(%{steps: steps, outputs: outputs}, table, env) do
    table = Enum.reduce(steps, table, &run_step(&1, &2, env))
    Enum.map(outputs, &Map.fetch!(table, &1))
  end

  # A sub-scope starts from the shared captures and consts plus its carry.
  defp run_sub(scope, carry, env) do
    table =
      carry
      |> Enum.with_index()
      |> Enum.reduce(env.shared, fn {buf, k}, table -> Map.put(table, {:carry, k}, buf) end)

    run_scope(scope, table, env)
  end

  defp run_step(%{kind: :segment} = step, table, env) do
    native = segment_native(env.ctx, step)
    buffers = Enum.map(step.ins, &Map.fetch!(table, &1)) ++ List.duplicate(nil, step.n_consts)
    Stats.bump(:segment_runs)

    table =
      step.outs
      |> Enum.zip(eval!(native, buffers))
      |> Enum.reduce(table, fn {key, buf}, table -> Map.put(table, key, buf) end)

    Map.drop(table, step.free)
  end

  defp run_step(%{kind: :cond} = step, table, env) do
    args = Enum.map(step.operands, &Map.fetch!(table, &1))
    table = Map.drop(table, step.free)
    put_results(table, step.live, pick_clause(step.clauses, step.else, args, env))
  end

  defp run_step(%{kind: :while} = step, table, env) do
    carry = Enum.map(step.operands, &Map.fetch!(table, &1))
    table = Map.drop(table, step.free)
    put_results(table, step.live, loop(step, carry, env))
  end

  defp run_step(%{kind: :host_call} = step, table, env) do
    buffers = Enum.map(step.operands, &Map.fetch!(table, &1))
    table = Map.drop(table, step.free)
    entry = Enum.at(env.host_calls, step.index)
    put_results(table, step.live, Host.run(entry, env.ctx, buffers))
  end

  defp put_results(table, live, results) do
    Enum.reduce(live, table, fn {pos, key}, table ->
      Map.put(table, key, Enum.at(results, pos))
    end)
  end

  defp pick_clause([{pred, body} | clauses], last, args, env) do
    if truthy?(pred, args, env),
      do: run_sub(body, args, env),
      else: pick_clause(clauses, last, args, env)
  end

  defp pick_clause([], last, args, env), do: run_sub(last, args, env)

  defp loop(%{static: count, body: body}, carry, env) when is_integer(count) do
    Enum.reduce(1..count//1, carry, fn _i, carry ->
      Stats.bump(:loop_iterations)
      run_sub(body, carry, env)
    end)
  end

  defp loop(%{cond: cond_scope, body: body} = step, carry, env) do
    if truthy?(cond_scope, carry, env) do
      Stats.bump(:loop_iterations)
      loop(step, run_sub(body, carry, env), env)
    else
      carry
    end
  end

  # Runs a predicate scope and reads its scalar back. Nx tests `!= 0`.
  defp truthy?(%{out_types: [{kind, bits} = type]} = scope, carry, env) do
    [buf] = run_sub(scope, carry, env)
    Stats.bump(:predicate_reads)
    bytes = NxCL.Native.buffer_read(buf, div(bits, 8))

    if kind in [:s, :u] do
      bytes != :binary.copy(<<0>>, byte_size(bytes))
    else
      bytes |> Nx.from_binary(type) |> Nx.reshape({}) |> Nx.to_number() != 0
    end
  end

  defp segment_native(ctx, %{key: key, wire: wire}), do: cached(ctx, key, wire)

  defp eval!(native, buffers) do
    case NxCL.Native.eval_program(native, buffers) do
      {:ok, outputs} -> outputs
      {:error, reason} -> raise NxCL.CompilerError, message: "device run failed: #{reason}"
    end
  end

  defp encode_slot(nil), do: nil
  defp encode_slot({type, shape}), do: {type, shape}

  @doc false
  @spec encode_const({term(), Nx.Type.t()}) :: {Nx.Type.t(), term()}
  def encode_const({value, type})
      when is_number(value) or value in [:nan, :infinity, :neg_infinity],
      do: {type, value}

  # A complex constant goes as a `{re, im}` pair.
  def encode_const({%Complex{re: re, im: im}, type})
      when (is_number(re) or re in [:nan, :infinity, :neg_infinity]) and
             (is_number(im) or im in [:nan, :infinity, :neg_infinity]),
      do: {type, {re, im}}

  def encode_const({value, type}) do
    raise NxCL.CompilerError,
      message: "constant #{inspect(value)} of type #{inspect(type)} is not supported yet"
  end

  @doc false
  @spec encode_instruction(Instruction.t()) :: tuple()
  def encode_instruction(%Instruction{} = instr),
    do: {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs}

  defp to_buffer(_ctx, {:buffer, ref}), do: ref
  defp to_buffer(ctx, arg), do: upload(ctx, arg.())

  defp upload(ctx, tensor), do: NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))
end
