defmodule NxCL.Compiler.Executor do
  @moduledoc false
  # Runs an `NxCL.Compiler.Program` on the device.
  #
  # A program with no `cond`, `while` or `host_call` runs natively whole.
  # Otherwise `Plan` cuts it at those instructions: each straight-line segment
  # runs natively, and this module runs the control steps. Buffers stay on the
  # device in a table keyed by ref; the last reader drops each one.

  alias Nx.Defn.Composite
  alias NxCL.Compiler.Executor.Host
  alias NxCL.Compiler.Executor.Plan
  alias NxCL.Compiler.Executor.Wire
  alias NxCL.Compiler.Program
  alias NxCL.Compiler.ProgramCache
  alias NxCL.Devices

  @doc """
  Runs `program` on `args`, the flat list of lazy argument functions from
  `Nx.Defn.Compiler`. Returns the output container. `opts` take `:device` and
  `:command_queue`.
  """
  @spec run(Program.t(), [(-> Nx.Tensor.t())], keyword()) :: Nx.Container.t()
  def run(%Program{} = program, args, opts \\ []) do
    program
    |> run_buffers(args, opts)
    |> then(&read_outputs(program, Devices.fetch!(opts).ctx, &1))
  end

  @doc """
  Runs `program` and returns one device buffer per output leaf. An argument is a
  lazy function, or `{:buffer, ref}` for a tensor already on the device.
  """
  @spec run_buffers(Program.t(), [(-> Nx.Tensor.t()) | {:buffer, reference()}], keyword()) ::
          [reference()]
  def run_buffers(%Program{} = program, args, opts \\ []) do
    check_arg_count!(program, args)
    dev = Devices.fetch!(opts)

    case plan_for(dev, program) do
      {:simple, native} -> run_simple(dev, program, native, args)
      %Plan{} = plan -> run_plan(dev, plan, program, args)
    end
  end

  defp check_arg_count!(%Program{inputs: inputs}, args) do
    if length(args) != length(inputs) do
      raise NxCL.CompilerError,
        message: "expected #{length(inputs)} arguments, got #{length(args)}"
    end
  end

  defp run_simple(%{ctx: ctx} = dev, program, native, args) do
    consts = Enum.map(program.constants, fn _ -> nil end)

    eval!(
      dev,
      native,
      input_buffers(ctx, program, args) ++ capture_buffers(ctx, program) ++ consts
    )
  end

  # `nil` for an argument the function never reads.
  defp input_buffers(ctx, %Program{inputs: inputs}, args) do
    inputs
    |> Enum.zip(args)
    |> Enum.map(fn
      {nil, _arg} -> nil
      {_slot, arg} -> to_buffer(ctx, arg)
    end)
  end

  defp capture_buffers(ctx, %Program{captures: captures}),
    do: Enum.map(captures, &tensor_buffer(ctx, &1))

  @doc """
  Returns `program` with each captured tensor on the device `opts` name, so runs
  reuse its buffer.
  """
  @spec with_resident_captures(Program.t(), keyword()) :: Program.t()
  def with_resident_captures(%Program{} = program, opts \\ []) do
    dev = Devices.fetch!(opts)

    %Program{program | captures: Enum.map(program.captures, &resident(dev, &1))}
  end

  defp resident(%{ctx: ctx}, %Nx.Tensor{data: %NxCL.Backend{ctx: ctx}} = tensor), do: tensor

  defp resident(%{key: key}, tensor),
    do: NxCL.Backend.from_binary(tensor, Nx.to_binary(tensor), device: key)

  # One wait for all outputs: each blocking read costs a device round trip.
  defp read_outputs(program, ctx, outputs) do
    leaves = Composite.flatten_list([program.output_template])

    binaries =
      ctx
      |> NxCL.Native.buffer_read_many(Enum.zip_with(outputs, leaves, &{&1, Nx.byte_size(&2)}))
      |> NxCL.Native.ok!("could not read the outputs")

    tensors =
      binaries
      |> Enum.zip(leaves)
      |> Enum.map(fn {binary, template} ->
        binary
        |> Nx.from_binary(template.type)
        |> Nx.reshape(template.shape, names: template.names)
      end)

    {container, []} =
      Composite.traverse(program.output_template, tensors, fn _template, [t | rest] ->
        {t, rest}
      end)

    container
  end

  # The key is the signature, so the wire term is built only on a miss.
  defp compiled(dev, program) do
    key = {:program, dev.key, Program.signature(program)}
    cached(dev, key, fn -> Wire.program(program) end)
  end

  defp cached(%{ctx: ctx}, key, wire) do
    ProgramCache.fetch_or_build(key, fn ->
      case NxCL.Native.compile_program(ctx, wire.()) do
        {:ok, native} -> native
        {:error, reason} -> raise NxCL.CompilerError, message: "invalid program: #{reason}"
      end
    end)
  end

  # Each segment of a plan compiles the first time it runs.
  defp plan_for(dev, program) do
    if Plan.control?(program) do
      ProgramCache.fetch_or_build({:plan, Program.signature(program)}, fn ->
        Plan.build(program)
      end)
    else
      {:simple, compiled(dev, program)}
    end
  end

  defp run_plan(%{ctx: ctx} = dev, plan, program, args) do
    captures =
      ctx
      |> capture_buffers(program)
      |> Enum.with_index()
      |> Map.new(fn {buf, i} -> {{:capture, i}, buf} end)

    shared = Map.merge(captures, const_buffers(dev, plan))
    env = %{dev: dev, ctx: ctx, shared: shared, host_calls: program.host_calls}

    table =
      ctx
      |> input_buffers(program, args)
      |> Enum.with_index()
      |> Enum.reduce(shared, fn
        {nil, _i}, table -> table
        {buf, i}, table -> Map.put(table, {:input, i}, buf)
      end)

    run_scope(plan.scope, table, env)
  end

  # Constants that a control step or an output reads as a buffer.
  defp const_buffers(_dev, %Plan{const_keys: []}), do: %{}

  defp const_buffers(dev, %Plan{const_keys: keys, const_wire: wire, const_key: key}) do
    native = cached(dev, {:program, dev.key, key}, fn -> wire end)
    buffers = eval!(dev, native, List.duplicate(nil, length(keys)))
    keys |> Enum.zip(buffers) |> Map.new()
  end

  defp run_scope(%{steps: steps, outputs: outputs}, table, env) do
    table = Enum.reduce(steps, table, &run_step(&1, &2, env))
    Enum.map(outputs, &Map.fetch!(table, &1))
  end

  defp run_sub(scope, carry, env) do
    table =
      carry
      |> Enum.with_index()
      |> Enum.reduce(env.shared, fn {buf, k}, table -> Map.put(table, {:carry, k}, buf) end)

    run_scope(scope, table, env)
  end

  defp run_step(%{kind: :segment} = step, table, env) do
    native = segment_native(env.dev, step)
    buffers = Enum.map(step.ins, &Map.fetch!(table, &1)) ++ List.duplicate(nil, step.n_consts)

    table =
      step.outs
      |> Enum.zip(eval!(env.dev, native, buffers))
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
      run_sub(body, carry, env)
    end)
  end

  defp loop(%{cond: cond_scope, body: body} = step, carry, env) do
    if truthy?(cond_scope, carry, env) do
      loop(step, run_sub(body, carry, env), env)
    else
      carry
    end
  end

  # Runs a predicate scope and reads its scalar back. Nx tests `!= 0`.
  defp truthy?(%{out_types: [{kind, bits} = type]} = scope, carry, env) do
    [buf] = run_sub(scope, carry, env)

    bytes =
      buf
      |> NxCL.Native.buffer_read(div(bits, 8))
      |> NxCL.Native.ok!("could not read a predicate")

    # Compare integer bytes with zero directly.
    if kind in [:s, :u] do
      bytes != :binary.copy(<<0>>, byte_size(bytes))
    else
      bytes |> Nx.from_binary(type) |> Nx.reshape({}) |> Nx.to_number() != 0
    end
  end

  defp segment_native(dev, %{key: key, wire: wire}),
    do: cached(dev, {:program, dev.key, key}, fn -> wire end)

  defp eval!(%{queue: queue}, native, buffers) do
    result =
      if queue,
        do: NxCL.Native.eval_program_on(native, buffers, queue),
        else: NxCL.Native.eval_program(native, buffers)

    case result do
      {:ok, outputs} -> outputs
      {:error, reason} -> raise NxCL.CompilerError, message: "device run failed: #{reason}"
    end
  end

  defp to_buffer(_ctx, {:buffer, ref}), do: ref
  defp to_buffer(ctx, arg), do: tensor_buffer(ctx, arg.())

  # A tensor already on this device needs no read back or upload.
  defp tensor_buffer(ctx, %Nx.Tensor{data: %NxCL.Backend{ref: ref, ctx: ctx}}), do: ref

  defp tensor_buffer(ctx, tensor),
    do:
      ctx
      |> NxCL.Native.buffer_write(Nx.to_binary(tensor))
      |> NxCL.Native.ok!("could not upload an argument")
end
