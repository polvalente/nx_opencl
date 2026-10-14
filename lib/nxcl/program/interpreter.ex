defmodule NxCL.Program.Interpreter do
  @moduledoc false
  # Temporary Elixir interpreter: runs an `NxCL.Program` one NIF call per
  # instruction. Stage 03 replaces it with the native executor.

  alias Nx.Defn.Composite
  alias NxCL.Program
  alias NxCL.Program.Instruction

  @f32 {:f, 32}

  @doc """
  Runs `program` on `args`, the flat list of lazy argument functions that
  `Nx.Defn.Compiler` passes. Returns the output container.
  """
  @spec run(Program.t(), [(-> Nx.Tensor.t())]) :: Nx.Container.t()
  def run(%Program{} = program, args) do
    ctx = get_ctx()

    inputs =
      program.inputs
      |> Enum.with_index()
      |> Map.new(fn
        {nil, i} -> {i, nil}
        {{type, shape}, i} -> {i, upload(ctx, Enum.at(args, i).(), type, shape, "input #{i}")}
      end)

    consts =
      Enum.map(program.constants, fn {value, type} ->
        upload(ctx, Nx.tensor(value, type: type), type, {}, "constant")
      end)

    captures =
      Enum.map(program.captures, fn t ->
        upload(ctx, t, t.type, t.shape, "captured tensor")
      end)

    env = %{inputs: inputs, consts: consts, captures: captures, results: %{}}

    env =
      program.instructions
      |> Enum.with_index()
      |> Enum.reduce(env, fn {instr, i}, env ->
        check_type!(instr.out_type, "result of #{inspect(instr.op)}")
        put_in(env.results[i], execute(ctx, instr, env))
      end)

    leaves = Composite.flatten_list([program.output_template])

    tensors =
      program.outputs
      |> Enum.zip(leaves)
      |> Enum.map(fn {ref, template} ->
        check_type!(template.type, "output")
        {buf, _shape} = fetch(env, ref)
        bin = NxCL.Native.buffer_read(buf, Nx.byte_size(template))

        bin
        |> Nx.from_binary(template.type)
        |> Nx.reshape(template.shape, names: template.names)
      end)

    {container, []} =
      Composite.traverse(program.output_template, tensors, fn _template, [t | rest] ->
        {t, rest}
      end)

    container
  end

  # Buffers are {nif_ref, shape} pairs.
  defp upload(ctx, tensor, type, shape, what) do
    check_type!(type, what)
    check_type!(tensor.type, what)
    {NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor)), shape}
  end

  defp check_type!(@f32, _what), do: :ok

  defp check_type!(type, what) do
    raise NxCL.CompilerError,
      message:
        "dtype #{inspect(type)} is not supported until stage 04 (NxCL.Compiler runs f32 only; got it in #{what})"
  end

  defp fetch(env, {:input, i}), do: Map.fetch!(env.inputs, i)
  defp fetch(env, {:const, i}), do: Enum.at(env.consts, i)
  defp fetch(env, {:capture, i}), do: Enum.at(env.captures, i)
  defp fetch(env, {:result, i}), do: Map.fetch!(env.results, i)

  defp execute(ctx, %Instruction{op: op, operands: [x], out_shape: shape}, env)
       when op in [:negate, :exp, :log, :tanh, :sigmoid, :abs] do
    {buf, _} = fetch(env, x)
    ref = NxCL.Native.elementwise_unary_op(ctx, Atom.to_string(op), buf, Nx.size(shape))
    {ref, shape}
  end

  defp execute(ctx, %Instruction{op: op, operands: [l, r], out_shape: shape}, env)
       when op in [:add, :subtract, :multiply, :divide] do
    {left, _} = fetch(env, l)
    {right, _} = fetch(env, r)
    ref = NxCL.Native.elementwise_binary_op(ctx, Atom.to_string(op), left, right, Nx.size(shape))
    {ref, shape}
  end

  defp execute(ctx, %Instruction{op: :dot, operands: [l, r], out_shape: {m, n}, attrs: [k]}, env) do
    {left, _} = fetch(env, l)
    {right, _} = fetch(env, r)
    {NxCL.Native.matmul_op(ctx, left, right, m, n, k), {m, n}}
  end

  # reshape and squeeze share the operand's buffer.
  defp execute(_ctx, %Instruction{op: op, operands: [x], out_shape: shape}, env)
       when op in [:reshape, :squeeze] do
    {buf, _} = fetch(env, x)
    {buf, shape}
  end

  defp execute(_ctx, %Instruction{op: op}, _env) do
    raise NxCL.CompilerError,
      message: "Unsupported op in NxCL.Program.Interpreter: #{inspect(op)}"
  end

  defp get_ctx() do
    case :persistent_term.get(:nxcl_device_ctx, nil) do
      nil ->
        ctx = NxCL.Native.device_ctx_create(0)
        :persistent_term.put(:nxcl_device_ctx, ctx)
        ctx

      ctx ->
        ctx
    end
  end
end
