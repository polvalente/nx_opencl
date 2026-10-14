defmodule NxCL.Executor do
  @moduledoc false
  # Runs an `NxCL.Program` on the device: upload the inputs, run the program
  # natively, read the outputs, rebuild the container. Stage 10 adds cut points
  # for control flow here.
  #
  # Wire term sent to `NxCL.Native.compile_program/2`:
  #
  #     {n_inputs, n_captures, n_consts, [instr], [out_ref]}
  #     instr = {op, [ref], out_bytes, [attr]}
  #     ref = {:input | :capture | :const | :result, index}
  #
  # A `dot` carries `[m, n, k]` as attrs. Buffers go to `eval_program/2` in the
  # order inputs, captures, constants.

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
    ctx = ctx()
    check_types!(program)
    native = compiled(ctx, program)

    inputs =
      program.inputs
      |> Enum.with_index()
      |> Enum.map(fn
        {nil, _i} -> nil
        {{_type, _shape}, i} -> upload(ctx, Enum.at(args, i).())
      end)

    captures = Enum.map(program.captures, &upload(ctx, &1))

    consts =
      Enum.map(program.constants, fn {value, type} ->
        upload(ctx, Nx.tensor(value, type: type))
      end)

    outputs =
      case NxCL.Native.eval_program(native, inputs ++ captures ++ consts) do
        {:ok, outputs} -> outputs
        {:error, reason} -> raise NxCL.CompilerError, message: "device run failed: #{reason}"
      end

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
    {length(program.inputs), length(program.captures), length(program.constants),
     Enum.map(program.instructions, &encode_instruction/1), program.outputs}
  end

  # The decoded program stays in `persistent_term`, so a repeat call skips decoding
  # and kernel lookups. Stage 15 replaces this with a bounded cache.
  defp compiled(ctx, program) do
    wire = encode(program)
    key = {__MODULE__, wire}

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

  defp encode_instruction(%Instruction{op: :dot} = instr) do
    {m, n} = instr.out_shape
    [k] = instr.attrs
    {:dot, instr.operands, out_bytes(instr), [m, n, k]}
  end

  defp encode_instruction(%Instruction{} = instr),
    do: {instr.op, instr.operands, out_bytes(instr), instr.attrs}

  defp out_bytes(%Instruction{out_type: {_, bits}, out_shape: shape}),
    do: Nx.size(shape) * div(bits, 8)

  defp upload(ctx, tensor), do: NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))

  # Every dtype but f32 waits for stage 04.
  defp check_types!(%Program{} = program) do
    for {type, _shape} <- Enum.reject(program.inputs, &is_nil/1),
        do: check_type!(type, "input")

    for {_value, type} <- program.constants, do: check_type!(type, "constant")
    for t <- program.captures, do: check_type!(t.type, "captured tensor")

    for instr <- program.instructions,
        do: check_type!(instr.out_type, "result of #{inspect(instr.op)}")

    for t <- Composite.flatten_list([program.output_template]), do: check_type!(t.type, "output")
    :ok
  end

  defp check_type!(@f32, _what), do: :ok

  defp check_type!(type, what) do
    raise NxCL.CompilerError,
      message:
        "dtype #{inspect(type)} is not supported until stage 04 (NxCL.Compiler runs f32 only; got it in #{what})"
  end
end
