defmodule NxCL.Executor do
  @moduledoc false
  # Runs an `NxCL.Program` on the device: upload the inputs, run the program
  # natively, read the outputs, rebuild the container. Stage 10 adds cut points
  # for control flow here.
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
  # A `dot` carries `[m, n, k]` as attrs. Buffers go to `eval_program/2` in the
  # order inputs, captures, constants; constants go as `nil`, because Rust holds
  # their values. The native side generates the kernel for each instruction from
  # the op, the types and the shapes.

  alias Nx.Defn.Composite
  alias NxCL.Program
  alias NxCL.Program.Instruction

  @doc """
  Runs `program` on `args`, the flat list of lazy argument functions that
  `Nx.Defn.Compiler` passes. Returns the output container.
  """
  @spec run(Program.t(), [(-> Nx.Tensor.t())]) :: Nx.Container.t()
  def run(%Program{} = program, args) do
    ctx = ctx()
    native = compiled(ctx, program)

    inputs =
      program.inputs
      |> Enum.with_index()
      |> Enum.map(fn
        {nil, _i} -> nil
        {{_type, _shape}, i} -> upload(ctx, Enum.at(args, i).())
      end)

    captures = Enum.map(program.captures, &upload(ctx, &1))

    consts = Enum.map(program.constants, fn _ -> nil end)

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
    {Enum.map(program.inputs, &encode_slot/1), Enum.map(program.captures, &{&1.type, &1.shape}),
     Enum.map(program.constants, &encode_const/1),
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

  defp encode_slot(nil), do: nil
  defp encode_slot({type, shape}), do: {type, shape}

  defp encode_const({value, type})
       when is_number(value) or value in [:nan, :infinity, :neg_infinity],
       do: {type, value}

  # A complex constant goes as a `{re, im}` pair.
  defp encode_const({%Complex{re: re, im: im}, type})
       when (is_number(re) or re in [:nan, :infinity, :neg_infinity]) and
              (is_number(im) or im in [:nan, :infinity, :neg_infinity]),
       do: {type, {re, im}}

  defp encode_const({value, type}) do
    raise NxCL.CompilerError,
      message: "constant #{inspect(value)} of type #{inspect(type)} is not supported yet"
  end

  defp encode_instruction(%Instruction{op: :dot} = instr) do
    {m, n} = instr.out_shape
    [k] = instr.attrs
    {:dot, instr.operands, instr.out_type, instr.out_shape, [m, n, k]}
  end

  defp encode_instruction(%Instruction{} = instr),
    do: {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs}

  defp upload(ctx, tensor), do: NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))
end
