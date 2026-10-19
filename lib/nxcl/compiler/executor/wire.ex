defmodule NxCL.Compiler.Executor.Wire do
  @moduledoc false
  # The term `NxCL.Native.compile_program/2` reads, and the functions that build it
  # from a `NxCL.Compiler.Program`. `NxCL.Compiler.Executor` and
  # `NxCL.Compiler.Executor.Plan` both use it.
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

  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Program

  @type const :: {Nx.Type.t(), number() | atom() | {number() | atom(), number() | atom()}}
  @type instruction ::
          {atom(), [Instruction.ref()], Nx.Type.t() | [Nx.Type.t()], tuple() | [tuple()],
           [Instruction.attr()]}
  @type t ::
          {[{Nx.Type.t(), tuple()} | nil], [{Nx.Type.t(), tuple()}], [const()], [instruction()],
           [Instruction.ref()]}

  @special [:nan, :infinity, :neg_infinity]

  @doc "Assembles the wire term from its five parts."
  @spec build(
          [{Nx.Type.t(), tuple()} | nil],
          [{Nx.Type.t(), tuple()}],
          [const()],
          [instruction()],
          [Instruction.ref()]
        ) :: t()
  def build(inputs, captures, constants, instructions, outputs),
    do: {inputs, captures, constants, instructions, outputs}

  @doc "Encodes a whole program."
  @spec program(Program.t()) :: t()
  def program(%Program{} = program) do
    build(
      program.inputs,
      Enum.map(program.captures, &{&1.type, &1.shape}),
      Enum.map(program.constants, &const/1),
      Enum.map(program.instructions, &instruction/1),
      program.outputs
    )
  end

  @doc """
  Encodes a constant `{value, type}`. A value is a finite number, `:nan`,
  `:infinity`, `:neg_infinity`, or a `Complex` of those.
  """
  @spec const({term(), Nx.Type.t()}) :: const()
  def const({value, type}) when is_number(value) or value in @special, do: {type, value}

  def const({%Complex{re: re, im: im}, type})
      when (is_number(re) or re in @special) and (is_number(im) or im in @special),
      do: {type, {re, im}}

  def const({value, type}) do
    raise NxCL.CompilerError,
      message:
        "constant #{inspect(value)} of type #{inspect(type)} is not supported: a constant " <>
          "is a number, :nan, :infinity, :neg_infinity, or a Complex of those"
  end

  @doc "Encodes one instruction."
  @spec instruction(Instruction.t()) :: instruction()
  def instruction(%Instruction{} = instr),
    do: {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs}
end
