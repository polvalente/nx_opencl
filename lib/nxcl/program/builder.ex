defmodule NxCL.Program.Builder do
  @moduledoc """
  Builds `NxCL.Program.Instruction` values.

  The lowering pass calls these functions. Stage 13 calls them from the eager
  backend, so both paths emit the same instructions.
  """

  alias NxCL.Program.Instruction

  @unary_ops [:negate, :exp, :log, :tanh, :sigmoid, :abs]
  @binary_ops [:add, :subtract, :multiply, :divide]

  @doc "Unary elementwise ops this builder knows."
  @spec unary_ops() :: [atom()]
  def unary_ops(), do: @unary_ops

  @doc "Binary elementwise ops this builder knows."
  @spec binary_ops() :: [atom()]
  def binary_ops(), do: @binary_ops

  @doc "An elementwise unary instruction."
  @spec unary(atom(), Instruction.ref(), Nx.Type.t(), tuple()) :: Instruction.t()
  def unary(op, operand, type, shape) when op in @unary_ops do
    %Instruction{op: op, operands: [operand], out_type: type, out_shape: shape}
  end

  @doc "An elementwise binary instruction. Both operands have the output shape."
  @spec binary(atom(), Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple()) ::
          Instruction.t()
  def binary(op, left, right, type, shape) when op in @binary_ops do
    %Instruction{op: op, operands: [left, right], out_type: type, out_shape: shape}
  end

  @doc "A 2D matrix product. `k` is the contracted size."
  @spec dot(Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple(), pos_integer()) ::
          Instruction.t()
  def dot(left, right, type, shape, k) do
    %Instruction{op: :dot, operands: [left, right], out_type: type, out_shape: shape, attrs: [k]}
  end

  @doc "A `reshape` or `squeeze`. The result shares the operand's buffer."
  @spec alias_op(:reshape | :squeeze, Instruction.ref(), Nx.Type.t(), tuple()) ::
          Instruction.t()
  def alias_op(op, operand, type, shape) when op in [:reshape, :squeeze] do
    %Instruction{
      op: op,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: [:alias]
    }
  end
end
