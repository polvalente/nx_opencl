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

  @doc """
  A generic kernel launch. `source` is OpenCL C, `entry` the kernel name.

  The kernel takes the operand buffers, then the output buffer, then each
  scalar as an `int`. `global` and `local` are work sizes, one per dimension
  (1 to 3); pass `[]` for `local` to let the driver pick.
  """
  @spec launch(
          [Instruction.ref()],
          Nx.Type.t(),
          tuple(),
          binary(),
          binary(),
          keyword()
        ) :: Instruction.t()
  def launch(operands, type, shape, source, entry, opts) do
    global = Keyword.fetch!(opts, :global)
    local = Keyword.get(opts, :local, [])
    local = if local == [], do: Enum.map(global, fn _ -> 0 end), else: local

    attrs =
      [source, entry, Keyword.get(opts, :build_options, ""), length(global)] ++
        global ++ local ++ Keyword.get(opts, :scalars, [])

    %Instruction{op: :launch, operands: operands, out_type: type, out_shape: shape, attrs: attrs}
  end
end
