defmodule NxCL.Program.Builder do
  @moduledoc """
  Builds `NxCL.Program.Instruction` values.

  The lowering pass calls these functions. Stage 13 calls them from the eager
  backend, so both paths emit the same instructions.
  """

  alias NxCL.Program.Instruction

  @unary_ops ~w(abs acos acosh asin asinh atan atanh bitwise_not cbrt ceil cos cosh
                count_leading_zeros erf erf_inv erfc exp expm1 floor is_infinity is_nan log
                log1p logical_not negate population_count round rsqrt sigmoid sign sin sinh
                sqrt tan tanh)a

  @binary_ops ~w(add atan2 bitwise_and bitwise_or bitwise_xor divide equal greater
                 greater_equal left_shift less less_equal logical_and logical_or logical_xor max
                 min multiply not_equal pow quotient remainder right_shift subtract)a

  @doc "Unary elementwise ops this builder knows."
  @spec unary_ops() :: [atom()]
  def unary_ops(), do: @unary_ops

  @doc "Binary elementwise ops this builder knows, comparisons included."
  @spec binary_ops() :: [atom()]
  def binary_ops(), do: @binary_ops

  @doc """
  An elementwise unary instruction. The operand has the output shape or is a
  scalar constant. The native side picks the kernel from the op and the types.
  """
  @spec unary(atom(), Instruction.ref(), Nx.Type.t(), tuple()) :: Instruction.t()
  def unary(op, operand, type, shape) when op in @unary_ops do
    %Instruction{op: op, operands: [operand], out_type: type, out_shape: shape}
  end

  @doc """
  An elementwise binary instruction. Operands have the output shape or are
  scalar constants. Operand types may differ from each other and from the
  result; the kernel converts them.
  """
  @spec binary(atom(), Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple()) ::
          Instruction.t()
  def binary(op, left, right, type, shape) when op in @binary_ops do
    %Instruction{op: op, operands: [left, right], out_type: type, out_shape: shape}
  end

  @doc "A cast to the output type. A cast to the operand's own type moves no data."
  @spec as_type(Instruction.ref(), Nx.Type.t(), tuple()) :: Instruction.t()
  def as_type(operand, type, shape) do
    %Instruction{op: :as_type, operands: [operand], out_type: type, out_shape: shape}
  end

  @doc "Reads the operand's bytes as another type of the same width. No copy."
  @spec bitcast(Instruction.ref(), Nx.Type.t(), tuple()) :: Instruction.t()
  def bitcast(operand, type, shape) do
    %Instruction{op: :bitcast, operands: [operand], out_type: type, out_shape: shape}
  end

  @doc "`pred != 0 ? on_true : on_false`, with both branches cast to the output type."
  @spec select(
          Instruction.ref(),
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple()
        ) :: Instruction.t()
  def select(pred, on_true, on_false, type, shape) do
    %Instruction{
      op: :select,
      operands: [pred, on_true, on_false],
      out_type: type,
      out_shape: shape
    }
  end

  @doc "`clip(x, min, max)`: raise to `min`, then cap at `max`."
  @spec clip(Instruction.ref(), Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple()) ::
          Instruction.t()
  def clip(operand, min, max, type, shape) do
    %Instruction{op: :clip, operands: [operand, min, max], out_type: type, out_shape: shape}
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
