defmodule NxCL.Program.Builder.Linalg do
  @moduledoc """
  Builds the linear algebra and FFT instructions. Attrs are semantic: the
  native side derives batches, strides and tables from the operand shapes.
  """

  alias NxCL.Program.Instruction

  @doc """
  `triangular_solve`: solves `op(a) x = b` (`left_side`) or `x op(a) = b`. The
  attrs are `[left_side, lower, transform_a]`, the flags as 0 or 1 and
  `transform_a` as `:none`, `:transpose` or `:conjugate`.
  """
  @spec triangular_solve(
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          keyword()
        ) :: Instruction.t()
  def triangular_solve(a, b, type, shape, opts) do
    %Instruction{
      op: :triangular_solve,
      operands: [a, b],
      out_type: type,
      out_shape: shape,
      attrs: [flag(opts[:left_side]), flag(opts[:lower]), opts[:transform_a]]
    }
  end

  @doc "`cholesky` of a batch of square matrices; the lower triangle, upper zero."
  @spec cholesky(Instruction.ref(), Nx.Type.t(), tuple()) :: Instruction.t()
  def cholesky(operand, type, shape) do
    %Instruction{op: :cholesky, operands: [operand], out_type: type, out_shape: shape}
  end

  @doc """
  `fft` or `ifft` along `axis` of a complex operand whose axis already has the
  transform length. `ifft` divides by the length, as Nx does.
  """
  @spec fft(:fft | :ifft, Instruction.ref(), Nx.Type.t(), tuple(), non_neg_integer()) ::
          Instruction.t()
  def fft(op, operand, type, shape, axis) when op in [:fft, :ifft] do
    %Instruction{op: op, operands: [operand], out_type: type, out_shape: shape, attrs: [axis]}
  end

  defp flag(true), do: 1
  defp flag(false), do: 0
end
