defmodule NxCL.Compiler.Ops do
  @moduledoc """
  Builds `NxCL.Compiler.Instruction` values.

  The lowering pass and the evaluator (`NxCL.Compiler.Evaluator`) call these
  functions, so both paths emit the same instructions. The linear algebra and
  FFT builders carry semantic attrs: the native side derives batches, strides
  and tables from the operand shapes.
  """

  alias NxCL.Compiler.Instruction

  @unary_ops ~w(abs acos acosh asin asinh atan atanh bitwise_not cbrt ceil cos cosh
                count_leading_zeros erf erf_inv erfc exp expm1 floor is_infinity is_nan log
                log1p logical_not negate population_count round rsqrt sigmoid sign sin sinh
                sqrt tan tanh real imag conjugate phase)a

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

  @doc """
  A batched matrix product: operands `[b, m, k]` and `[b, k, n]` give `[b, m, n]`.
  Every `dot` lowers to this form (`NxCL.Compiler.Lower.Contract`). Both operands have the
  output type.
  """
  @spec batched_gemm(
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          {pos_integer(), non_neg_integer(), non_neg_integer()},
          non_neg_integer()
        ) :: Instruction.t()
  def batched_gemm(left, right, type, {b, m, n} = shape, k) do
    %Instruction{
      op: :batched_gemm,
      operands: [left, right],
      out_type: type,
      out_shape: shape,
      attrs: [b, m, n, k]
    }
  end

  @doc """
  A convolution over a normalized layout: input `[n, c, spatial...]`, kernel
  `[o, i, kernel spatial...]`, output `[o, n / batch_groups, output spatial...]`.
  `opts` holds `:strides`, `:padding` (`{low, high}` per axis), `:kernel_dilation`,
  `:input_dilation`, `:feature_groups` and `:batch_groups`. Both operands have
  the output type.
  """
  @spec conv(Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple(), map()) ::
          Instruction.t()
  def conv(input, kernel, type, shape, opts) do
    attrs =
      [length(opts.strides) | opts.strides] ++
        Enum.flat_map(opts.padding, &Tuple.to_list/1) ++
        opts.kernel_dilation ++
        opts.input_dilation ++ [opts.feature_groups, opts.batch_groups]

    %Instruction{
      op: :conv,
      operands: [input, kernel],
      out_type: type,
      out_shape: shape,
      attrs: attrs
    }
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
  Broadcasts the operand to the output shape. `axes` gives, for each operand
  axis, the output axis it fills. A scalar constant operand fills the output.
  """
  @spec broadcast(Instruction.ref(), Nx.Type.t(), tuple(), [non_neg_integer()]) ::
          Instruction.t()
  def broadcast(operand, type, shape, axes) do
    %Instruction{
      op: :broadcast,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: axes
    }
  end

  @doc "Output axis `i` is operand axis `axes[i]`."
  @spec transpose(Instruction.ref(), Nx.Type.t(), tuple(), [non_neg_integer()]) ::
          Instruction.t()
  def transpose(operand, type, shape, axes) do
    %Instruction{
      op: :transpose,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: axes
    }
  end

  @doc "Reverses the operand along `axes`."
  @spec reverse(Instruction.ref(), Nx.Type.t(), tuple(), [non_neg_integer()]) ::
          Instruction.t()
  def reverse(operand, type, shape, axes) do
    %Instruction{
      op: :reverse,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: axes
    }
  end

  @doc """
  A strided slice. `starts` are refs to scalar integers (constants or
  tensors); the device clamps each to `[0, dim - length]`. `lengths` are the
  input spans, `strides` the steps.
  """
  @spec slice(
          Instruction.ref(),
          [Instruction.ref()],
          Nx.Type.t(),
          tuple(),
          [non_neg_integer()],
          [pos_integer()]
        ) :: Instruction.t()
  def slice(operand, starts, type, shape, lengths, strides) do
    %Instruction{
      op: :slice,
      operands: [operand | starts],
      out_type: type,
      out_shape: shape,
      attrs: lengths ++ strides
    }
  end

  @doc "Writes `update` into a copy of `operand` at the clamped `starts`."
  @spec put_slice(Instruction.ref(), Instruction.ref(), [Instruction.ref()], Nx.Type.t(), tuple()) ::
          Instruction.t()
  def put_slice(operand, update, starts, type, shape) do
    %Instruction{
      op: :put_slice,
      operands: [operand, update | starts],
      out_type: type,
      out_shape: shape
    }
  end

  @doc "Pads with `value`. `config` is `{low, high, interior}` per axis; low and high may be negative."
  @spec pad(
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          [{integer(), integer(), non_neg_integer()}]
        ) :: Instruction.t()
  def pad(operand, value, type, shape, config) do
    %Instruction{
      op: :pad,
      operands: [operand, value],
      out_type: type,
      out_shape: shape,
      attrs: Enum.flat_map(config, &Tuple.to_list/1)
    }
  end

  @doc "Joins the operands along `axis` (`:concatenate`) or on a new axis (`:stack`)."
  @spec join(:concatenate | :stack, [Instruction.ref()], Nx.Type.t(), tuple(), non_neg_integer()) ::
          Instruction.t()
  def join(op, operands, type, shape, axis) when op in [:concatenate, :stack] do
    %Instruction{op: op, operands: operands, out_type: type, out_shape: shape, attrs: [axis]}
  end

  @doc "Counts along `axis`, or over the flat index when `axis` is `nil`."
  @spec iota(Nx.Type.t(), tuple(), non_neg_integer() | nil) :: Instruction.t()
  def iota(type, shape, axis) do
    %Instruction{op: :iota, operands: [], out_type: type, out_shape: shape, attrs: [axis]}
  end

  @doc "Identity matrices over the last two axes."
  @spec eye(Nx.Type.t(), tuple()) :: Instruction.t()
  def eye(type, shape) do
    %Instruction{op: :eye, operands: [], out_type: type, out_shape: shape}
  end

  @doc """
  A reduction over `axes`: `sum`, `product`, `reduce_max`, `reduce_min`, `all` or
  `any`. The output type is the accumulator type; `keep_axes` only shows in the
  output shape.
  """
  @spec reduce(atom(), Instruction.ref(), Nx.Type.t(), tuple(), [non_neg_integer()]) ::
          Instruction.t()
  def reduce(op, operand, type, shape, axes)
      when op in [:sum, :product, :reduce_max, :reduce_min, :all, :any] do
    %Instruction{op: op, operands: [operand], out_type: type, out_shape: shape, attrs: axes}
  end

  @doc "`argmax` or `argmin` over `axes`; the output holds the index. `tie_break` is `:low` or `:high`."
  @spec arg_reduce(
          :argmax | :argmin,
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          :low | :high,
          [non_neg_integer()]
        ) :: Instruction.t()
  def arg_reduce(op, operand, type, shape, tie_break, axes)
      when op in [:argmax, :argmin] and tie_break in [:low, :high] do
    %Instruction{
      op: op,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: [tie_break | axes]
    }
  end

  @doc """
  A generic `reduce` over `axes`. `steps` is the combiner as `{op, a, b}`
  triples over variables: 0 and 1 are the two accumulator values and step `k`
  defines variable `2 + k`. `operands` is the tensor alone for an `add`,
  `multiply`, `max` or `min` body (the initial accumulator is folded in
  afterwards), or the tensor and the initial accumulator for any other body.
  """
  @spec reduce_steps(
          [Instruction.ref()],
          Nx.Type.t(),
          tuple(),
          [{atom(), non_neg_integer(), non_neg_integer()}],
          [non_neg_integer()]
        ) :: Instruction.t()
  def reduce_steps(operands, type, shape, steps, axes) do
    %Instruction{
      op: :reduce,
      operands: operands,
      out_type: type,
      out_shape: shape,
      attrs: steps_attrs(steps) ++ axes
    }
  end

  @doc "A cumulative `:cumulative_sum`, `:cumulative_product`, `:cumulative_min` or `:cumulative_max` along `axis`."
  @spec scan(atom(), Instruction.ref(), Nx.Type.t(), tuple(), non_neg_integer(), boolean()) ::
          Instruction.t()
  def scan(op, operand, type, shape, axis, reverse)
      when op in [:cumulative_sum, :cumulative_product, :cumulative_min, :cumulative_max] do
    %Instruction{
      op: op,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: [axis, if(reverse, do: 1, else: 0)]
    }
  end

  @doc """
  A window op: `:window_sum`, `:window_product`, `:window_max` or `:window_min`.
  `padding` is `{low, high}` per axis.
  """
  @spec window(
          atom(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          tuple(),
          [pos_integer()],
          [pos_integer()],
          [{integer(), integer()}]
        ) :: Instruction.t()
  def window(op, operand, type, shape, dims, strides, dilations, padding)
      when op in [:window_sum, :window_product, :window_max, :window_min] do
    %Instruction{
      op: op,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: window_attrs(dims, strides, dilations, padding)
    }
  end

  @doc """
  A generic `window_reduce`; `init` is a scalar ref that also pads the window.
  `window` is `{dims, strides, dilations, padding}`.
  """
  @spec window_reduce(
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          [{atom(), non_neg_integer(), non_neg_integer()}],
          {tuple(), [pos_integer()], [pos_integer()], [{integer(), integer()}]}
        ) :: Instruction.t()
  def window_reduce(operand, init, type, shape, steps, {dims, strides, dilations, padding}) do
    %Instruction{
      op: :window_reduce,
      operands: [operand, init],
      out_type: type,
      out_shape: shape,
      attrs: steps_attrs(steps) ++ window_attrs(dims, strides, dilations, padding)
    }
  end

  @doc """
  `:window_scatter_max` or `:window_scatter_min`: adds each `source` value at the
  position its window selects. `window` is `{dims, strides, padding}`.
  """
  @spec window_scatter(
          atom(),
          Instruction.ref(),
          Instruction.ref(),
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          {tuple(), [pos_integer()], [{integer(), integer()}]}
        ) :: Instruction.t()
  def window_scatter(op, operand, source, init, type, shape, {dims, strides, padding})
      when op in [:window_scatter_max, :window_scatter_min] do
    dilations = List.duplicate(1, tuple_size(dims))

    %Instruction{
      op: op,
      operands: [operand, source, init],
      out_type: type,
      out_shape: shape,
      attrs: window_attrs(dims, strides, dilations, padding)
    }
  end

  @doc "`:sort` or `:argsort` along `axis`; `direction` is `:asc` or `:desc`."
  @spec sort(
          :sort | :argsort,
          Instruction.ref(),
          Nx.Type.t(),
          tuple(),
          non_neg_integer(),
          :asc | :desc
        ) :: Instruction.t()
  def sort(op, operand, type, shape, axis, direction)
      when op in [:sort, :argsort] and direction in [:asc, :desc] do
    %Instruction{
      op: op,
      operands: [operand],
      out_type: type,
      out_shape: shape,
      attrs: [axis, direction]
    }
  end

  @doc """
  `gather`: the operands are the tensor and the indices `[batch..., length(axes)]`.
  `axes` are the tensor axes the indices address, sorted. The output is
  `[batch..., remaining tensor axes...]`.
  """
  @spec gather(Instruction.ref(), Instruction.ref(), Nx.Type.t(), tuple(), [non_neg_integer()]) ::
          Instruction.t()
  def gather(tensor, indices, type, shape, axes) do
    %Instruction{
      op: :gather,
      operands: [tensor, indices],
      out_type: type,
      out_shape: shape,
      attrs: axes
    }
  end

  @doc """
  `indexed_add` or `indexed_put` over `{target, indices, updates}`. Indices are
  `{n, length(axes)}` and updates `{n, window...}`; target and updates have the
  output type. A repeated index adds up (`indexed_add`) or keeps the last update
  (`indexed_put`).
  """
  @spec indexed(
          :indexed_add | :indexed_put,
          {Instruction.ref(), Instruction.ref(), Instruction.ref()},
          Nx.Type.t(),
          tuple(),
          [non_neg_integer()]
        ) :: Instruction.t()
  def indexed(op, {target, indices, updates}, type, shape, axes)
      when op in [:indexed_add, :indexed_put] do
    %Instruction{
      op: op,
      operands: [target, indices, updates],
      out_type: type,
      out_shape: shape,
      attrs: axes
    }
  end

  defp steps_attrs(steps),
    do: [length(steps) | Enum.flat_map(steps, fn {op, a, b} -> [op, a, b] end)]

  defp window_attrs(dims, strides, dilations, padding),
    do: Tuple.to_list(dims) ++ strides ++ dilations ++ Enum.flat_map(padding, &Tuple.to_list/1)

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
