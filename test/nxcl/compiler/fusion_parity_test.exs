defmodule NxCL.FusionParityTest do
  @moduledoc """
  Fused kernels against the unfused path. Each row wraps an elementwise op in a
  small chain (a broadcast input, an add, a multiply) so the op lands inside a
  `:fused` instruction, runs it with `fusion: false`, and holds the fused result
  to the harness tolerance for the output type: integers match exactly, floats
  within the per-type tolerance (NaN in the same places).

  The unfused path is the reference here because the elementwise, complex and
  cast rows already hold it to `Nx.BinaryBackend`. The rows cover every unary
  and binary elementwise op on each dtype Nx accepts for it, plus `select`,
  `clip` and `as_type`. A pair Nx rejects is left out, found by calling Nx.

  `f64` and `c128` rows need `cl_khr_fp64`; `f16` rows need `cl_khr_fp16`.
  """
  use ExUnit.Case, async: false

  import NxCL.Parity

  alias NxCL.Parity.Gen

  @moduletag :opencl

  @ints [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}, {:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]
  @floats [{:f, 16}, {:bf, 16}, {:f, 32}, {:f, 64}]
  @complex [{:c, 64}, {:c, 128}]
  @types @ints ++ @floats ++ @complex

  @unary ~w(exp expm1 log log1p sin cos tan sinh cosh tanh asin acos atan asinh acosh atanh
            sqrt rsqrt cbrt erf erfc erf_inv sigmoid abs negate sign ceil floor round
            bitwise_not count_leading_zeros population_count is_nan is_infinity logical_not
            real imag conjugate)a

  @binary ~w(add subtract multiply divide quotient remainder pow atan2 max min bitwise_and
             bitwise_or bitwise_xor left_shift right_shift equal not_equal greater
             greater_equal less less_equal logical_and logical_or logical_xor)a

  # Whether Nx accepts `op` on `type`, found by calling it on two small tensors.
  # A closure, because the `for` below runs while the module compiles.
  # `atan2` on a complex tensor traces in Nx but `NxCL.Lower` rejects it.
  accepted? = fn op, type, arity ->
    try do
      if op == :atan2 and elem(type, 0) == :c, do: raise("rejected by Lower")
      t = Nx.tensor([1, 2], type: type)
      apply(Nx, op, List.duplicate(t, arity))
      true
    rescue
      _ -> false
    end
  end

  tag_for = fn
    {:f, 64} -> :fp64
    {:c, 128} -> :fp64
    {:f, 16} -> :fp16
    _ -> nil
  end

  # Four rows of five: random values, then edge values (zeros, NaN, infinities,
  # integer limits).
  defp data(type, salt \\ 0) do
    {kind, bits} = type

    case kind do
      :c ->
        part = {:f, div(bits, 2)}
        Nx.complex(data(part, salt), data(part, salt + 1) |> Nx.reverse())

      _ ->
        edge =
          Nx.concatenate([Gen.edge_values(type), Gen.edge_values(type), Gen.random({10}, type)])

        edge = Nx.reshape(Nx.slice(edge, [0], [10]), {2, 5})
        rand = Gen.random({2, 5}, type)
        data = Nx.concatenate([rand, edge])
        if salt == 0, do: data, else: Nx.reverse(data)
    end
  end

  # A broadcast input of the result type, `{5}` against the `{4, 5}` result.
  defp tail(r) do
    c = Nx.iota({5}, type: :s32) |> Nx.as_type(Nx.type(r))
    r |> Nx.add(c) |> Nx.multiply(2)
  end

  defp assert_fused(label, fun, args) do
    expected = apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler, fusion: false), args)

    assert_expected(label, fun, args, expected,
      compiler_opts: [fusion: true],
      record: false,
      backend: false
    )
  end

  describe "unary ops inside a fused group" do
    for op <- @unary, type <- @types, accepted?.(op, type, 1) do
      if tag_for.(type), do: @tag(requires: tag_for.(type))

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        assert_fused(op, fn x -> tail(apply(Nx, op, [x])) end, [data(type)])
      end
    end
  end

  describe "binary ops inside a fused group" do
    for op <- @binary, type <- @types, accepted?.(op, type, 2) do
      if tag_for.(type), do: @tag(requires: tag_for.(type))

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        a = data(type)
        # No zero divisors: integer division by zero is covered by the elementwise rows.
        b =
          if op in [:divide, :quotient, :remainder],
            do: nonzero(data(type, 1)),
            else: data(type, 1)

        assert_fused(op, fn x, y -> tail(apply(Nx, op, [x, y])) end, [a, b])
      end
    end
  end

  defp nonzero(%Nx.Tensor{type: type} = t),
    do: Nx.select(Nx.equal(t, 0), Nx.tensor(1, type: type), t)

  describe "select, clip and as_type" do
    for type <- @ints ++ [{:bf, 16}, {:f, 32}] do
      test "select #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_fused(
          :select,
          fn p, a, b -> tail(Nx.select(Nx.greater(p, 0), a, b)) end,
          [data(type), data(type, 1), data(type) |> Nx.reverse()]
        )
      end

      test "clip #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        assert_fused(:clip, fn x -> tail(Nx.clip(x, 1, 100)) end, [data(type)])
      end
    end

    for from <- @ints ++ [{:bf, 16}, {:f, 32}],
        to <- @ints ++ [{:bf, 16}, {:f, 32}],
        from != to do
      test "as_type #{Nx.Type.to_string(from)} to #{Nx.Type.to_string(to)}" do
        from = unquote(Macro.escape(from))
        to = unquote(Macro.escape(to))
        assert_fused(:as_type, fn x -> tail(Nx.as_type(x, to)) end, [data(from)])
      end
    end
  end

  describe "chains" do
    test "a transformer-style chain with constants, broadcasts and selects" do
      fun = fn x, g, b ->
        mean = Nx.mean(x, axes: [-1], keep_axes: true)
        c = Nx.subtract(x, mean)

        s =
          c
          |> Nx.multiply(
            Nx.rsqrt(Nx.add(Nx.mean(Nx.multiply(c, c), axes: [-1], keep_axes: true), 1.0e-5))
          )

        y = s |> Nx.multiply(g) |> Nx.add(b)
        Nx.select(Nx.greater(y, 0), Nx.multiply(y, 0.5), Nx.tanh(y))
      end

      x = Gen.random({4, 16}, {:f, 32})
      assert_fused(:chain, fun, [x, Gen.random({16}, {:f, 32}), Gen.random({16}, {:f, 32})])
    end
  end
end
