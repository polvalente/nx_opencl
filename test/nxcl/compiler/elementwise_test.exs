defmodule NxCL.ElementwiseTest do
  @moduledoc """
  Differential rows for every elementwise op on each dtype Nx accepts for it. A
  second group holds one row per rule (NaN, overflow, zero divisor, shift width,
  integer division and remainder, casts). Where `BinaryBackend` raises or is wrong,
  the row states the expected values.
  """
  use ExUnit.Case, async: false

  import Bitwise
  import NxCL.TestSupport.Compare
  import NxCL.TestSupport.Device, only: [capability?: 1]

  alias NxCL.TestSupport.Gen

  @moduletag :opencl

  @uints [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}]
  @sints [{:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]
  @ints @uints ++ @sints
  @floats [{:f, 16}, {:bf, 16}, {:f, 32}, {:f, 64}]
  @all @ints ++ @floats

  # Tolerances for a result the reference gets wrong or only roughly right. An
  # entry needs a note on why. `BinaryBackend` computes `erf_inv` of an `f64` to
  # single precision (about 6e-8 off); the "erf_inv f64 inverts erf" row checks
  # our value against `erf` at full precision.
  @tolerances %{{:erf_inv, {:f, 64}} => [rtol: 1.0e-6, atol: 1.0e-6]}

  defp tol(op, type), do: Map.get(@tolerances, {op, type}, [])

  defp data(type, shape \\ {65}) do
    Nx.concatenate([Gen.random(shape, type), Gen.edge_values(type)])
  end

  # Input for the float functions. The infinities are left out: `BinaryBackend`
  # raises on some (`log`) and is wrong on others (`tanh(inf)` is NaN there); a
  # rule row below states the expected values. Integers of 32 bits or more stay
  # at or below 2^24: the kernel converts them to `f32` first, the reference
  # uses the exact integer, and `sin` of a huge value tells them apart.
  defp fn_data({kind, _} = type) when kind in [:f, :bf] do
    Nx.concatenate([Gen.random({65}, type), Nx.tensor([0.0, -0.0, 1.0, -1.0, :nan], type: type)])
  end

  defp fn_data({_, bits} = type) when bits < 32, do: data(type)

  defp fn_data(type) do
    Nx.concatenate([Gen.random({65}, type), Nx.tensor([0, 1, 16_777_216], type: type)])
  end

  # `atanh` is real only on (-1, 1); outside it the reference raises.
  defp op_data(:atanh, {kind, _} = type) when kind in [:f, :bf] do
    Nx.concatenate([
      Nx.divide(Gen.random({65}, type), Nx.tensor(4.0, type: type)),
      Nx.tensor([0.0, -0.0], type: type)
    ])
  end

  defp op_data(:atanh, type), do: Nx.tensor([0, 0, 0], type: type)
  defp op_data(_op, type), do: fn_data(type)

  defp right_operand(:pow, type), do: Nx.abs(Gen.random({65}, type))
  defp right_operand(_op, type), do: Gen.random({65}, type)

  defp wide(type, n \\ 40) do
    {kind, bits} = type
    :rand.seed(:exsss, {seed(), bits, n})

    {lo, hi} =
      if kind == :s,
        do: {-(1 <<< (bits - 1)), (1 <<< (bits - 1)) - 1},
        else: {0, (1 <<< bits) - 1}

    Nx.tensor(for(_ <- 1..n, do: lo + :rand.uniform(hi - lo + 1) - 1), type: type)
  end

  defp nonzero(%Nx.Tensor{type: type} = t) do
    Nx.select(Nx.equal(t, 0), Nx.tensor(1, type: type), t)
  end

  defp bits({_, bits}), do: bits

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  defp assert_list(tensor, expected) do
    assert Nx.to_flat_list(tensor) == expected
  end

  @float_fns ~w(exp expm1 log log1p sin cos tan sinh cosh tanh asin acos atan asinh acosh atanh
                sqrt rsqrt cbrt erf erfc sigmoid)a

  describe "float functions" do
    for op <- @float_fns, type <- @floats ++ @ints do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))

        assert_same(fn x -> apply(Nx, op, [x]) end, [op_data(op, type)], tol(op, type))
      end
    end

    # `erf_inv` is defined on (-1, 1). Integers give 0 only.
    for type <- @floats do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "erf_inv #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Nx.divide(Gen.random({65}, type), Nx.tensor(4.0, type: type))
        assert_same(&Nx.erf_inv/1, [x], tol(:erf_inv, type))
      end
    end

    @tag requires: :fp64
    test "erf_inv f64 inverts erf at full precision" do
      y = Nx.divide(Gen.random({65}, {:f, 64}), Nx.tensor(4.0, type: :f64))
      x = run(&Nx.erf_inv/1, [y])
      assert x.type == {:f, 64}
      back = Nx.erf(Nx.backend_copy(x, Nx.BinaryBackend))
      assert Nx.all_close(back, y, rtol: 1.0e-14, atol: 1.0e-15) |> Nx.to_number() == 1
    end

    for type <- @ints do
      test "erf_inv #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        assert_same(&Nx.erf_inv/1, [Nx.tensor([0, 0], type: type)])
      end
    end
  end

  describe "unary" do
    for op <- ~w(abs negate sign ceil floor round)a, type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        {kind, _} = type

        # `BinaryBackend` reads NaN and the infinities as atoms that sort above
        # numbers, so its `sign` is wrong for them and `round`, `ceil` and
        # `floor` raise. The rules below state those cases.
        x =
          if kind in [:f, :bf] and op in [:sign, :round, :ceil, :floor],
            do: Gen.random({65}, type),
            else: data(type)

        assert_same(fn x -> apply(Nx, op, [x]) end, [x], tol(op, type))
      end
    end

    for op <- ~w(bitwise_not count_leading_zeros population_count)a, type <- @ints do
      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))

        assert_same(fn x -> apply(Nx, op, [x]) end, [
          Nx.concatenate([data(type), wide(type)])
        ])
      end
    end

    for op <- ~w(is_nan is_infinity)a, type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        assert_same(fn x -> apply(Nx, op, [x]) end, [data(type)])
      end
    end

    for type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "logical_not #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        assert_same(&Nx.logical_not/1, [data(type)])
      end
    end
  end

  describe "binary" do
    for op <- ~w(add subtract multiply max min)a, type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        {kind, _} = type
        a = if kind in [:s, :u], do: Nx.concatenate([data(type), wide(type)]), else: data(type)

        b =
          if kind in [:s, :u],
            do: Nx.concatenate([data(type), Nx.reverse(wide(type))]),
            else: Nx.reverse(data(type))

        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b], tol(op, type))
      end
    end

    for type <- @floats ++ @ints do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "divide #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        {kind, _} = type
        a = data(type)
        # Integer zero divisors give inf in the float result; both sides agree.
        b = if kind in [:s, :u], do: Nx.reverse(a), else: Nx.reverse(a)
        assert_same(&Nx.divide/2, [a, b], tol(:divide, type))
      end
    end

    for type <- @ints do
      test "atan2 #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = fn_data(type)
        assert_same(&Nx.atan2/2, [a, Nx.reverse(a)])
      end
    end

    for type <- @floats do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "atan2 #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Gen.random({65}, type)
        b = Nx.reverse(Gen.random({65}, type))
        assert_same(&Nx.atan2/2, [a, b], tol(:atan2, type))
      end
    end

    # Floats: `BinaryBackend` raises on a zero divisor, so the divisor has no zero.
    for type <- @floats do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      if type == {:f, 64}, do: @tag(requires: :fp64)

      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "remainder #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Gen.random({65}, type)
        b = Nx.reverse(nonzero(Gen.random({65}, type)))
        assert_same(&Nx.remainder/2, [a, b], tol(:remainder, type))
      end

      if type == {:f, 64}, do: @tag(requires: :fp64)

      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "pow #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Gen.random({65}, type)
        b = Nx.reverse(Gen.random({65}, type))
        assert_same(&Nx.pow/2, [a, b], tol(:pow, type))
      end
    end

    # Integers: zero divisors and negative exponents have their own rules below.
    for type <- @ints do
      test "quotient #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Nx.concatenate([data(type), wide(type)])
        b = Nx.concatenate([nonzero(Nx.reverse(data(type))), nonzero(Nx.reverse(wide(type)))])
        assert_same(&Nx.quotient/2, [a, b])
      end

      test "remainder #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Nx.concatenate([data(type), wide(type)])
        b = Nx.concatenate([nonzero(Nx.reverse(data(type))), nonzero(Nx.reverse(wide(type)))])
        assert_same(&Nx.remainder/2, [a, b])
      end

      test "pow #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = Nx.concatenate([data(type), wide(type)])

        exps =
          Nx.bitwise_and(Nx.concatenate([data(type), wide(type)]), Nx.tensor(63, type: type))

        assert_same(&Nx.pow/2, [a, Nx.reverse(exps)])
      end

      for op <- ~w(bitwise_and bitwise_or bitwise_xor)a do
        test "#{op} #{Nx.Type.to_string(type)}" do
          op = unquote(op)
          type = unquote(Macro.escape(type))
          a = Nx.concatenate([data(type), wide(type)])
          b = Nx.reverse(a)
          assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
        end
      end

      for op <- ~w(left_shift right_shift)a do
        test "#{op} #{Nx.Type.to_string(type)}" do
          op = unquote(op)
          type = unquote(Macro.escape(type))
          a = Nx.concatenate([data(type), wide(type)])
          # Counts from 0 to twice the width: both sides of the guard.
          n = Nx.size(a)

          counts =
            Nx.remainder(Nx.iota({n}, type: :s32), 2 * bits(type) + 1) |> Nx.as_type(type)

          assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, counts])
        end
      end
    end

    for op <-
          ~w(equal not_equal greater greater_equal less less_equal logical_and logical_or logical_xor)a,
        type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        a = data(type)
        # Same values, shifted by one place: equal and unequal pairs both occur.
        b = Nx.concatenate([a[1..-1//1], a[0..0]])
        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, a])
      end
    end
  end

  describe "mixed operand types" do
    # `Nx.add(u8, s8)` reaches the compiler as an `add` of `u8` and `s8` with an
    # `s16` result. No `as_type` node comes first. Codegen reads each operand in
    # the type the op computes in.
    @pairs [
      {{:u, 8}, {:s, 8}},
      {{:u, 8}, {:s, 32}},
      {{:s, 16}, {:u, 16}},
      {{:s, 32}, {:f, 32}},
      {{:u, 8}, {:f, 32}},
      {{:f, 16}, {:f, 32}},
      {{:bf, 16}, {:f, 32}},
      {{:f, 16}, {:bf, 16}},
      {{:s, 64}, {:f, 32}},
      {{:u, 64}, {:s, 64}}
    ]

    for op <-
          ~w(add subtract multiply max min divide pow atan2 equal not_equal greater greater_equal less less_equal logical_and logical_or logical_xor)a,
        {ta, tb} <- @pairs do
      if {:f, 16} in [ta, tb], do: @tag(requires: :fp16)

      test "#{op} #{Nx.Type.to_string(ta)} with #{Nx.Type.to_string(tb)}" do
        op = unquote(op)
        ta = unquote(Macro.escape(ta))
        tb = unquote(Macro.escape(tb))
        a = Gen.random({65}, ta)
        # Positive exponents keep `pow` away from the integer cases that raise.
        b = right_operand(op, tb)
        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
      end
    end

    for op <- ~w(bitwise_and bitwise_or bitwise_xor quotient remainder)a,
        {ta, tb} <- [{{:u, 8}, {:s, 8}}, {{:s, 16}, {:u, 16}}, {{:u, 8}, {:s, 32}}] do
      test "#{op} #{Nx.Type.to_string(ta)} with #{Nx.Type.to_string(tb)}" do
        op = unquote(op)
        ta = unquote(Macro.escape(ta))
        tb = unquote(Macro.escape(tb))
        a = Gen.random({65}, ta)
        b = nonzero(Gen.random({65}, tb))
        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
      end
    end
  end

  describe "scalar constants" do
    test "a constant of another type: u8 plus 300 wraps" do
      assert_same(fn x -> Nx.add(x, 300) end, [Nx.tensor([1, 2, 255], type: :u8)])
    end

    for {op, c} <-
          [
            add: 2,
            subtract: 3,
            multiply: -2,
            divide: 4,
            max: 1,
            min: 1,
            pow: 2,
            atan2: 2,
            remainder: 3
          ] do
      test "#{op} with a float constant" do
        op = unquote(op)
        c = unquote(c) * 1.0
        assert_same(fn x -> apply(Nx, op, [x, c]) end, [Gen.random({65}, {:f, 32})])
        assert_same(fn x -> apply(Nx, op, [c, x]) end, [Gen.random({65}, {:f, 32})])
      end
    end

    test "NaN and infinity constants" do
      x = Gen.random({65}, {:f, 32})
      assert_same(fn x -> Nx.add(x, :nan) end, [x])
      assert_same(fn x -> Nx.max(x, :infinity) end, [x])
      assert_same(fn x -> Nx.min(x, :neg_infinity) end, [x])
    end

    for type <- @ints do
      test "integer constants on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Nx.concatenate([data(type), wide(type)])
        assert_same(fn x -> Nx.add(x, 7) end, [x])
        assert_same(fn x -> Nx.multiply(x, 3) end, [x])
        assert_same(fn x -> Nx.bitwise_and(x, 5) end, [x])
        assert_same(fn x -> Nx.left_shift(x, 2) end, [x])
        assert_same(fn x -> Nx.less(x, 3) end, [x])
      end
    end

    test "select and clip with scalar constants" do
      x = data({:f, 32})
      assert_same(fn x -> Nx.select(Nx.greater(x, 0), x, 0.5) end, [x])
      assert_same(fn x -> Nx.select(Nx.greater(x, 0), 1, x) end, [x])
      assert_same(fn x -> Nx.clip(x, -1.0, 1.0) end, [Gen.random({65}, {:f, 32})])
      assert_same(fn x -> Nx.clip(x, 2, 5) end, [Nx.iota({10}, type: :s32)])
    end

    test "a scalar tensor mixed with a vector gets an explicit broadcast" do
      assert_same(fn x, s -> Nx.add(x, s) end, [Nx.tensor([1.0, 2.0]), Nx.tensor(1.0)])
    end
  end

  describe "select and clip" do
    for type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      if type == {:f, 64}, do: @tag(requires: :fp64)

      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "select #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = data(type)
        b = Nx.reverse(a)
        pred = Nx.greater(Nx.iota(Nx.shape(a), type: :s32), 30)
        assert_same(&Nx.select/3, [pred, a, b])
      end

      if type == {:f, 64}, do: @tag(requires: :fp64)

      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "clip #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        {kind, _} = type
        x = if kind in [:f, :bf], do: Gen.random({65}, type), else: data(type)
        assert_same(fn x -> Nx.clip(x, 1, 5) end, [x])
      end
    end

    test "select with a float predicate and mixed branch types" do
      p = Nx.tensor([0.0, 1.5, :nan, -0.0], type: :f32)
      a = Nx.tensor([1, 2, 3, 4], type: :s32)
      b = Nx.tensor([10.5, 20.5, 30.5, 40.5], type: :f32)
      assert_same(&Nx.select/3, [p, a, b])
    end
  end

  describe "as_type" do
    for from <- @all, to <- @all do
      if {:f, 64} in [from, to], do: @tag(requires: :fp64)
      if {:f, 16} in [from, to], do: @tag(requires: :fp16)

      test "#{Nx.Type.to_string(from)} to #{Nx.Type.to_string(to)}" do
        from = unquote(Macro.escape(from))
        to = unquote(Macro.escape(to))
        {kind, _} = from

        # Finite values only: the rules below cover NaN, infinity and range.
        x =
          cond do
            kind in [:s, :u] ->
              Nx.concatenate([Gen.random({65}, from), Gen.edge_values(from)])

            true ->
              Nx.concatenate([
                Gen.random({65}, from),
                Nx.tensor([0.0, 1.0, -1.0, 250.7], type: from)
              ])
          end

        assert_same(fn x -> Nx.as_type(x, to) end, [x])
      end
    end

    for {a, b} <-
          [
            {{:f, 32}, {:s, 32}},
            {{:f, 32}, {:u, 32}},
            {{:u, 32}, {:s, 32}},
            {{:f, 64}, {:s, 64}},
            {{:f, 64}, {:u, 64}},
            {{:s, 64}, {:u, 64}},
            {{:f, 16}, {:bf, 16}},
            {{:f, 16}, {:s, 16}},
            {{:bf, 16}, {:u, 16}},
            {{:s, 8}, {:u, 8}}
          ] do
      test "bitcast #{Nx.Type.to_string(a)} to #{Nx.Type.to_string(b)}" do
        a = unquote(Macro.escape(a))
        b = unquote(Macro.escape(b))
        {kind, _} = a
        x = if kind in [:f, :bf], do: Gen.random({65}, a), else: data(a)
        assert_same(fn x -> Nx.bitcast(x, b) end, [x])
        assert_same(fn x -> x |> Nx.bitcast(b) |> Nx.bitcast(a) end, [x])
      end
    end
  end

  describe "native dtype buffers" do
    for type <- @all do
      if type == {:f, 64}, do: @tag(requires: :fp64)
      if type == {:f, 16}, do: @tag(requires: :fp16)

      test "parameter, tensor, constant, reshape and squeeze keep #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data(type)
        assert_same(fn x -> x end, [x])
        literal = Nx.reverse(x)
        assert_same(fn x -> Nx.add(x, literal) end, [x])
        assert_same(fn x -> Nx.add(x, Nx.tensor(1, type: type)) end, [x])
        assert_same(fn x -> Nx.reshape(x, {Nx.size(x), 1}) end, [x])
        assert_same(fn x -> x |> Nx.reshape({1, Nx.size(x)}) |> Nx.squeeze() end, [x])
      end
    end

    test "s32 add returns s32 and u8 add wraps" do
      s = run(&Nx.add/2, [Nx.tensor([1, 2], type: :s32), Nx.tensor([3, 4], type: :s32)])
      assert s.type == {:s, 32}
      assert_list(s, [4, 6])

      u = run(&Nx.add/2, [Nx.tensor([200, 255], type: :u8), Nx.tensor([100, 1], type: :u8)])
      assert u.type == {:u, 8}
      assert_list(u, [44, 0])
    end

    test "bf16 round-trips" do
      x = Nx.tensor([1.0, -2.5, 3.140625, 0.0, 65280.0, 1.0e-30], type: :bf16)
      y = run(fn x -> x end, [x])
      assert y.type == {:bf, 16}
      assert Nx.to_binary(y) == Nx.to_binary(x)
      assert Nx.to_flat_list(run(&Nx.negate/1, [x])) == Nx.to_flat_list(Nx.negate(x))
    end
  end

  describe "rule: NaN" do
    test "max and min return NaN if either side is NaN" do
      a = Nx.tensor([:nan, 1.0, :nan, 2.0])
      b = Nx.tensor([1.0, :nan, :nan, 3.0])
      assert_same(&Nx.max/2, [a, b])
      assert_same(&Nx.min/2, [a, b])
    end

    test "comparisons with NaN are false, not_equal is true" do
      a = Nx.tensor([:nan, 1.0, :nan, :infinity])
      b = Nx.tensor([1.0, :nan, :nan, :infinity])

      for op <- ~w(equal not_equal greater greater_equal less less_equal)a do
        assert_same(fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
      end
    end

    test "NaN counts as true in the logical ops and select" do
      a = Nx.tensor([:nan, 0.0, -0.0, 1.0])
      b = Nx.tensor([1.0, :nan, 0.0, 0.0])
      assert_same(&Nx.logical_and/2, [a, b])
      assert_same(&Nx.logical_or/2, [a, b])
      assert_same(&Nx.logical_xor/2, [a, b])
      assert_same(&Nx.select/3, [a, Nx.tensor([1, 2, 3, 4]), Nx.tensor([5, 6, 7, 8])])
    end

    test "is_nan and is_infinity find NaN and the infinities" do
      x = Nx.tensor([:nan, :infinity, :neg_infinity, 1.0, 0.0])
      assert_same(&Nx.is_nan/1, [x])
      assert_same(&Nx.is_infinity/1, [x])
    end

    test "clip turns NaN into the lower bound, as BinaryBackend does" do
      x = Nx.tensor([:nan, 5.0, -5.0, :infinity, :neg_infinity, 0.5])
      assert_same(fn x -> Nx.clip(x, -1.0, 1.0) end, [x])
    end

    test "NaN input to a float function gives NaN" do
      x = Nx.tensor([:nan, 1.0])

      for op <- ~w(exp log sin tanh sqrt sigmoid)a do
        assert_same(fn x -> apply(Nx, op, [x]) end, [x])
      end
    end

    test "sign keeps NaN and signed zero, and gives -1 for -inf (BinaryBackend does not)" do
      x = Nx.tensor([:nan, :infinity, :neg_infinity, -0.0, 0.0, -3.0, 2.0])
      y = run(&Nx.sign/1, [x])
      assert [:nan, 1.0, -1.0, _, zero, -1.0, 1.0] = Nx.to_flat_list(y)
      assert zero == 0.0
      <<_::binary-size(12), minus_zero::binary-size(4), _::binary>> = Nx.to_binary(y)
      assert minus_zero == <<-0.0::float-32-native>>
    end

    test "infinities and NaN through float functions and rounding (BinaryBackend raises or errs)" do
      check = fn op, input, expected ->
        out = run(fn x -> apply(Nx, op, [x]) end, [Nx.tensor(input)])
        assert Nx.to_flat_list(out) == expected, "#{op}"
      end

      inf = :infinity
      ninf = :neg_infinity
      check.(:exp, [inf, ninf], [inf, 0.0])
      check.(:expm1, [inf, ninf], [inf, -1.0])
      check.(:tanh, [inf, ninf], [1.0, -1.0])
      check.(:sigmoid, [inf, ninf], [1.0, 0.0])
      check.(:sqrt, [inf, :nan], [inf, :nan])
      check.(:rsqrt, [inf, 0.0], [0.0, inf])
      check.(:log, [inf, 0.0], [inf, ninf])
      check.(:log1p, [inf, -1.0], [inf, ninf])
      check.(:cosh, [inf, ninf], [inf, inf])
      check.(:sinh, [inf, ninf], [inf, ninf])
      check.(:erf, [inf, ninf], [1.0, -1.0])
      check.(:erfc, [inf, ninf], [0.0, 2.0])
      check.(:cbrt, [inf, ninf], [inf, ninf])
      check.(:atanh, [1.0, -1.0], [inf, ninf])
      check.(:asinh, [inf, ninf], [inf, ninf])
      check.(:acosh, [inf, 1.0], [inf, 0.0])

      for op <- ~w(floor ceil round)a do
        check.(op, [:nan, inf, ninf, 0.0], [:nan, inf, ninf, 0.0])
      end
    end

    test "integer-only ops on a float tensor raise in Nx before the compiler runs" do
      for op <- ~w(bitwise_and bitwise_or bitwise_xor left_shift right_shift quotient)a do
        assert_raise ArgumentError, fn ->
          Nx.Defn.jit_apply(fn x -> apply(Nx, op, [x, x]) end, [Nx.tensor([1.0])],
            compiler: NxCL.Compiler
          )
        end
      end

      for op <- ~w(bitwise_not count_leading_zeros population_count)a do
        assert_raise ArgumentError, fn ->
          Nx.Defn.jit_apply(fn x -> apply(Nx, op, [x]) end, [Nx.tensor([1.0])],
            compiler: NxCL.Compiler
          )
        end
      end
    end

    test "NaN cast to an integer gives 0" do
      x = Nx.tensor([:nan, 1.5])
      assert_same(fn x -> Nx.as_type(x, :s32) end, [x])
      assert_same(fn x -> Nx.as_type(x, :u8) end, [x])
    end
  end

  describe "rule: overflow" do
    test "u8 add and multiply wrap" do
      a = Nx.tensor([200, 255, 16, 0], type: :u8)
      b = Nx.tensor([100, 1, 17, 255], type: :u8)
      assert_same(&Nx.add/2, [a, b])
      assert_same(&Nx.multiply/2, [a, b])
      assert_list(run(&Nx.add/2, [a, b]), [44, 0, 33, 255])

      assert_list(
        run(&Nx.subtract/2, [Nx.tensor([0, 1], type: :u8), Nx.tensor([1, 2], type: :u8)]),
        [255, 255]
      )
    end

    test "signed add, subtract, multiply and negate wrap at every width" do
      for type <- @sints do
        max = Nx.to_number(Nx.Constants.max_finite(type))
        min = Nx.to_number(Nx.Constants.min_finite(type))
        a = Nx.tensor([max, min, max, min, 3], type: type)
        b = Nx.tensor([1, -1, max, min, -4], type: type)
        assert_same(&Nx.add/2, [a, b])
        assert_same(&Nx.subtract/2, [a, b])
        assert_same(&Nx.multiply/2, [a, b])
        assert_same(&Nx.negate/1, [a])
        assert_same(&Nx.abs/1, [a])
      end
    end

    test "u16 multiply stays correct above the signed 32-bit range" do
      a = Nx.tensor([65535, 65535, 40000], type: :u16)
      assert_same(&Nx.multiply/2, [a, a])
    end

    test "float overflow gives infinity" do
      a = Nx.tensor([3.0e38, -3.0e38, 1.0e30], type: :f32)
      assert_same(&Nx.multiply/2, [a, a])
      assert_same(&Nx.add/2, [a, a])
      assert_same(&Nx.exp/1, [Nx.tensor([100.0, -200.0, 88.0])])
    end

    test "casts: float out of range wraps, infinity saturates, int narrows by wrapping" do
      f = Nx.tensor([1.0e10, -1.0e10, 3.0e9, 300.7, -1.5, 1.0e30, :infinity, :neg_infinity])

      for to <- [:s8, :u8, :s16, :u16, :s32, :u32, :s64, :u64] do
        assert_same(fn x -> Nx.as_type(x, to) end, [f])
      end

      wide = Nx.tensor([1.0e19, -1.0e19, 1.0e20, 2.0e19], type: :f32)
      assert_same(fn x -> Nx.as_type(x, :u64) end, [wide])
      assert_same(fn x -> Nx.as_type(x, :s64) end, [wide])

      for from <- @ints, to <- @ints do
        x = Nx.concatenate([data(from), wide(from)])
        assert_same(fn x -> Nx.as_type(x, to) end, [x])
      end
    end

    @half_cast_values [
      1.0e-8,
      6.0e-8,
      3.0e-5,
      65504.0,
      65519.0,
      65520.0,
      7.0e4,
      1.0e30,
      0.1,
      -0.1,
      1.00048828125,
      1.0009765625
    ]

    @tag requires: :fp16
    test "casts to f16 round to nearest even and overflow to infinity" do
      x = Nx.tensor(@half_cast_values, type: :f32)

      assert_same(fn x -> Nx.as_type(x, :f16) end, [x])
      assert_same(fn x -> x |> Nx.as_type(:f16) |> Nx.as_type(:f32) end, [x])
    end

    test "casts to bf16 round to nearest even and overflow to infinity" do
      x = Nx.tensor(@half_cast_values, type: :f32)

      assert_same(fn x -> Nx.as_type(x, :bf16) end, [x])
    end

    test "casts to bf16 round to nearest even; BinaryBackend truncates, so the row states the values" do
      x =
        Nx.tensor(
          [65504.0, 1.00390625, 1.01171875, 0.1, -0.1, :nan, :infinity, 3.4028235e38],
          type: :f32
        )

      y = run(fn x -> Nx.as_type(x, :bf16) end, [x])
      assert y.type == {:bf, 16}

      assert Nx.to_flat_list(y) ==
               [65536.0, 1.0, 1.015625, 0.10009765625, -0.10009765625, :nan, :infinity, :infinity]
    end

    @tag requires: :fp16
    test "f16 values read back exactly, including subnormals" do
      x =
        Nx.iota({1024}, type: :s32) |> Nx.multiply(Nx.tensor(61, type: :s32)) |> Nx.as_type(:u16)

      f = Nx.bitcast(x, :f16)
      finite = Nx.select(Nx.is_nan(f), Nx.tensor(0.0, type: :f16), f)
      assert_same(fn x -> Nx.as_type(x, :f32) end, [finite])
    end

    test "integer pow wraps" do
      a = Nx.tensor([3, 2, -3, 7], type: :s32)
      b = Nx.tensor([40, 31, 21, 20], type: :s32)
      assert_same(&Nx.pow/2, [a, b])
      a8 = Nx.tensor([3, 2, 255, 7], type: :u8)
      b8 = Nx.tensor([5, 9, 3, 4], type: :u8)
      assert_same(&Nx.pow/2, [a8, b8])
    end
  end

  describe "rule: zero divisor" do
    test "float divide by zero gives inf and NaN" do
      a = Nx.tensor([1.0, -1.0, 0.0, :nan])
      b = Nx.tensor([0.0, 0.0, 0.0, 0.0])
      assert_same(&Nx.divide/2, [a, b])
      assert_list(run(&Nx.divide/2, [a, b]), [:infinity, :neg_infinity, :nan, :nan])
    end

    test "integer divide by zero goes through float, so it gives inf" do
      a = Nx.tensor([1, -1, 0, 7], type: :s32)
      b = Nx.tensor([0, 0, 0, 2], type: :s32)
      assert_same(&Nx.divide/2, [a, b])
    end

    # `BinaryBackend` raises on these, so the rows state the device's rule:
    # quotient by zero is all ones, remainder by zero is the dividend.
    test "integer quotient by zero is all ones; remainder by zero is the dividend" do
      a = Nx.tensor([5, -5, 0, 127], type: :s8)
      z = Nx.tensor([0, 0, 0, 0], type: :s8)
      assert_list(run(&Nx.quotient/2, [a, z]), [-1, -1, -1, -1])
      assert_list(run(&Nx.remainder/2, [a, z]), [5, -5, 0, 127])

      ua = Nx.tensor([7, 255, 0], type: :u8)
      uz = Nx.tensor([0, 0, 0], type: :u8)
      assert_list(run(&Nx.quotient/2, [ua, uz]), [255, 255, 255])
      assert_list(run(&Nx.remainder/2, [ua, uz]), [7, 255, 0])

      for type <- [{:s, 16}, {:s, 32}, {:s, 64}, {:u, 16}, {:u, 32}, {:u, 64}] do
        a = Nx.tensor([9, 1], type: type)
        z = Nx.tensor([0, 0], type: type)
        max = if elem(type, 0) == :s, do: -1, else: Nx.to_number(Nx.Constants.max_finite(type))
        assert_list(run(&Nx.quotient/2, [a, z]), [max, max])
        assert_list(run(&Nx.remainder/2, [a, z]), [9, 1])
      end
    end

    test "float remainder by zero is NaN" do
      a = Nx.tensor([5.0, -5.0, 0.0])
      z = Nx.tensor([0.0, 0.0, 0.0])
      assert_list(run(&Nx.remainder/2, [a, z]), [:nan, :nan, :nan])
    end

    test "integer pow with a negative exponent: 1 and -1 keep their value, the rest give 0" do
      a = Nx.tensor([1, -1, -1, 2, 0, 5], type: :s32)
      e = Nx.tensor([-3, -3, -2, -1, -1, -4], type: :s32)
      assert_list(run(&Nx.pow/2, [a, e]), [1, -1, 1, 0, 0, 0])
    end
  end

  describe "rule: integer division and remainder" do
    test "quotient truncates toward zero; remainder takes the sign of the dividend" do
      a = Nx.tensor([7, -7, 7, -7, 6, -6, 0], type: :s32)
      b = Nx.tensor([2, 2, -2, -2, 3, 3, 5], type: :s32)
      assert_same(&Nx.quotient/2, [a, b])
      assert_same(&Nx.remainder/2, [a, b])
      assert_list(run(&Nx.quotient/2, [a, b]), [3, -3, -3, 3, 2, -2, 0])
      assert_list(run(&Nx.remainder/2, [a, b]), [1, -1, 1, -1, 0, 0, 0])
    end

    test "the smallest value divided by -1 wraps; the remainder is 0" do
      for type <- @sints do
        min = Nx.to_number(Nx.Constants.min_finite(type))
        a = Nx.tensor([min, min], type: type)
        b = Nx.tensor([-1, 1], type: type)
        assert_list(run(&Nx.quotient/2, [a, b]), [min, min])
        assert_list(run(&Nx.remainder/2, [a, b]), [0, 0])
      end
    end

    test "unsigned quotient and remainder at the top of the range" do
      for type <- @uints do
        max = Nx.to_number(Nx.Constants.max_finite(type))
        a = Nx.tensor([max, max, 1], type: type)
        b = Nx.tensor([2, max, max], type: type)
        assert_same(&Nx.quotient/2, [a, b])
        assert_same(&Nx.remainder/2, [a, b])
      end
    end

    test "float remainder takes the sign of the dividend" do
      a = Nx.tensor([5.5, -5.5, 5.5, -5.5])
      b = Nx.tensor([2.0, 2.0, -2.0, -2.0])
      assert_same(&Nx.remainder/2, [a, b])
    end
  end

  describe "rule: shift width" do
    test "counts at or above the width give 0 on left shifts and the sign fill on right shifts" do
      for type <- @ints do
        w = bits(type)
        counts = Nx.tensor([0, 1, w - 1, w, w + 1, 2 * w - 1], type: type)
        max = Nx.to_number(Nx.Constants.max_finite(type))
        min = Nx.to_number(Nx.Constants.min_finite(type))

        for v <- Enum.uniq([max, min]) do
          a = Nx.broadcast(Nx.tensor(v, type: type), {6})
          assert_same(&Nx.left_shift/2, [a, counts])
          assert_same(&Nx.right_shift/2, [a, counts])
        end
      end
    end

    test "signed right shift is arithmetic" do
      a = Nx.tensor([-128, -1, 127, -64, 64], type: :s8)
      b = Nx.tensor([2, 1, 3, 6, 6], type: :s8)
      assert_same(&Nx.right_shift/2, [a, b])
      assert_list(run(&Nx.right_shift/2, [a, b]), [-32, -1, 15, -1, 1])
    end

    test "left shift keeps only the bits that fit" do
      a = Nx.tensor([-1, 100, 1, 255], type: :s16)
      b = Nx.tensor([15, 1, 15, 12], type: :s16)
      assert_same(&Nx.left_shift/2, [a, b])
    end
  end

  describe "rule: pow and round" do
    test "float pow: negative base with a fraction is NaN, 0^0 is 1" do
      a = Nx.tensor([-8.0, 0.0, 2.0, -2.0, 0.0, 4.0])
      b = Nx.tensor([0.3333, 0.0, 0.5, 3.0, 2.0, -1.0])
      assert_same(&Nx.pow/2, [a, b])
    end

    test "integer pow by repeated squaring" do
      for type <- @ints do
        a = Nx.tensor([0, 1, 2, 3, 5, 7], type: type)
        b = Nx.tensor([0, 5, 7, 4, 3, 2], type: type)
        assert_same(&Nx.pow/2, [a, b])
      end
    end

    test "round is half away from zero" do
      x = Nx.tensor([2.5, -2.5, 0.5, -0.5, 1.5, -1.5, 0.4999, 3.0])
      assert_same(&Nx.round/1, [x])
      assert_list(run(&Nx.round/1, [x]), [3.0, -3.0, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0])
    end

    test "ceil, floor and round leave integers alone" do
      x = Nx.tensor([-3, 0, 4], type: :s16)
      for op <- ~w(ceil floor round)a, do: assert_same(fn x -> apply(Nx, op, [x]) end, [x])
    end
  end

  describe "rule: special functions" do
    test "sigmoid stays finite for large negative input" do
      x = Nx.tensor([-100.0, -1000.0, 100.0, 0.0, -20.0, 20.0])
      assert_same(&Nx.sigmoid/1, [x])
      assert Enum.all?(Nx.to_flat_list(run(&Nx.sigmoid/1, [x])), &is_float/1)
    end

    test "erf_inv tails" do
      x = Nx.tensor([0.999999, -0.999999, 0.9999, 0.5, -0.5, 0.0, 1.0e-8, 0.99999994])
      assert_same(&Nx.erf_inv/1, [x])
    end

    test "erf_inv at +-1 is infinite and outside (-1, 1) is NaN" do
      x = Nx.tensor([1.0, -1.0, 1.5, -2.0, :nan])
      assert_list(run(&Nx.erf_inv/1, [x]), [:infinity, :neg_infinity, :nan, :nan, :nan])
    end

    test "erf, erfc and cbrt on negatives and large values" do
      x = Nx.tensor([-8.0, -0.5, 0.0, 0.5, 3.0, 6.0, 10.0])
      for op <- ~w(erf erfc cbrt)a, do: assert_same(fn x -> apply(Nx, op, [x]) end, [x])
    end

    test "count_leading_zeros and population_count at every width" do
      for type <- @ints do
        max = Nx.to_number(Nx.Constants.max_finite(type))
        min = Nx.to_number(Nx.Constants.min_finite(type))
        x = Nx.tensor([0, 1, 2, max, min, 16, 3], type: type)
        assert_same(&Nx.count_leading_zeros/1, [x])
        assert_same(&Nx.population_count/1, [x])
      end

      assert_list(run(&Nx.count_leading_zeros/1, [Nx.tensor([0, 1, -1, 16], type: :s8)]), [
        8,
        7,
        0,
        3
      ])
    end
  end

  describe "a type the device cannot run" do
    test "an f16 program raises NxCL.CompilerError naming the extension" do
      unless capability?(:fp16) do
        jitted = Nx.Defn.jit(&Nx.negate/1, compiler: NxCL.Compiler)

        assert_raise NxCL.CompilerError, ~r/missing extension cl_khr_fp16/, fn ->
          jitted.(Nx.tensor([1.0], type: :f16))
        end
      end
    end

    test "an f64 program raises NxCL.CompilerError naming the extension" do
      unless capability?(:fp64) do
        jitted = Nx.Defn.jit(&Nx.negate/1, compiler: NxCL.Compiler)

        assert_raise NxCL.CompilerError, ~r/missing extension cl_khr_fp64/, fn ->
          jitted.(Nx.tensor([1.0], type: :f64))
        end
      end
    end
  end
end
