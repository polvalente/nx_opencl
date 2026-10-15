defmodule NxCL.ComplexParityTest do
  @moduledoc """
  Differential rows for `c64` and `c128`: every elementwise op Nx accepts on a
  complex tensor, through `NxCL.Parity` against `Nx.BinaryBackend`. Rule rows
  state their expected values where `BinaryBackend` is wrong or raises: the sign
  of zero on a branch cut, `sign`, and NaN or infinity parts.

  Ops Nx 1.0 accepts on complex input (probed on `c64`; output type in
  brackets): `abs` [real], `acos acosh asin asinh atan atanh cbrt cos cosh exp expm1
  log log1p negate rsqrt sigmoid sign sin sinh sqrt tan tanh add subtract multiply
  divide pow conjugate` [complex], `real imag` [real], `equal not_equal logical_and
  logical_or logical_xor logical_not is_nan is_infinity` [`u8`], `select`, and
  `as_type` in both directions. A complex value is true when either part is not
  zero; `is_nan` and `is_infinity` look at both parts.

  Ops Nx raises on for complex input, which `NxCL.Lower` also rejects:
  `atan2 bitwise_and bitwise_not bitwise_or bitwise_xor bitcast ceil clip
  count_leading_zeros erf erf_inv erfc floor greater greater_equal left_shift less
  less_equal max min population_count quotient remainder right_shift round`. EMLX
  rejects nothing in its lowering; it relies on Nx to raise while tracing, as here.

  `c128` rows carry `requires: :fp64` and skip on a device without
  `cl_khr_fp64` (the Apple GPU). The device runs `c64` in `float`, so `c64`
  results differ from the `f64` reference by single-precision rounding; the
  default `f32` tolerance covers that, and the rows that need more say why.
  """
  use ExUnit.Case, async: false

  import NxCL.Parity

  alias NxCL.Parity.Gen

  @moduletag :opencl

  @types [{:c, 64}, {:c, 128}]

  # Explicit-expected points for the branch cuts, from C99 (Python `cmath`).
  # `BinaryBackend` puts `acos acosh asin asinh atan atanh` on the wrong side of
  # a cut at some of these points, and drops the sign of zero for `sqrt`, `tan`
  # and `tanh`. Each entry is `{{re, im}, {expected_re, expected_im}}`.
  @cuts [
    log: [
      {{-1.0, 0.0}, {0.0, 3.141592653589793}},
      {{-1.0, -0.0}, {0.0, -3.141592653589793}},
      {{-2.0, 0.0}, {0.6931471805599453, 3.141592653589793}},
      {{-2.0, -0.0}, {0.6931471805599453, -3.141592653589793}},
      {{0.0, 0.0}, {:neg_infinity, 0.0}},
      {{-0.0, 0.0}, {:neg_infinity, 3.141592653589793}},
      {{0.0, -0.0}, {:neg_infinity, -0.0}},
      {{-0.0, -0.0}, {:neg_infinity, -3.141592653589793}},
      {{1.0, 0.0}, {0.0, 0.0}},
      {{1.0e30, 1.0e30}, {69.42412638010134, 0.7853981633974483}},
      {{1.0e-30, 1.0e-30}, {-68.7309791995414, 0.7853981633974483}}
    ],
    sqrt: [
      {{-4.0, 0.0}, {0.0, 2.0}},
      {{-4.0, -0.0}, {0.0, -2.0}},
      {{4.0, 0.0}, {2.0, 0.0}},
      {{4.0, -0.0}, {2.0, -0.0}},
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{-0.0, -0.0}, {0.0, -0.0}},
      {{1.0e30, 1.0e30}, {1_098_684_113_467_810.0, 455_089_860_562_227.3}},
      {{1.0e-30, 1.0e-30}, {1.09868411346781e-15, 4.550898605622274e-16}},
      {{-1.0e30, 1.0e-30}, {5.000000000000001e-46, 1_000_000_000_000_000.0}},
      {{-1.0e30, -1.0e-30}, {5.000000000000001e-46, -1_000_000_000_000_000.0}}
    ],
    acosh: [
      {{-2.0, 0.0}, {1.3169578969248166, 3.141592653589793}},
      {{-2.0, -0.0}, {1.3169578969248166, -3.141592653589793}},
      {{0.5, 0.0}, {0.0, 1.0471975511965979}},
      {{0.5, -0.0}, {0.0, -1.0471975511965979}},
      {{2.0, 0.0}, {1.3169578969248166, 0.0}},
      {{2.0, -0.0}, {1.3169578969248166, -0.0}},
      {{0.0, 0.0}, {0.0, 1.5707963267948966}},
      {{0.0, -0.0}, {0.0, -1.5707963267948966}},
      {{1.0, 0.0}, {0.0, 0.0}},
      {{1.0, -0.0}, {0.0, -0.0}},
      {{-1.0, 0.0}, {0.0, 3.141592653589793}},
      {{-1.0, -0.0}, {0.0, -3.141592653589793}},
      {{-0.5, 0.0}, {0.0, 2.0943951023931953}},
      {{-0.5, -0.0}, {0.0, -2.0943951023931953}}
    ],
    asin: [
      {{2.0, 0.0}, {1.5707963267948966, 1.3169578969248166}},
      {{2.0, -0.0}, {1.5707963267948966, -1.3169578969248166}},
      {{-2.0, 0.0}, {-1.5707963267948966, 1.3169578969248166}},
      {{-2.0, -0.0}, {-1.5707963267948966, -1.3169578969248166}},
      {{0.5, 0.0}, {0.5235987755982989, 0.0}},
      {{0.5, -0.0}, {0.5235987755982989, -0.0}},
      {{1.0, 0.0}, {1.5707963267948966, 0.0}},
      {{1.0, -0.0}, {1.5707963267948966, -0.0}},
      {{-1.0, 0.0}, {-1.5707963267948966, 0.0}},
      {{-1.0, -0.0}, {-1.5707963267948966, -0.0}},
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{-0.0, -0.0}, {-0.0, -0.0}}
    ],
    acos: [
      {{2.0, 0.0}, {0.0, -1.3169578969248166}},
      {{2.0, -0.0}, {0.0, 1.3169578969248166}},
      {{-2.0, 0.0}, {3.141592653589793, -1.3169578969248166}},
      {{-2.0, -0.0}, {3.141592653589793, 1.3169578969248166}},
      {{0.5, 0.0}, {1.0471975511965979, -0.0}},
      {{0.5, -0.0}, {1.0471975511965979, 0.0}},
      {{1.0, 0.0}, {0.0, -0.0}},
      {{1.0, -0.0}, {0.0, 0.0}},
      {{-1.0, 0.0}, {3.141592653589793, -0.0}},
      {{-1.0, -0.0}, {3.141592653589793, 0.0}},
      {{0.0, 0.0}, {1.5707963267948966, -0.0}},
      {{0.0, -0.0}, {1.5707963267948966, 0.0}}
    ],
    atan: [
      {{0.0, 2.0}, {1.5707963267948966, 0.5493061443340549}},
      {{-0.0, 2.0}, {-1.5707963267948966, 0.5493061443340549}},
      {{0.0, -2.0}, {1.5707963267948966, -0.5493061443340549}},
      {{-0.0, -2.0}, {-1.5707963267948966, -0.5493061443340549}},
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{0.5, 0.0}, {0.4636476090008061, 0.0}},
      {{0.5, -0.0}, {0.4636476090008061, -0.0}}
    ],
    asinh: [
      {{0.0, 2.0}, {1.3169578969248166, 1.5707963267948966}},
      {{-0.0, 2.0}, {-1.3169578969248166, 1.5707963267948966}},
      {{0.0, -2.0}, {1.3169578969248166, -1.5707963267948966}},
      {{-0.0, -2.0}, {-1.3169578969248166, -1.5707963267948966}},
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{0.5, 0.0}, {0.48121182505960347, 0.0}},
      {{0.5, -0.0}, {0.48121182505960347, -0.0}}
    ],
    atanh: [
      {{2.0, 0.0}, {0.5493061443340549, 1.5707963267948966}},
      {{2.0, -0.0}, {0.5493061443340549, -1.5707963267948966}},
      {{-2.0, 0.0}, {-0.5493061443340549, 1.5707963267948966}},
      {{-2.0, -0.0}, {-0.5493061443340549, -1.5707963267948966}},
      {{0.5, 0.0}, {0.5493061443340549, 0.0}},
      {{0.5, -0.0}, {0.5493061443340549, -0.0}},
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{0.0, 2.0}, {0.0, 1.1071487177940904}},
      {{0.0, -2.0}, {0.0, -1.1071487177940904}}
    ],
    tan: [
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{1.0, 30.0}, {1.5924545408982664e-26, 1.0}},
      {{1.0, -30.0}, {1.5924545408982664e-26, -1.0}},
      {{0.5, 0.5}, {0.40389645531602586, 0.5640831412674986}}
    ],
    tanh: [
      {{0.0, 0.0}, {0.0, 0.0}},
      {{-0.0, 0.0}, {-0.0, 0.0}},
      {{0.0, -0.0}, {0.0, -0.0}},
      {{30.0, 1.0}, {1.0, 1.5924545408982664e-26}},
      {{-30.0, 1.0}, {-1.0, 1.5924545408982664e-26}},
      {{0.5, 0.5}, {0.5640831412674986, 0.40389645531602586}}
    ]
  ]

  # Tolerances for results the reference computes in `f64` and we compute in `f32`.
  # `pow` raises a magnitude to a power and multiplies by `exp`, so the single
  # precision error grows with the result.
  @tolerances %{{:pow, {:c, 64}} => [rtol: 1.0e-4, atol: 1.0e-4]}

  defp tol(op, type), do: Map.get(@tolerances, {op, type}, [])

  # -- Data ----------------------------------------------------------------

  defp part({:c, 64}), do: {:f, 32}
  defp part({:c, 128}), do: {:f, 64}

  # `Nx.complex/2` loses the sign of a zero part, so build from `Complex` values.
  defp complex(pairs, type) do
    Nx.tensor(for({r, i} <- pairs, do: Complex.new(r, i)), type: type)
  end

  # Finite values only: zeros with a positive imaginary part, plus random points.
  # Points on a branch cut (a zero imaginary part of either sign) or at a pole are
  # left to the rule rows, where the reference is wrong or undefined.
  defp data(type) do
    zeros = complex([{0.0, 0.0}, {-0.0, 0.0}], type)
    Nx.concatenate([Gen.random({65}, type), zeros])
  end

  # `data` with the zeros replaced by 1, for divisors.
  defp nonzero(type) do
    z = data(type)
    Nx.select(Nx.equal(z, Nx.tensor(0, type: type)), Nx.tensor(1, type: type), z)
  end

  # Values with NaN and infinity in either part, for the rows that state their result.
  defp specials(type) do
    nan = :nan
    inf = :infinity
    ninf = :neg_infinity

    complex(
      [{0.0, 0.0}, {nan, 1.0}, {inf, 1.0}, {1.0, inf}, {0.0, 2.0}, {-0.0, 0.0}, {ninf, 0.0}] ++
        [{1.0, ninf}, {inf, inf}, {nan, nan}, {2.0, nan}],
      type
    )
  end

  # `BinaryBackend` returns the negative of the principal value of `acos` and
  # `acosh` where the real part would come out below zero (about half the plane).
  # C99 and XLA keep the real part at or above zero. Flipping the reference's
  # sign there gives the principal value; our kernels already return it.
  defp principal(op, x) when op in [:acos, :acosh] do
    r = apply(Nx, op, [x])
    Nx.select(Nx.less(Nx.real(r), 0), Nx.negate(r), r)
  end

  defp principal(op, x), do: apply(Nx, op, [x])

  # `rsqrt` of zero is NaN in both, but a divisor-free input tests the values.
  defp unary_data(:rsqrt, type), do: nonzero(type)
  defp unary_data(_op, type), do: data(type)

  defp right_operand(:divide, type), do: Nx.reverse(nonzero(type))

  defp right_operand(:pow, type),
    do: Nx.multiply(Nx.reverse(data(type)), Nx.tensor(0.25, type: part(type)))

  defp right_operand(_op, type), do: Nx.reverse(data(type))

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  # The eager path: arguments on `NxCL.Backend`, each op a one-op program.
  defp eager(fun, args) do
    Nx.with_default_backend(NxCL.Backend, fn ->
      apply(
        Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator),
        Enum.map(args, &Nx.backend_copy(&1, NxCL.Backend))
      )
    end)
  end

  defp bit(v) when is_float(v) do
    <<s::1, _::63>> = <<v::float-64>>
    s
  end

  # A part is right when it has the expected value within `tol`, and the same
  # sign when both are zero.
  defp part_ok?(e, a, _tol) when is_atom(e) or is_atom(a), do: e == a
  defp part_ok?(e, a, _tol) when e == 0 and a == 0, do: bit(e) == bit(a)
  defp part_ok?(e, a, tol), do: abs(e - a) <= tol + tol * abs(e)

  defp assert_values(op, type, rows, tol, runner \\ &run/2) do
    z = complex(for({{x, y}, _} <- rows, do: {x, y}), type)
    out = runner.(&apply(Nx, op, [&1]), [z])
    assert out.type == type

    for {{{x, y}, {er, ei}}, r, i} <-
          Enum.zip([rows, Nx.to_flat_list(Nx.real(out)), Nx.to_flat_list(Nx.imag(out))]) do
      assert part_ok?(er, r, tol) and part_ok?(ei, i, tol),
             "#{op}(#{inspect(x)}, #{inspect(y)}): expected #{inspect({er, ei})}, got #{inspect({r, i})}"
    end
  end

  # -- Unary: complex in, complex out -------------------------------------------

  @unary ~w(negate sign exp expm1 log log1p sqrt rsqrt cbrt sin cos tan sinh cosh tanh asin acos
            atan asinh acosh atanh sigmoid conjugate)a

  describe "unary" do
    for op <- @unary, type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      # `sign` of a complex value is z / |z|; `BinaryBackend` returns 1 for every
      # value, so the rule rows below state `sign`.
      if op != :sign do
        test "#{op} #{Nx.Type.to_string(type)}" do
          op = unquote(op)
          type = unquote(Macro.escape(type))
          assert_parity(op, fn x -> principal(op, x) end, [unary_data(op, type)], tol(op, type))
        end
      end
    end

    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "abs #{Nx.Type.to_string(type)} returns the real type" do
        type = unquote(Macro.escape(type))
        out = run(&Nx.abs/1, [data(type)])
        assert out.type == part(type)
        assert_parity(:abs, &Nx.abs/1, [data(type)])
      end

      for op <- ~w(real imag)a do
        if type == {:c, 128}, do: @tag(requires: :fp64)

        test "#{op} #{Nx.Type.to_string(type)}" do
          type = unquote(Macro.escape(type))
          assert_parity(unquote(op), &apply(Nx, unquote(op), [&1]), [data(type)])
        end
      end

      for op <- ~w(is_nan is_infinity)a do
        if type == {:c, 128}, do: @tag(requires: :fp64)

        test "#{op} #{Nx.Type.to_string(type)}" do
          type = unquote(Macro.escape(type))
          assert_parity(unquote(op), &apply(Nx, unquote(op), [&1]), [data(type)])
        end
      end

      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "logical_not #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        assert_parity(:block_logical_not, &Nx.logical_not/1, [data(type)])
      end
    end

    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "phase #{Nx.Type.to_string(type)} is the angle of each value" do
        type = unquote(Macro.escape(type))
        assert_parity(:block_phase, &Nx.phase/1, [data(type)])
      end
    end

    for type <- [{:f, 32}, {:s, 32}, {:u, 8}] do
      test "phase of #{Nx.Type.to_string(type)} is 0 or pi" do
        type = unquote(Macro.escape(type))
        x = Nx.concatenate([Gen.random({65}, type), Nx.tensor([0, 1], type: type)])
        assert_parity(:block_phase, &Nx.phase/1, [x])
      end
    end

    # `real` and `conjugate` also take real tensors. `conjugate` of a
    # real value has imaginary part -0.0.
    for type <- [{:f, 32}, {:s, 32}, {:u, 8}] do
      test "real and conjugate of #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Nx.concatenate([Gen.random({65}, type), Nx.tensor([0, 1], type: type)])
        # `Nx.imag` of a real tensor traces to a zero broadcast, not an `imag` op
        # (stage 06 lowers broadcasts).
        assert_parity(:real, &Nx.real/1, [x])
        assert_parity(:conjugate, &Nx.conjugate/1, [x], signed_zero: true)
      end
    end
  end

  # -- Binary --------------------------------------------------------------------

  describe "binary" do
    for op <- ~w(add subtract multiply divide pow)a, type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        a = data(type)

        b = right_operand(op, type)

        assert_parity(op, fn x, y -> apply(Nx, op, [x, y]) end, [a, b], tol(op, type))
      end
    end

    for op <- ~w(equal not_equal logical_and logical_or logical_xor)a, type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "#{op} #{Nx.Type.to_string(type)}" do
        op = unquote(op)
        type = unquote(Macro.escape(type))
        a = data(type)
        # Half the pairs are equal, so `equal` sees both outcomes.
        b = Nx.select(Nx.less(Nx.iota(Nx.shape(a)), 40), a, Nx.reverse(a))
        assert_parity(op, fn x, y -> apply(Nx, op, [x, y]) end, [a, b])
      end
    end
  end

  describe "mixed operands" do
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "complex with real tensors and scalars #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        z = data(type)
        r = Gen.random(Nx.shape(z), {:f, 32})
        i = Nx.iota(Nx.shape(z), type: :s32)

        for op <- ~w(add subtract multiply)a do
          assert_parity(op, fn a, b -> apply(Nx, op, [a, b]) end, [z, r])
          assert_parity(op, fn a, b -> apply(Nx, op, [b, a]) end, [z, i])
        end

        assert_parity(:divide, fn a, b -> Nx.divide(a, b) end, [z, Nx.add(r, 5.0)])
      end

      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "complex constants and scalar arguments #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        z = data(type)
        scalar = complex([{1.5, -2.0}], type) |> Nx.squeeze()

        assert_parity(:add, fn x -> Nx.add(x, Complex.new(1.0, 2.0)) end, [z])
        assert_parity(:multiply, fn x -> Nx.multiply(x, Complex.new(0.0, 1.0)) end, [z])
        assert_parity(:subtract, fn x -> Nx.subtract(Complex.new(-0.5, 3.0), x) end, [z])
        assert_parity(:multiply, fn x -> Nx.multiply(x, 2.5) end, [z])

        assert_parity(:add, fn x, s -> Nx.add(x, s) end, [
          Nx.reduce_max(Nx.abs(z)),
          Nx.abs(scalar)
        ])

        assert_parity(:multiply, fn x, y -> Nx.multiply(x, y) end, [scalar, scalar])
      end
    end
  end

  # -- select and casts ---------------------------------------------------------

  describe "select and as_type" do
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "select #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        a = data(type)
        b = Nx.reverse(a)
        pred = Nx.less(Nx.iota(Nx.shape(a)), 30)
        assert_parity(:select, &Nx.select/3, [pred, a, b])
        # A complex predicate is true when either part is not zero.
        assert_parity(:select, &Nx.select/3, [a, a, b])
      end
    end

    for from <- [{:f, 32}, {:s, 32}, {:u, 8}, {:s, 64}, {:bf, 16}] do
      test "#{Nx.Type.to_string(from)} to c64 gives a zero imaginary part" do
        from = unquote(Macro.escape(from))
        x = Nx.concatenate([Gen.random({65}, from), Nx.tensor([0, 1], type: from)])
        assert_parity(:as_type, &Nx.as_type(&1, :c64), [x], signed_zero: true)
      end
    end

    # The reference drops the imaginary part; so do we.
    for to <- [{:f, 32}, {:s, 32}, {:u, 8}, {:s, 64}, {:bf, 16}] do
      test "c64 to #{Nx.Type.to_string(to)} drops the imaginary part" do
        to = unquote(Macro.escape(to))
        z = data({:c, 64})
        assert_parity(:as_type, &Nx.as_type(&1, to), [z])
      end
    end

    @tag requires: :fp64
    test "c64 to c128 and back" do
      z = data({:c, 64})
      assert_parity(:as_type, &Nx.as_type(&1, :c128), [z])
      assert_parity(:as_type, &Nx.as_type(&1, :c64), [data({:c, 128})])
      back = run(fn x -> x |> Nx.as_type(:c128) |> Nx.as_type(:c64) end, [z])
      assert back.type == {:c, 64}
      assert Nx.to_binary(back) == Nx.to_binary(z)
    end

    @tag requires: :fp64
    test "c128 to real and f64 to c128" do
      z = data({:c, 128})
      assert_parity(:as_type, &Nx.as_type(&1, :f64), [z])
      assert_parity(:as_type, &Nx.as_type(&1, :f32), [z])
      assert_parity(:as_type, &Nx.as_type(&1, :c128), [Gen.random({65}, {:f, 64})])
    end
  end

  # -- Rules: expected values stated here ---------------------------------------------

  describe "rule: upload and readback" do
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "a #{Nx.Type.to_string(type)} tensor round-trips bit for bit" do
        type = unquote(Macro.escape(type))
        z = Nx.concatenate([data(type), specials(type)])
        out = run(fn x -> x end, [z])
        assert out.type == type
        assert Nx.to_binary(out) == Nx.to_binary(z)
      end
    end
  end

  describe "rule: overflow" do
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "cdiv and cabs near the overflow limit stay finite #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        big = if type == {:c, 64}, do: 1.0e30, else: 1.0e300
        z = complex([{big, big}, {big, -big}, {-big, big}], type)

        out = run(&Nx.divide(&1, &1), [z])
        assert Nx.to_flat_list(Nx.real(out)) == [1.0, 1.0, 1.0]
        assert Nx.to_flat_list(Nx.imag(out)) == [0.0, 0.0, 0.0]

        abs = run(&Nx.abs/1, [z]) |> Nx.to_flat_list()
        want = big * :math.sqrt(2.0)
        assert Enum.all?(abs, &(is_float(&1) and abs(&1 - want) <= 1.0e-6 * want))

        # Erlang floats raise on 1.0e300 * 1.0e300, so `Complex.divide` cannot
        # serve as the reference for `c128` here; the values above are exact.
        if type == {:c, 64} do
          assert_parity(:divide, &Nx.divide/2, [z, Nx.reverse(z)])
          assert_parity(:abs, &Nx.abs/1, [z])
        end
      end
    end
  end

  describe "rule: branch cuts" do
    for {op, rows} <- @cuts, type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "#{op} on the cuts keeps the sign of zero #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        tol = if type == {:c, 64}, do: 1.0e-5, else: 1.0e-12
        assert_values(unquote(op), type, unquote(Macro.escape(rows)), tol)
      end
    end
  end

  describe "rule: NaN and infinity" do
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "abs, sign, tests and comparisons on special values #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        z = specials(type)

        # An infinite part wins over NaN: hypot(nan, inf) is inf. `BinaryBackend`
        # gives inf for (nan, 1), which is wrong; C99 says NaN.
        abs = run(&Nx.abs/1, [z]) |> Nx.to_flat_list()

        assert abs ==
                 [0.0, :nan, :infinity, :infinity, 2.0, 0.0, :infinity] ++
                   [:infinity, :infinity, :nan, :nan]

        assert run(&Nx.is_nan/1, [z]) |> Nx.to_flat_list() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]

        assert run(&Nx.is_infinity/1, [z]) |> Nx.to_flat_list() == [
                 0,
                 0,
                 1,
                 1,
                 0,
                 0,
                 1,
                 1,
                 1,
                 0,
                 0
               ]

        assert run(&Nx.equal(&1, &1), [z]) |> Nx.to_flat_list() == [
                 1,
                 0,
                 1,
                 1,
                 1,
                 1,
                 1,
                 1,
                 1,
                 0,
                 0
               ]

        assert run(&Nx.not_equal(&1, &1), [z]) |> Nx.to_flat_list() == [
                 0,
                 1,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 1,
                 1
               ]

        assert run(&Nx.logical_not/1, [z]) |> Nx.to_flat_list() == [
                 1,
                 0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0,
                 0,
                 0
               ]
      end

      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "log, sqrt, exp, divide on special values #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        inf = :infinity
        nan = :nan

        # C99 Annex G. `BinaryBackend` returns inf + NaN i for log(nan + 1i).
        assert_values(
          :log,
          type,
          [
            {{nan, 1.0}, {nan, nan}},
            {{inf, 1.0}, {inf, 0.0}},
            {{1.0, inf}, {inf, 1.5707963267948966}},
            {{:neg_infinity, 1.0}, {inf, 3.141592653589793}}
          ],
          1.0e-5
        )

        assert_values(
          :sqrt,
          type,
          [
            {{nan, 1.0}, {nan, nan}},
            {{inf, 1.0}, {inf, 0.0}},
            {{1.0, inf}, {inf, inf}},
            {{:neg_infinity, 1.0}, {0.0, inf}},
            {{:neg_infinity, -1.0}, {0.0, :neg_infinity}}
          ],
          1.0e-5
        )

        assert_values(
          :exp,
          type,
          [
            {{inf, 1.0}, {inf, 1.682941969615793}} |> then(fn {z, _} -> {z, {inf, inf}} end),
            {{1.0, inf}, {nan, nan}},
            {{nan, 1.0}, {nan, nan}},
            {{:neg_infinity, 1.0}, {0.0, 0.0}}
          ],
          1.0e-5
        )

        # Smith's division: (inf + i) / (2 + i) = inf - inf i, and a zero divisor is NaN.
        out =
          run(&Nx.divide/2, [
            complex([{inf, 1.0}, {3.0, 4.0}], type),
            complex([{2.0, 1.0}, {0.0, 0.0}], type)
          ])

        assert Nx.to_flat_list(Nx.real(out)) == [inf, nan]
        assert Nx.to_flat_list(Nx.imag(out)) == [:neg_infinity, nan]
      end
    end

    # Reference: `Nx.sign` of a complex value in `BinaryBackend` is 1 whatever the
    # input. We use z / |z| (as XLA and MLX do), and 0 for 0.
    for type <- @types do
      if type == {:c, 128}, do: @tag(requires: :fp64)

      test "sign is z over |z| #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        # Random points against z / |z|, computed on `BinaryBackend`; the compiler and the
        # eager backend each get the same checks.
        z = nonzero(type)
        want = Nx.divide(z, Nx.as_type(Nx.abs(z), type))
        tol = if type == {:c, 64}, do: 1.0e-5, else: 1.0e-12

        for {runner, path} <- [{&run/2, :compiler}, {&eager/2, :backend}] do
          got = runner.(&Nx.sign/1, [z]) |> Nx.backend_copy(Nx.BinaryBackend)

          for part <- [&Nx.real/1, &Nx.imag/1] do
            assert Nx.all_close(part.(got), part.(want), rtol: tol, atol: tol) |> Nx.to_number() ==
                     1
          end

          NxCL.Parity.Recorder.record(:sign, type, path)

          assert_values(
            :sign,
            type,
            [
              {{3.0, 4.0}, {0.6, 0.8}},
              {{0.0, 2.0}, {0.0, 1.0}},
              {{-5.0, 0.0}, {-1.0, 0.0}},
              {{0.0, 0.0}, {0.0, 0.0}},
              {{1.0, -1.0}, {0.7071067811865475, -0.7071067811865475}}
            ],
            1.0e-5,
            runner
          )
        end
      end
    end
  end

  # -- Lowering -------------------------------------------------------------------------

  describe "rejected ops" do
    @unary_rejected ~w(bitwise_not ceil count_leading_zeros erf erf_inv erfc floor
                       population_count round)a
    @binary_rejected ~w(atan2 bitwise_and bitwise_or bitwise_xor greater greater_equal
                        left_shift less less_equal max min quotient remainder right_shift)a

    for op <- @unary_rejected do
      test "#{op} raises while tracing a complex operand, and in NxCL.Lower" do
        op = unquote(op)
        z = Nx.tensor([Complex.new(1.0, 2.0)], type: :c64)

        assert_raise ArgumentError, fn ->
          Nx.Defn.jit(&apply(Nx, op, [&1]), compiler: NxCL.Compiler).(z)
        end

        p = Nx.Defn.Expr.parameter(:root, {:c, 64}, {2}, 0)
        expr = apply(Nx.Defn.Expr, op, [Nx.template({2}, {:c, 64}), p])

        assert_raise ArgumentError, ~r/does not support complex tensors/, fn ->
          NxCL.Lower.lower(expr, 1)
        end
      end
    end

    for op <- @binary_rejected do
      test "#{op} raises while tracing a complex operand, and in NxCL.Lower" do
        op = unquote(op)
        z = Nx.tensor([Complex.new(1.0, 2.0)], type: :c64)

        assert_raise ArgumentError, fn ->
          Nx.Defn.jit(&apply(Nx, op, [&1, &2]), compiler: NxCL.Compiler).(z, z)
        end

        p = Nx.Defn.Expr.parameter(:root, {:c, 64}, {2}, 0)
        expr = apply(Nx.Defn.Expr, op, [Nx.template({2}, {:c, 64}), p, p])

        assert_raise ArgumentError, ~r/does not support complex tensors/, fn ->
          NxCL.Lower.lower(expr, 1)
        end
      end
    end

    test "clip and bitcast raise in NxCL.Lower" do
      p = Nx.Defn.Expr.parameter(:root, {:c, 64}, {2}, 0)
      template = Nx.template({2}, {:c, 64})

      assert_raise ArgumentError, ~r/:clip does not support complex tensors/, fn ->
        NxCL.Lower.lower(Nx.Defn.Expr.clip(template, p, p, p), 1)
      end

      assert_raise ArgumentError, ~r/:bitcast does not support complex tensors/, fn ->
        NxCL.Lower.lower(Nx.Defn.Expr.bitcast(Nx.template({2}, {:u, 64}), p), 1)
      end
    end
  end

  describe "a type the device cannot run" do
    test "a c128 program raises NxCL.CompilerError naming the extension" do
      unless capability?(:fp64) do
        jitted = Nx.Defn.jit(&Nx.negate/1, compiler: NxCL.Compiler)

        assert_raise NxCL.CompilerError, ~r/missing extension cl_khr_fp64/, fn ->
          jitted.(Nx.tensor([Complex.new(1.0, 2.0)], type: :c128))
        end
      end
    end
  end
end
