defmodule NxCL.LinalgTest.Custom do
  @moduledoc false
  defstruct []
end

defmodule NxCL.LinalgTest do
  @moduledoc """
  Differential rows for `cholesky`, `triangular_solve`, `solve`, `determinant`,
  `qr`, `eigh`, `svd`, `lu`, `all_close`, `top_k` and a user block. Rows use
  `NxCL.TestSupport.Compare` against `Nx.BinaryBackend`, except `qr`, `eigh`, `svd`
  and `lu`, whose factors may differ from the reference in sign and order; those
  rows use the invariant checks of `NxCL.TestSupport.Factors`.

  Inputs are well conditioned: a Cholesky input is `L L^H` for a lower factor `L`
  with a diagonal of about 2 and small off-diagonal entries; a triangular solve
  gets a dense matrix with a diagonal of about 2, so a solver that read the wrong
  triangle would show. Integer types use fixed small matrices. Sizes are 1, 3, 64
  and 257 for `f32`, `f64`, `c64` and `c128`; the half types and the integers stop
  at 3 (a `bf16` factorization rounds at every step, so the two sides drift apart
  at 64 by more than the type's tolerance, and 64 integer rows would overflow).

  `Nx.BinaryBackend` is wrong, or raises, in these cases (probed on Nx 1.0), so those
  rows state the answer:

    * `cholesky` on `f64` and `c128`: the reference divides by `l_jj + 1.0e-10`, an
      error of 1e-10 against a tolerance of 1e-12. The answer is the factor `L`
      the input was built from.
    * `solve` and `triangular_solve` on a singular matrix: the reference raises
      "can't solve for singular matrix". The compiler returns infinities and NaN, as
      a device does; the row checks that.

  `f64` and `c128` rows carry `requires: :fp64`; `f16` rows carry `requires: :fp16`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare
  import NxCL.TestSupport.Factors

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 1_800_000

  @types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 8},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:f, 16},
    {:bf, 16},
    {:f, 32},
    {:f, 64},
    {:c, 64},
    {:c, 128}
  ]

  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  defp integer?({kind, _}), do: kind in [:s, :u]
  defp complex?({kind, _}), do: kind == :c
  defp wide?(type), do: type in [{:f, 32}, {:f, 64}, {:c, 64}, {:c, 128}]
  defp double?(type), do: type in [{:f, 64}, {:c, 128}]

  # Sizes a type takes; integers use their own fixed matrices.
  defp sizes(type), do: if(wide?(type), do: [1, 3, 64, 257], else: [1, 3])

  # -- Inputs ---------------------------------------------------------------

  defp wide_type(type), do: if(complex?(type), do: {:c, 128}, else: {:f, 64})

  # A dense matrix with a diagonal of about 2, in the wide type of `type`.
  defp dense(n, type, seed) do
    wide = wide_type(type)
    r = Gen.random({n, n}, wide, seed: NxCL.TestSupport.Compare.seed() + seed)
    scaled = Nx.multiply(r, Nx.tensor(0.1 / :math.sqrt(n), type: {:f, 64}))
    Nx.add(scaled, Nx.multiply(Nx.eye(n), Nx.tensor(2.0, type: {:f, 64})))
  end

  # A lower factor with a real diagonal of about 2.
  defp factor(n, type, seed) do
    d = dense(n, type, seed)
    diag = Nx.real(Nx.take_diagonal(d))
    strict = Nx.tril(d, k: -1)
    Nx.add(strict, Nx.make_diagonal(diag))
  end

  # `L L^H`, the matrix whose Cholesky factor is `L`.
  defp gram(l) do
    lh = if Nx.Type.complex?(Nx.type(l)), do: Nx.conjugate(Nx.transpose(l)), else: Nx.transpose(l)
    Nx.dot(l, lh)
  end

  # The half types round at every operation, and the reference truncates where the
  # device rounds to nearest. A determinant is a sum of products that cancel, so on
  # general values the two sides differ by more than the tolerance. Whole numbers
  # keep every product and sum exact, so those rows test the lowering, not rounding.
  defp whole(x, {kind, 16}) when kind in [:f, :bf], do: Nx.round(x)
  defp whole(x, _type), do: x

  defp triangular_from_4(x, n, {kind, 16}) when n > 3 and kind in [:f, :bf], do: Nx.triu(x)
  defp triangular_from_4(x, _n, _type), do: x

  defp stack_of(count, fun), do: Nx.stack(for i <- 1..count, do: fun.(i))

  defp stacked_or_single(nil, fun), do: fun.(1)
  defp stacked_or_single(count, fun), do: stack_of(count, fun)

  # Integer matrices small enough for every integer type.
  defp int_matrix(1), do: Nx.tensor([[3]])
  defp int_matrix(3), do: Nx.tensor([[4, 1, 2], [1, 5, 1], [2, 1, 6]])

  defp small(type, n, batch \\ nil) do
    mat = fn i ->
      if integer?(type) do
        int_matrix(n) |> Nx.add(Nx.multiply(Nx.eye(n, type: :s32), i - 1)) |> Nx.as_type(type)
      else
        dense(n, type, 30 + i) |> Nx.as_type(type)
      end
    end

    if batch, do: stack_of(batch, mat), else: mat.(1)
  end

  # Every size runs unbatched and with a leading batch of 2 (3 also runs 4).
  defp batches(3), do: [nil, 2, 4]
  defp batches(_), do: [nil, 2]

  defp spd_ints(a) do
    # Symmetric and diagonally dominant: `a + a^T` plus `n` on the diagonal.
    n = elem(Nx.shape(a), tuple_size(Nx.shape(a)) - 1)
    sym = Nx.add(a, Nx.transpose(a, axes: transpose_axes(a)))
    Nx.add(sym, Nx.multiply(Nx.eye(n, type: Nx.type(a)), 3))
  end

  defp transpose_axes(a) do
    r = Nx.rank(a)
    Enum.to_list(0..(r - 3)//1) ++ [r - 1, r - 2]
  end

  defp small_or_dense(type, n, _batch) do
    if integer?(type), do: small(type, n), else: dense(n, type, 20) |> Nx.as_type(type)
  end

  defp batched(type, batch, n) do
    count = Enum.product(batch)

    stacked =
      stack_of(count, fn i ->
        small_or_dense(type, n, nil) |> Nx.add(Nx.multiply(Nx.eye(n, type: type), i - 1))
      end)

    Nx.reshape(stacked, List.to_tuple(batch ++ [n, n]))
  end

  defp rhs(type, n, left, m) do
    shape = if left, do: {n, m}, else: {m, n}
    Gen.random(shape, type, seed: NxCL.TestSupport.Compare.seed() + 4)
  end

  # -- cholesky ---------------------------------------------------------------

  describe "cholesky" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for n <- sizes(type), batch <- batches(n) do
          factors = fn i -> factor(n, type, 10 + i) end

          if integer?(type) do
            a = small(type, n, batch) |> spd_ints()
            assert_same(&Nx.LinAlg.cholesky/1, [a])
          else
            a = stacked_or_single(batch, &gram(factors.(&1))) |> Nx.as_type(type)

            if double?(type) do
              l = stacked_or_single(batch, factors) |> Nx.as_type(type)
              assert_expected(&Nx.LinAlg.cholesky/1, [a], l)
            else
              assert_same(&Nx.LinAlg.cholesky/1, [a])
            end
          end
        end
      end
    end

    for type <- [{:f, 32}, {:c, 64}] do
      test "a matrix that is not positive definite, or singular, matches on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        not_pd = Nx.tensor([[1.0, 2.0], [2.0, 1.0]]) |> Nx.as_type(type)
        singular = Nx.tensor([[1.0, 2.0], [2.0, 4.0]]) |> Nx.as_type(type)
        ill = Nx.tensor([[1.0, 1.0], [1.0, 1.0001]]) |> Nx.as_type(type)

        for a <- [not_pd, singular, ill] do
          assert_same(&Nx.LinAlg.cholesky/1, [a])
        end
      end
    end
  end

  # -- triangular_solve -----------------------------------------------------

  @all_options for left <- [true, false],
                   lower <- [true, false],
                   t <- [:none, :transpose, :conjugate],
                   do: [left_side: left, lower: lower, transform_a: t]
  @some_options [
    [left_side: true, lower: true, transform_a: :none],
    [left_side: true, lower: false, transform_a: :transpose],
    [left_side: false, lower: true, transform_a: :none],
    [left_side: false, lower: false, transform_a: :conjugate]
  ]

  describe "triangular_solve" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        # The reference takes tens of seconds a row at 257. There one left-side and one
        # right-side option run, with no vector right-hand side, and one batched row.
        for n <- sizes(type) do
          options =
            cond do
              n <= 3 -> @all_options
              n >= 257 -> [Enum.at(@some_options, 0), Enum.at(@some_options, 2)]
              true -> @some_options
            end

          a = small_or_dense(type, n, nil)

          for opts <- options do
            b = rhs(type, n, opts[:left_side], 2)
            assert_same(&Nx.LinAlg.triangular_solve(&1, &2, opts), [a, b])
          end

          # A vector right-hand side.
          if n < 257 do
            for opts <- Enum.take(options, 4) do
              b = Gen.random({n}, type, seed: NxCL.TestSupport.Compare.seed() + 5)
              assert_same(&Nx.LinAlg.triangular_solve(&1, &2, opts), [a, b])
            end
          end
        end

        # Batched at every size: a leading batch of 3, then {2, 2}; one of 2 at 257, with
        # one option.
        for n <- sizes(type),
            batch <- if(n >= 257, do: [[2]], else: [[3], [2, 2]]),
            opts <- if(n >= 257, do: Enum.take(@some_options, 1), else: @some_options) do
          a = batched(type, batch, n)
          shape = List.to_tuple(batch ++ if(opts[:left_side], do: [n, 2], else: [2, n]))
          b = Gen.random(shape, type, seed: NxCL.TestSupport.Compare.seed() + 6)
          assert_same(&Nx.LinAlg.triangular_solve(&1, &2, opts), [a, b])

          vector =
            Gen.random(List.to_tuple(batch ++ [n]), type,
              seed: NxCL.TestSupport.Compare.seed() + 7
            )

          assert_same(&Nx.LinAlg.triangular_solve(&1, &2, opts), [a, vector])
        end
      end
    end

    test "operands of different types solve in the floating type that holds both" do
      a = dense(3, {:f, 32}, 20) |> Nx.as_type(:f32)
      ai = int_matrix(3) |> Nx.as_type(:s32)
      bf = Gen.random({3, 2}, {:f, 32})
      bi = Gen.random({3, 2}, {:s, 32})
      bc = Gen.random({3, 2}, {:c, 64})

      for {x, y} <- [{a, bi}, {ai, bf}, {a, bc}, {ai, bc}] do
        assert_same(&Nx.LinAlg.triangular_solve/2, [x, y])
      end
    end

    test "a zero on the diagonal gives infinities and NaN where the reference raises" do
      a = Nx.tensor([[1.0, 0.0], [1.0, 0.0]])
      b = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/singular/, fn ->
        Nx.Defn.jit(&Nx.LinAlg.triangular_solve/2, compiler: Nx.Defn.Evaluator).(a, b)
      end

      result = Nx.Defn.jit(&Nx.LinAlg.triangular_solve/2, compiler: NxCL.Compiler).(a, b)
      assert Nx.to_flat_list(result) == [1.0, :infinity]
    end
  end

  # -- solve ----------------------------------------------------------------

  describe "solve" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for n <- sizes(type) do
          # The eager run at 257 replays the same default expression op by op for about
          # 20 s a row and covers no op the smaller rows miss, so it runs below 257 only.
          eager = [backend: n < 257]
          a = small_or_dense(type, n, nil)
          vector = Gen.random({n}, type, seed: NxCL.TestSupport.Compare.seed() + 8)
          matrix = Gen.random({n, 2}, type, seed: NxCL.TestSupport.Compare.seed() + 9)

          # The reference takes tens of seconds a row at 257: there the vector and the
          # batched matrix right-hand sides are left out.
          if n < 257, do: assert_same(&Nx.LinAlg.solve/2, [a, vector], eager)
          assert_same(&Nx.LinAlg.solve/2, [a, matrix], eager)

          # Batched: 3 matrices, or 2 at size 257.
          count = if n >= 257, do: 2, else: 3
          a = small(type, n, count)
          b = Gen.random({count, n}, type, seed: NxCL.TestSupport.Compare.seed() + 10)
          assert_same(&Nx.LinAlg.solve/2, [a, b], eager)

          if n < 257 do
            b = Gen.random({count, n, 2}, type, seed: NxCL.TestSupport.Compare.seed() + 11)
            assert_same(&Nx.LinAlg.solve/2, [a, b], eager)
          end
        end
      end
    end

    test "an ill-conditioned matrix matches" do
      a = Nx.tensor([[1.0, 1.0], [1.0, 1.0001]])
      assert_same(&Nx.LinAlg.solve/2, [a, Nx.tensor([1.0, 2.0])])
    end

    test "a singular matrix gives NaN where the reference raises" do
      a = Nx.tensor([[1.0, 2.0], [2.0, 4.0]])
      b = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/singular/, fn ->
        Nx.Defn.jit(&Nx.LinAlg.solve/2, compiler: Nx.Defn.Evaluator).(a, b)
      end

      result = Nx.Defn.jit(&Nx.LinAlg.solve/2, compiler: NxCL.Compiler).(a, b)
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == :nan))
    end
  end

  # -- qr, eigh, svd, lu, determinant, all_close, top_k, a user block -------------

  describe "qr, eigh, svd, lu, determinant" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- [{3, 3}, {4, 3}, {3, 5}, {2, 4, 4}] do
          x = Gen.random(shape, type, seed: NxCL.TestSupport.Compare.seed() + 40)
          assert_factors(:qr, &Nx.LinAlg.qr/1, x)
          assert_factors(:qr, &Nx.LinAlg.qr(&1, mode: :complete), x)
          # SVD is not defined on complex input; Nx raises.
          unless complex?(type), do: assert_factors(:svd, &Nx.LinAlg.svd/1, x)
        end

        for shape <- [{3, 3}, {2, 4, 4}] do
          x = Gen.random(shape, type, seed: NxCL.TestSupport.Compare.seed() + 41)
          assert_factors(:lu, &Nx.LinAlg.lu/1, x)

          # Make it Hermitian here: on the device the sum would round to nearest, in the
          # reference it truncates, and the two inputs to `eigh` would differ.
          xt = Nx.transpose(x, axes: transpose_axes(x))
          sym = Nx.add(x, if(complex?(type), do: Nx.conjugate(xt), else: xt))
          assert_factors(:eigh, &Nx.LinAlg.eigh/1, sym)
        end

        for n <- [2, 3, 4, 5] do
          x = Gen.random({n, n}, type, seed: NxCL.TestSupport.Compare.seed() + 42) |> whole(type)
          # From 4 on, `determinant` runs `lu`, which divides. On a half type the reference
          # truncates and the device rounds to nearest, so general values differ by more
          # than the tolerance (both are off from the exact value). An upper triangular
          # matrix has no row swap and no division, so those rows stay exact.
          x = triangular_from_4(x, n, type)
          assert_same(&Nx.LinAlg.determinant/1, [x])
        end

        batch =
          Gen.random({2, 3, 3}, type, seed: NxCL.TestSupport.Compare.seed() + 43) |> whole(type)

        assert_same(&Nx.LinAlg.determinant/1, [batch])
      end
    end

    # Larger inputs: the defaults loop with `while`, so size shows in the time, not in
    # the lowering. A 64 x 64 `svd` takes about 2 s on the device.
    for type <- [{:f, 32}, {:c, 64}] do
      test "qr, eigh, svd and lu at 32 x 32, batched by 2, on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Gen.random({2, 32, 32}, type, seed: NxCL.TestSupport.Compare.seed() + 44)
        xt = Nx.transpose(x, axes: [0, 2, 1])

        assert_factors(:qr, &Nx.LinAlg.qr/1, x)
        assert_factors(:lu, &Nx.LinAlg.lu/1, x)

        assert_factors(
          :eigh,
          &Nx.LinAlg.eigh/1,
          Nx.add(x, if(complex?(type), do: Nx.conjugate(xt), else: xt))
        )

        unless complex?(type), do: assert_factors(:svd, &Nx.LinAlg.svd/1, x)
      end
    end

    test "lu of a singular matrix matches" do
      a = Nx.tensor([[1.0, 2.0], [2.0, 4.0]])
      assert_factors(:lu, &Nx.LinAlg.lu/1, a)
    end
  end

  describe "all_close" do
    for type <- @types do
      # `all_close` of two integer tensors widens to `f64` before `abs`, so it needs `fp64`.
      @tag requires: if(elem(type, 0) in [:s, :u], do: :fp64, else: Map.get(@needs, type))
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Gen.random({4, 5}, type, seed: NxCL.TestSupport.Compare.seed() + 50)
        near = Nx.add(x, Nx.tensor(0, type: type))
        assert_same(&Nx.all_close/2, [x, near])
        assert_same(&Nx.all_close(&1, &2, atol: 1.0e-3), [x, Nx.negate(x)])
      end
    end
  end

  describe "top_k and a user block" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = Gen.random({4, 5}, type, seed: NxCL.TestSupport.Compare.seed() + 50)

        # Nx rejects `top_k` on complex input.
        unless complex?(type) do
          assert_same(&Nx.top_k(&1, k: 3), [x])
          assert_same(&Nx.top_k(&1, k: 1), [Gen.random({7}, type)])
        end

        # One exact operation: the half types round differently on the two sides.
        custom = fn x ->
          Nx.block(%NxCL.LinalgTest.Custom{}, [x], x, fn _struct, t -> Nx.add(t, t) end)
        end

        assert_same(custom, [x])
      end
    end
  end
end
