defmodule NxCL.DotParityTest do
  @moduledoc """
  Differential rows for `dot` through `NxCL.Parity` against `Nx.BinaryBackend`, on
  every dtype Nx accepts: vector with vector, matrix with vector, vector with
  matrix, matrix with matrix, batch axes, an outer product (no contracted axis),
  and three contracted axes. Sizes 1, 7, 33, 128 and 513 appear on every axis role.
  Square products stay below 513 on all three axes together, since the reference
  is slow.

  Float and complex inputs hold small integers, so every partial sum is exact and
  the order of accumulation cannot change the answer. Integer rows use the small
  range of `Gen.random/3` and a full-range row that wraps. `f64` and `c128` rows
  carry `requires: :fp64`; `f16` rows carry `requires: :fp16`.

  The `K == 0` and zero-size rows go through the native wire, since Nx 1.0 builds
  no zero-size tensors.
  """
  use ExUnit.Case, async: false

  import NxCL.Parity

  alias NxCL.Parity.Gen

  @moduletag :opencl
  @moduletag timeout: 600_000

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

  # Small integers held in `type`, so sums are exact.
  defp exact({:c, size}, shape, seed) do
    part = {:f, div(size, 2)}
    re = exact(part, shape, seed)
    im = exact(part, shape, seed + 1)
    Nx.as_type(Nx.complex(Nx.as_type(re, :f32), Nx.as_type(im, :f32)), {:c, size})
  end

  defp exact({kind, _} = type, shape, seed) do
    key = Nx.Random.key(NxCL.Parity.seed() + seed)
    lo = if kind == :u, do: 0, else: -2
    {t, _} = Nx.Random.randint(key, lo, 3, shape: shape, type: :s32)
    Nx.as_type(t, type)
  end

  # A pair of operands for `x` and `y` shapes.
  defp pair(type, xs, ys), do: [exact(type, xs, 1), exact(type, ys, 7)]

  for type <- @types do
    describe "dot on #{Nx.Type.to_string(type)}" do
      @tag requires: Map.get(@needs, type)
      test "vector and vector" do
        type = unquote(Macro.escape(type))

        for n <- [1, 7, 33, 128, 513] do
          assert_parity(:dot, &Nx.dot/2, pair(type, {n}, {n}))
        end
      end

      @tag requires: Map.get(@needs, type)
      test "matrix and vector, vector and matrix" do
        type = unquote(Macro.escape(type))

        for {m, k} <- [{1, 1}, {7, 33}, {513, 7}, {33, 513}, {128, 128}] do
          assert_parity(:dot, &Nx.dot/2, pair(type, {m, k}, {k}))
          assert_parity(:dot, &Nx.dot/2, pair(type, {k}, {k, m}))
        end
      end

      @tag requires: Map.get(@needs, type)
      test "matrix and matrix" do
        type = unquote(Macro.escape(type))

        shapes = [
          {{1, 1}, {1, 1}},
          {{7, 33}, {33, 128}},
          {{33, 513}, {513, 7}},
          {{513, 7}, {7, 33}},
          {{130, 70}, {70, 65}},
          {{128, 1}, {1, 128}},
          {{1, 33}, {33, 513}},
          {{513, 33}, {33, 1}}
        ]

        for {xs, ys} <- shapes do
          assert_parity(:dot, &Nx.dot/2, pair(type, xs, ys))
        end
      end

      @tag requires: Map.get(@needs, type)
      test "batch axes" do
        type = unquote(Macro.escape(type))

        for {xs, ys} <- [
              {{3, 7, 33}, {3, 33, 40}},
              {{2, 513, 7}, {2, 7, 33}},
              {{2, 1, 33}, {2, 33, 70}},
              {{2, 70, 33}, {2, 33, 1}}
            ] do
          assert_parity(
            :dot,
            &Nx.dot(&1, [2], [0], &2, [1], [0]),
            pair(type, xs, ys)
          )
        end

        # Vectors per batch entry, and two batch axes.
        assert_parity(:dot, &Nx.dot(&1, [1], [0], &2, [1], [0]), pair(type, {4, 33}, {4, 33}))

        assert_parity(
          :dot,
          &Nx.dot(&1, [3], [0, 1], &2, [2], [0, 1]),
          pair(type, {2, 3, 7, 9}, {2, 3, 9, 5})
        )
      end

      @tag requires: Map.get(@needs, type)
      test "an outer product has no contracted axis" do
        type = unquote(Macro.escape(type))

        assert_parity(:dot, &Nx.dot(&1, [], &2, []), pair(type, {7}, {33}))
        assert_parity(:dot, &Nx.dot(&1, [], [0], &2, [], [0]), pair(type, {4, 7}, {4, 33}))
      end

      @tag requires: Map.get(@needs, type)
      test "three contracted axes, and contracted axes that are not adjacent" do
        type = unquote(Macro.escape(type))

        assert_parity(
          :dot,
          &Nx.dot(&1, [1, 2, 3], &2, [0, 1, 2]),
          pair(type, {7, 3, 4, 5}, {3, 4, 5, 9})
        )

        assert_parity(
          :dot,
          &Nx.dot(&1, [0, 2], &2, [1, 0]),
          pair(type, {5, 7, 3}, {3, 5, 9})
        )

        assert_parity(
          :dot,
          &Nx.dot(&1, [2, 0, 1], [], &2, [0, 1, 3], []),
          pair(type, {3, 4, 5}, {5, 3, 6, 4})
        )
      end
    end
  end

  describe "dot results" do
    test "integers wrap on the width of the type" do
      for type <- [{:s, 8}, {:u, 8}, {:s, 16}, {:u, 32}, {:s, 32}, {:u, 64}, {:s, 64}] do
        key = Nx.Random.key(NxCL.Parity.seed())
        {big, _} = Nx.Random.randint(key, -100_000, 100_000, shape: {9, 33}, type: :s64)
        {w, _} = Nx.Random.randint(key, -100_000, 100_000, shape: {33, 5}, type: :s64)
        assert_parity(:dot, &Nx.dot/2, [Nx.as_type(big, type), Nx.as_type(w, type)])
      end
    end

    test "an s32 dot stays s32" do
      x = Nx.iota({4, 4}, type: :s32)
      out = Nx.Defn.jit(&Nx.dot/2, compiler: NxCL.Compiler).(x, x)
      assert out.type == {:s, 32}
    end

    test "operands of different types merge as Nx does" do
      x = Gen.random({7, 33}, {:s, 32})
      y = Gen.random({33, 5}, {:f, 32})
      assert_parity(:dot, &Nx.dot/2, [x, y], record: false)

      assert_parity(:dot, &Nx.dot/2, [Nx.as_type(x, {:s, 8}), Nx.as_type(y, {:bf, 16})],
        record: false
      )

      z = Nx.as_type(Gen.random({33, 5}, {:f, 32}), {:c, 64})
      assert_parity(:dot, &Nx.dot/2, [Nx.as_type(x, {:f, 32}), z], record: false)
    end

    test "random floats stay within the tolerance of the reference" do
      for {xs, ys} <- [{{33, 128}, {128, 33}}, {{7, 513}, {513, 7}}] do
        assert_parity(:dot, &Nx.dot/2, [Gen.random(xs, {:f, 32}), Gen.random(ys, {:f, 32})],
          rtol: 1.0e-4,
          atol: 1.0e-4
        )
      end
    end

    test "a dot inside a larger function matches the reference" do
      assert_parity(
        :dot,
        fn x, w, b -> x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid() end,
        [exact({:f, 32}, {5, 70}, 1), exact({:f, 32}, {70, 3}, 2), exact({:f, 32}, {3}, 3)],
        rtol: 1.0e-4,
        atol: 1.0e-4,
        record: false
      )
    end
  end
end
