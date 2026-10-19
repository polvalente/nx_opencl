defmodule NxCL.ReductionTest do
  @moduledoc """
  Differential rows for reductions, arg reductions, scans and windows, through
  `NxCL.TestSupport.Compare` against `Nx.BinaryBackend`: `sum product reduce_max reduce_min
  all any argmax argmin reduce`, `cumulative_sum cumulative_product
  cumulative_min cumulative_max`, and `window_sum window_product window_max
  window_min window_reduce window_scatter_max window_scatter_min`, on every
  dtype Nx accepts for the op. Nx raises for `reduce_max reduce_min argmax
  argmin`, the min/max scans and windows, and `sort` on `c64` and `c128`, so
  those combinations are not supported.

  Float inputs for sums and scans hold small integers, so every partial sum is
  exact and the order of a tree reduction cannot change the answer. `f16` and
  `bf16` hold only -1, 0 and 1, since a partial sum past 2048 (`f16`) or 256
  (`bf16`) would round. Random non-integer floats appear only on shapes with
  few elements per line, where the default tolerance covers the rounding.
  Products of floats and complex values use entries of magnitude 1.

  `f64` and `c128` rows carry `requires: :fp64`; `f16` rows carry
  `requires: :fp16`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

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

  @real_types Enum.reject(@types, &match?({:c, _}, &1))

  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  # -- Inputs ------------------------------------------------------------------

  defp low?(type), do: type in [{:f, 16}, {:bf, 16}]

  # Small integers held in `type`, so sums are exact.
  defp exact({:c, size}, shape, seed) do
    part = {:f, div(size, 2)}
    re = exact(part, shape, seed)
    im = exact(part, shape, seed + 1)
    Nx.as_type(Nx.complex(Nx.as_type(re, :f32), Nx.as_type(im, :f32)), {:c, size})
  end

  defp exact({kind, _} = type, shape, seed) do
    key = Nx.Random.key(NxCL.TestSupport.Compare.seed() + seed)
    lim = if low?(type), do: 1, else: 8
    lo = if kind == :u, do: 0, else: -lim
    {t, _} = Nx.Random.randint(key, lo, lim + 1, shape: shape, type: :s32)
    Nx.as_type(t, type)
  end

  defp exact(type, shape), do: exact(type, shape, 0)

  # Entries of magnitude 1, so a long product neither overflows nor vanishes.
  # Integers keep their random values: their products wrap the same way.
  defp units({:c, _} = type, shape) do
    key = Nx.Random.key(NxCL.TestSupport.Compare.seed())
    {k, _} = Nx.Random.randint(key, 0, 4, shape: shape, type: :s32)
    re = Nx.take(Nx.tensor([1.0, 0.0, -1.0, 0.0]), k)
    im = Nx.take(Nx.tensor([0.0, 1.0, 0.0, -1.0]), k)
    Nx.as_type(Nx.complex(re, im), type)
  end

  defp units({kind, _} = type, shape) when kind in [:f, :bf] do
    Nx.select(Nx.greater_equal(exact({:s, 32}, shape), 0), 1.0, -1.0) |> Nx.as_type(type)
  end

  defp units(type, shape), do: Gen.random(shape, type)

  # Random factors for a product. A complex factor has a random phase and a modulus
  # within 10% of 1, so a product of 15 or 24 of them stays near 1 (between about 0.2
  # and 10): far above the tolerance, and its rounding error stays near 1e-6, under it.
  defp random_factors({:c, size} = type, shape) do
    part = {:f, div(size, 2)}
    angle = Gen.random(shape, part)
    modulus = Nx.add(1.0, Nx.multiply(Gen.random(shape, part), 0.025))
    re = Nx.multiply(modulus, Nx.cos(angle))
    im = Nx.multiply(modulus, Nx.sin(angle))
    Nx.as_type(Nx.complex(re, im), type)
  end

  # A real factor has a random sign and a magnitude in [1, 1.5], so a product of 3 to 24
  # of them is at least 1, well over the tolerance of even the half types. Integers keep
  # their random values: their products wrap the same way on both sides.
  defp random_factors({kind, _} = type, shape) when kind in [:f, :bf] do
    x = Gen.random(shape, type)
    Nx.multiply(Nx.sign(x), Nx.add(1.0, Nx.divide(Nx.abs(x), 8.0))) |> Nx.as_type(type)
  end

  defp random_factors(type, shape), do: Gen.random(shape, type)

  # A product row compares a result with the reference, so it means nothing when the
  # reference is within reach of the tolerance: a wrong answer of 0 would pass. The
  # largest expected magnitude must be at least 100 times the absolute tolerance.
  # Integers compare exactly and need no margin.
  defp assert_product(fun, args, opts \\ []) do
    expected = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)

    unless Nx.Type.integer?(expected.type) do
      tol = Keyword.get(opts, :atol, NxCL.TestSupport.Compare.default_tolerance(expected.type))
      largest = expected |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert largest >= 100 * tol,
             "the expected product is at most #{largest}, under 100 times the tolerance #{tol}: " <>
               "the row cannot tell a wrong result from a right one"
    end

    assert_same(fun, args, opts)
  end

  # Values with ties and, for floats, the special values, as one line of `n`.
  defp with_edges(type, n) do
    edges = Gen.edge_values(type)
    Nx.concatenate([edges, Gen.random({max(n - Nx.size(edges), 1)}, type)])
  end

  # Shapes per dtype: `bf16` keeps 4097 elements out, since its sums stop being exact at 256.
  defp shapes(type), do: fit(type, Gen.shapes())

  defp fit(type, shapes), do: if(type == {:bf, 16}, do: shapes -- [{4097}], else: shapes)

  defp axes_for({}), do: [nil]
  defp axes_for({_}), do: [nil, [0]]
  defp axes_for({_, _}), do: [nil, [0], [1], [0, 1]]
  defp axes_for({_, _, _}), do: [nil, [0], [1], [2], [0, 2], [1, 2], [0, 1, 2]]
  defp axes_for({_, _, _, _, _}), do: [nil, [1, 3], [0, 4], [2], [0, 2, 4]]

  defp axis_list(shape), do: Enum.to_list(0..(tuple_size(shape) - 1)//1)

  # -- sum, product, all, any --------------------------------------------------

  describe "sum, product, all, any" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "sum on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- shapes(type), axes <- axes_for(shape), keep <- [false, true] do
          assert_same(&Nx.sum(&1, axes: axes, keep_axes: keep), [exact(type, shape)])
        end

        # Few elements per line: random non-integer values stay inside the tolerance.
        for shape <- [{3, 5}, {2, 3, 4}, {63}, {65}], axes <- [nil, [0]] do
          assert_same(&Nx.sum(&1, axes: axes), [Gen.random(shape, type)],
            atol: if(low?(type), do: 0.5, else: 1.0e-4)
          )
        end

        # Few outputs and many elements (sliced), many outputs (one group each).
        assert_same(&Nx.sum(&1, axes: [0]), [exact(type, {4097, 3})])
        assert_same(&Nx.sum(&1, axes: [1]), [exact(type, {3, 4097})])
        assert_same(&Nx.sum(&1, axes: [1]), [exact(type, {70, 100})])
        assert_same(&Nx.sum(&1, axes: [0]), [exact(type, {100, 70})])
      end

      @tag requires: Map.get(@needs, type)
      test "product on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- shapes(type), axes <- axes_for(shape), keep <- [false, true] do
          assert_product(&Nx.product(&1, axes: axes, keep_axes: keep), [units(type, shape)])
        end

        # Random values. Only the half types need a loose tolerance: they round at every
        # multiply. The other types use the harness default.
        for shape <- [{3, 5}, {2, 3, 4}], axes <- [nil, [1]] do
          assert_product(
            &Nx.product(&1, axes: axes),
            [random_factors(type, shape)],
            if(low?(type), do: [atol: 1.0e-2, rtol: 1.0e-2], else: [])
          )
        end
      end

      @tag requires: Map.get(@needs, type)
      test "all and any on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- Gen.shapes(), axes <- axes_for(shape), keep <- [false, true] do
          assert_same(&Nx.all(&1, axes: axes, keep_axes: keep), [Gen.random(shape, type)])
          assert_same(&Nx.any(&1, axes: axes, keep_axes: keep), [Gen.random(shape, type)])
        end
      end
    end

    for type <- @real_types do
      @tag requires: Map.get(@needs, type)
      test "all and any on lines of one value, and edge values, on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        ones = Nx.broadcast(Nx.tensor(1, type: type), {3, 70})
        zeros = Nx.broadcast(Nx.tensor(0, type: type), {3, 70})

        for t <- [ones, zeros, Nx.concatenate([ones, zeros], axis: 0)] do
          assert_same(&Nx.all(&1, axes: [1]), [t])
          assert_same(&Nx.any(&1, axes: [1]), [t])
          assert_same(&Nx.all/1, [t])
          assert_same(&Nx.any/1, [t])
        end

        assert_same(&Nx.all/1, [with_edges(type, 70)])
        assert_same(&Nx.any/1, [with_edges(type, 70)])
      end
    end

    # `Nx.BinaryBackend` reports `all` and `any` as 1 for every complex value, zero
    # included. A complex value is true when either part is not zero, so the answers
    # are stated here.
    for type <- [{:c, 64}, {:c, 128}] do
      @tag requires: Map.get(@needs, type)
      test "all and any on #{Nx.Type.to_string(type)} with zero values, stated" do
        type = unquote(Macro.escape(type))

        re = Nx.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 2, 3]], type: :f32)
        im = Nx.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]], type: :f32)
        t = Nx.as_type(Nx.complex(re, im), type)

        all = Nx.Defn.jit(&Nx.all(&1, axes: [1]), compiler: NxCL.Compiler).(t)
        any = Nx.Defn.jit(&Nx.any(&1, axes: [1]), compiler: NxCL.Compiler).(t)

        assert Nx.to_flat_list(all) == [0, 0, 0, 1]
        assert Nx.to_flat_list(any) == [0, 1, 1, 1]
        assert all.type == {:u, 8}
        assert Nx.to_number(Nx.Defn.jit(&Nx.all/1, compiler: NxCL.Compiler).(t)) == 0
        assert Nx.to_number(Nx.Defn.jit(&Nx.any/1, compiler: NxCL.Compiler).(t)) == 1
      end
    end
  end

  # -- reduce_max, reduce_min, argmax, argmin ----------------------------------

  describe "reduce_max, reduce_min, argmax, argmin" do
    for type <- @real_types do
      @tag requires: Map.get(@needs, type)
      test "reduce_max and reduce_min on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- Gen.shapes(), axes <- axes_for(shape), keep <- [false, true] do
          t = Gen.random(shape, type)
          assert_same(&Nx.reduce_max(&1, axes: axes, keep_axes: keep), [t])
          assert_same(&Nx.reduce_min(&1, axes: axes, keep_axes: keep), [t])
        end

        # Edge values: NaN, infinities and signed zero for floats; the extremes for integers.
        for n <- [7, 70, 4097] do
          t = with_edges(type, n)
          assert_same(&Nx.reduce_max/1, [t])
          assert_same(&Nx.reduce_min/1, [t])
        end

        assert_same(&Nx.reduce_max(&1, axes: [0]), [Gen.random({4097, 3}, type)])
        assert_same(&Nx.reduce_min(&1, axes: [1]), [Gen.random({70, 100}, type)])
      end

      @tag requires: Map.get(@needs, type)
      test "argmax and argmin on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- Gen.shapes(),
            tie <- [:low, :high],
            keep <- [false, true],
            axis <- [nil | axis_list(shape)] do
          t = Gen.random(shape, type)

          assert_same(
            &Nx.argmax(&1, axis: axis, tie_break: tie, keep_axis: keep),
            [t]
          )

          assert_same(
            &Nx.argmin(&1, axis: axis, tie_break: tie, keep_axis: keep),
            [t]
          )
        end

        # Ties on every element, and edge values (NaN beats numbers).
        for tie <- [:low, :high], n <- [7, 70, 4097] do
          assert_same(&Nx.argmax(&1, tie_break: tie), [with_edges(type, n)])
          assert_same(&Nx.argmin(&1, tie_break: tie), [with_edges(type, n)])
          flat = Nx.broadcast(Nx.tensor(3, type: type), {n})
          assert_same(&Nx.argmax(&1, tie_break: tie), [flat])
          assert_same(&Nx.argmin(&1, tie_break: tie), [flat])
        end

        assert_same(&Nx.argmax(&1, axis: 0, type: :s32), [Gen.random({4097, 3}, type)])
        assert_same(&Nx.argmin(&1, axis: 1, type: :u16), [Gen.random({70, 100}, type)])
      end
    end
  end

  # -- reduce ------------------------------------------------------------------

  describe "reduce" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "reduce with a custom body on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        init = fn t -> Nx.tensor(0, type: Nx.type(t)) end
        one = fn t -> Nx.tensor(1, type: Nx.type(t)) end

        for shape <- [{}, {1}, {63}, {65}, {3, 5}, {2, 3, 4}], axes <- axes_for(shape) do
          t = exact(type, shape)

          # The bodies Nx folds like a sum and a product.
          assert_same(
            &Nx.reduce(&1, init.(&1), [axes: axes], fn a, b -> Nx.add(a, b) end),
            [t]
          )

          # A body that is not a monoid folds in order: `x * x + acc`.
          assert_same(
            &Nx.reduce(&1, init.(&1), [axes: axes, keep_axes: true], fn a, acc ->
              Nx.add(Nx.multiply(a, a), acc)
            end),
            [t]
          )
        end

        assert_same(
          &Nx.reduce(&1, one.(&1), [axes: [1]], fn a, b -> Nx.multiply(a, b) end),
          [
            units(type, {3, 70})
          ]
        )

        # A non-identity initial value is folded in once.
        assert_same(
          &Nx.reduce(&1, Nx.tensor(3, type: Nx.type(&1)), [axes: [1]], fn a, b -> Nx.add(a, b) end),
          [exact(type, {3, 70})]
        )
      end
    end

    for type <- @real_types do
      @tag requires: Map.get(@needs, type)
      test "reduce with max and min bodies on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_same(
          &Nx.reduce(&1, Nx.Constants.min_finite(Nx.type(&1)), [axes: [1]], fn a, b ->
            Nx.max(a, b)
          end),
          [Gen.random({3, 70}, type)]
        )

        assert_same(
          &Nx.reduce(&1, Nx.Constants.max_finite(Nx.type(&1)), [axes: [0]], fn a, b ->
            Nx.min(a, b)
          end),
          [Gen.random({70, 3}, type)]
        )
      end
    end
  end

  # -- scans -------------------------------------------------------------------

  describe "cumulative ops" do
    @scan_shapes [{1}, {63}, {65}, {1000}, {4097}, {3, 5}, {2, 3, 4}, {2, 1, 3, 1, 2}]

    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "cumulative_sum and cumulative_product on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- fit(type, @scan_shapes),
            axis <- axis_list(shape),
            reverse <- [false, true] do
          assert_same(
            &Nx.cumulative_sum(&1, axis: axis, reverse: reverse),
            [exact(type, shape)]
          )

          assert_same(
            &Nx.cumulative_product(&1, axis: axis, reverse: reverse),
            [units(type, shape)]
          )
        end

        # Lines past one block, with a few and with many lines.
        for shape <- [{2, 600}, {600, 2}, {3, 5000}, {70, 300}], axis <- [0, 1] do
          assert_same(&Nx.cumulative_sum(&1, axis: axis), [
            exact(type, shape)
          ])
        end
      end
    end

    for type <- @real_types do
      @tag requires: Map.get(@needs, type)
      test "cumulative_min and cumulative_max on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- @scan_shapes, axis <- axis_list(shape), reverse <- [false, true] do
          t = Gen.random(shape, type)

          assert_same(
            &Nx.cumulative_min(&1, axis: axis, reverse: reverse),
            [t]
          )

          assert_same(
            &Nx.cumulative_max(&1, axis: axis, reverse: reverse),
            [t]
          )
        end

        for n <- [7, 70, 4097], reverse <- [false, true] do
          t = with_edges(type, n)
          assert_same(&Nx.cumulative_min(&1, reverse: reverse), [t])
          assert_same(&Nx.cumulative_max(&1, reverse: reverse), [t])
        end
      end
    end
  end

  # -- windows -----------------------------------------------------------------

  # {shape, window, opts}: strides, padding and dilations in several forms.
  @window_cases [
    {{7}, {3}, []},
    {{63}, {3}, [strides: [2]]},
    {{65}, {4}, [padding: :same]},
    {{65}, {2}, [padding: [{1, 2}], window_dilations: [3]]},
    {{4097}, {5}, [strides: [3], padding: [{2, 0}]]},
    {{3, 5}, {2, 2}, []},
    {{3, 5}, {2, 3}, [padding: :same, strides: [1, 2]]},
    {{3, 5}, {1, 2}, [window_dilations: [1, 2], padding: [{0, 0}, {1, 1}]]},
    {{2, 3, 4}, {1, 2, 2}, [strides: [1, 1, 2]]},
    {{2, 3, 4}, {2, 3, 1}, [padding: [{1, 0}, {0, 2}, {0, 0}]]},
    {{2, 1, 3, 1, 2}, {1, 1, 2, 1, 2}, [padding: :same]}
  ]

  describe "window ops" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "window_sum, window_product, window_reduce on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for {shape, window, opts} <- @window_cases do
          assert_same(&Nx.window_sum(&1, window, opts), [exact(type, shape)])

          assert_same(&Nx.window_product(&1, window, opts), [
            units(type, shape)
          ])

          assert_same(
            &Nx.window_reduce(&1, Nx.tensor(0, type: Nx.type(&1)), window, opts, fn a, b ->
              Nx.add(a, b)
            end),
            [exact(type, shape)]
          )

          # A body that is not a monoid, so the cells are folded in order.
          assert_same(
            &Nx.window_reduce(&1, Nx.tensor(1, type: Nx.type(&1)), window, opts, fn a, acc ->
              Nx.add(Nx.multiply(a, a), acc)
            end),
            [exact(type, shape)]
          )
        end

        # A non-zero initial value that is not the identity.
        assert_same(
          &Nx.window_reduce(
            &1,
            Nx.tensor(5, type: Nx.type(&1)),
            {1, 2},
            [padding: [{0, 0}, {1, 1}]],
            fn a, b ->
              Nx.add(a, b)
            end
          ),
          [exact(type, {3, 5})]
        )
      end
    end

    for type <- @real_types do
      @tag requires: Map.get(@needs, type)
      test "window_max, window_min, window_scatter on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for {shape, window, opts} <- @window_cases do
          t = Gen.random(shape, type)
          assert_same(&Nx.window_max(&1, window, opts), [t])
          assert_same(&Nx.window_min(&1, window, opts), [t])

          # The source has the shape of the windowed output. Nx's scatter ops take no
          # `window_dilations`.
          sopts = Keyword.delete(opts, :window_dilations)
          source = Nx.window_sum(exact(type, shape), window, sopts)
          init = fn t -> Nx.tensor(0, type: Nx.type(t)) end

          assert_same(
            &Nx.window_scatter_max(&1, &2, init.(&1), window, sopts),
            [t, source]
          )

          assert_same(
            &Nx.window_scatter_min(&1, &2, init.(&1), window, sopts),
            [t, source]
          )
        end

        # Special values and ties inside a window.
        t = with_edges(type, 70) |> Nx.reshape({7, 10})
        assert_same(&Nx.window_max(&1, {2, 3}), [t])
        assert_same(&Nx.window_min(&1, {2, 3}, padding: :same), [t])
      end
    end
  end

  # -- Explicit expected values --------------------------------------------------
  #
  # `Nx.BinaryBackend` is too slow at 16M elements, so these rows state the answer.

  describe "large inputs with stated answers" do
    @tag timeout: 600_000
    test "sum of 16M f32 values of 0.1 matches the exact product" do
      n = 16_777_216
      <<tenth::float-32>> = <<0.1::float-32>>
      # Scaling by a power of two is exact, so this is the exact sum of the f32 inputs.
      exact = tenth * n

      x = Nx.broadcast(Nx.tensor(0.1, type: :f32), {n}) |> Nx.backend_copy(Nx.BinaryBackend)
      out = Nx.Defn.jit(&Nx.sum/1, compiler: NxCL.Compiler).(x) |> Nx.to_number()

      assert_in_delta out, exact, exact * 1.0e-6
    end

    test "sum of 1M s32 ones is exact" do
      n = 1_048_576
      x = Nx.broadcast(Nx.tensor(1, type: :s32), {n}) |> Nx.backend_copy(Nx.BinaryBackend)
      total = Nx.Defn.jit(&Nx.sum/1, compiler: NxCL.Compiler).(x)
      assert Nx.to_number(total) == n
      # Nx 1.0 types the sum of `s32` as `s32`, not `s64`; `u8` and `s8` widen to 64 bits.
      assert total.type == {:s, 32}
      assert Nx.Defn.jit(&Nx.product/1, compiler: NxCL.Compiler).(x) |> Nx.to_number() == 1
      assert Nx.Defn.jit(&Nx.all/1, compiler: NxCL.Compiler).(x) |> Nx.to_number() == 1
    end

    test "1M s32 values against the reference" do
      x = Gen.random({1_048_576}, {:s, 32})
      assert_same(&Nx.sum/1, [x])
      assert_same(&Nx.reduce_max/1, [x])
      assert_same(&Nx.argmax(&1, tie_break: :high), [x])
      assert_same(&Nx.argmin/1, [x])
    end

    test "cumulative_sum of 1M s32 ones counts from 1" do
      n = 1_048_576
      x = Nx.broadcast(Nx.tensor(1, type: :s32), {n}) |> Nx.backend_copy(Nx.BinaryBackend)
      out = Nx.Defn.jit(&Nx.cumulative_sum/1, compiler: NxCL.Compiler).(x)
      assert Nx.to_binary(out) == Nx.to_binary(Nx.iota({n}, type: :s32) |> Nx.add(1))

      rev = Nx.Defn.jit(&Nx.cumulative_sum(&1, reverse: true), compiler: NxCL.Compiler).(x)

      assert Nx.to_binary(rev) ==
               Nx.to_binary(Nx.iota({n}, type: :s32) |> Nx.reverse() |> Nx.add(1))
    end
  end
end
