defmodule NxCL.ShapeTest do
  @moduledoc """
  Differential rows for the shape and data-movement ops: `broadcast`,
  `transpose`, `reverse`, `slice`, `put_slice`, `pad`, `concatenate`, `stack`,
  `iota`, `eye`, a constant of a shape, `metadata` and `elem`, on every dtype
  through `NxCL.TestSupport.Compare` against `Nx.BinaryBackend`. A second group states the
  expected values itself: dynamic starts below 0 and past the end, and rank 0.

  `f64` and `c128` rows carry `requires: :fp64`; `f16` rows carry
  `requires: :fp16`. The ops only move values, so the same rows cover `c64`
  and `c128` with no extra code.

  Nx 1.0 rejects a zero-size dimension, so zero-size inputs are checked in
  `NxCL.ShapeOpsTest` through the native wire.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl

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

  # The extension a dtype needs; `nil` means none.
  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  defp data(type, shape), do: Gen.random(shape, type)

  defp scalar(type, n), do: Nx.tensor(n, type: type)

  # -- Every op, every dtype ---------------------------------------------------

  describe "movement ops" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "broadcast, transpose and reverse on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_same(&Nx.broadcast(&1, {2, 3, 4}, axes: [1, 2]), [data(type, {3, 1})])

        assert_same(&Nx.broadcast(&1, {2, 3, 4}, axes: [0, 1]), [data(type, {2, 3})])

        assert_same(&Nx.broadcast(&1, {5, 7}), [data(type, {})])
        assert_same(&Nx.broadcast(&1, {4097}, axes: [0]), [data(type, {1})])
        assert_same(&Nx.broadcast(&1, {3, 65}), [data(type, {65})])

        assert_same(&Nx.transpose/1, [data(type, {3, 5})])
        assert_same(&Nx.transpose(&1, axes: [2, 0, 1]), [data(type, {2, 3, 4})])

        assert_same(&Nx.transpose(&1, axes: [4, 2, 0, 1, 3]), [
          data(type, {2, 1, 3, 1, 2})
        ])

        assert_same(&Nx.transpose/1, [data(type, {})])

        assert_same(&Nx.reverse/1, [data(type, {4097})])
        assert_same(&Nx.reverse(&1, axes: [0, 2]), [data(type, {2, 3, 4})])
        assert_same(&Nx.reverse(&1, axes: [1]), [data(type, {3, 5})])
        assert_same(&Nx.reverse/1, [data(type, {})])
      end

      @tag requires: Map.get(@needs, type)
      test "slice and put_slice on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_same(&Nx.slice(&1, [1, 0, 1], [1, 3, 2]), [data(type, {2, 3, 4})])

        assert_same(&Nx.slice(&1, [0, 1, 0], [2, 2, 4], strides: [1, 1, 3]), [
          data(type, {2, 3, 4})
        ])

        assert_same(&Nx.slice(&1, [3], [1000], strides: [7]), [data(type, {1000})])
        assert_same(&Nx.slice(&1, [], []), [data(type, {})])
        assert_same(&Nx.slice(&1, [1], [60]), [data(type, {65})])

        # Dynamic starts, in range.
        assert_same(
          fn x, i, j -> Nx.slice(x, [i, j], [2, 3]) end,
          [data(type, {5, 6}), scalar({:s, 32}, 2), scalar({:u, 8}, 1)]
        )

        assert_same(
          &Nx.put_slice(&1, [1, 2], &2),
          [data(type, {4, 5}), data(type, {2, 3})]
        )

        assert_same(
          fn x, u, i -> Nx.put_slice(x, [i, 0], u) end,
          [
            data(type, {4097}) |> Nx.reshape({4097, 1}),
            data(type, {5, 1}),
            scalar({:s, 64}, 4000)
          ]
        )

        # Rank 0 `put_slice` crashes `BinaryBackend`; an explicit row covers it below.

        assert_same(&Nx.put_slice(&1, [0], &2), [data(type, {65}), data(type, {65})])
      end

      @tag requires: Map.get(@needs, type)
      test "pad on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        zero = scalar(type, 0)
        seven = scalar(type, 7)

        assert_same(&Nx.pad(&1, zero, [{1, 2, 0}, {0, 0, 0}]), [data(type, {3, 5})])
        assert_same(&Nx.pad(&1, seven, [{1, 1, 2}, {2, 0, 1}]), [data(type, {3, 5})])
        # Negative low and high padding, with and without interior padding.
        assert_same(&Nx.pad(&1, seven, [{-1, -2, 0}, {2, -1, 1}]), [data(type, {5, 4})])
        assert_same(&Nx.pad(&1, zero, [{-2, 3, 2}]), [data(type, {65})])
        assert_same(&Nx.pad(&1, seven, [{0, 0, 0}]), [data(type, {63})])
        assert_same(&Nx.pad(&1, seven, []), [data(type, {})])

        assert_same(&Nx.pad(&1, seven, [{0, 1, 1}, {1, 0, 0}, {2, 2, 0}]), [
          data(type, {2, 3, 4})
        ])

        # The pad value comes from a tensor argument.
        assert_same(fn x, v -> Nx.pad(x, v, [{2, 1, 1}]) end, [
          data(type, {10}),
          data(type, {})
        ])
      end

      @tag requires: Map.get(@needs, type)
      test "concatenate and stack on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_same(
          fn a, b, c -> Nx.concatenate([a, b, c], axis: 1) end,
          [data(type, {2, 3}), data(type, {2, 1}), data(type, {2, 4})]
        )

        assert_same(
          fn a, b -> Nx.concatenate([a, b]) end,
          [data(type, {65}), data(type, {63})]
        )

        assert_same(
          fn a, b -> Nx.concatenate([a, b], axis: 2) end,
          [data(type, {2, 3, 4}), data(type, {2, 3, 1})]
        )

        assert_same(fn a -> Nx.concatenate([a, a, a]) end, [data(type, {1000})])

        for axis <- 0..2 do
          assert_same(
            fn a, b -> Nx.stack([a, b, a], axis: axis) end,
            [data(type, {2, 3}), data(type, {2, 3})]
          )
        end

        assert_same(fn a, b -> Nx.stack([a, b]) end, [data(type, {}), data(type, {})])
        assert_same(fn a -> Nx.stack([a, a]) end, [data(type, {4097})])
      end

      @tag requires: Map.get(@needs, type)
      test "iota, eye and constants of a shape on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        assert_same(fn -> Nx.iota({2, 3, 4}, axis: 1, type: type) end, [])
        assert_same(fn -> Nx.iota({2, 3, 4}, axis: 2, type: type) end, [])
        assert_same(fn -> Nx.iota({2, 3, 4}, type: type) end, [])
        assert_same(fn -> Nx.iota({65}, type: type) end, [])
        assert_same(fn -> Nx.iota({}, type: type) end, [])
        assert_same(fn -> Nx.iota({4097}, axis: 0, type: type) end, [])

        assert_same(fn -> Nx.eye({3, 3}, type: type) end, [])
        assert_same(fn -> Nx.eye({2, 3, 5}, type: type) end, [])
        assert_same(fn -> Nx.eye(65, type: type) end, [])

        # A constant of a shape is a scalar constant that fills the shape.
        c = scalar(type, 5)

        assert_same(fn x -> Nx.add(x, Nx.broadcast(c, {2, 3})) end, [
          data(type, {2, 3})
        ])

        assert_same(fn -> Nx.broadcast(c, {63}) end, [])

        # A scalar tensor next to a tensor gets an explicit broadcast.
        assert_same(fn x, s -> Nx.add(x, s) end, [data(type, {65}), data(type, {})])

        assert_same(fn x, y -> Nx.add(x, y) end, [
          data(type, {2, 3}),
          data(type, {3})
        ])

        assert_same(fn x, y -> Nx.add(x, y) end, [
          data(type, {2, 1, 4}),
          data(type, {3, 1})
        ])
      end

      @tag requires: Map.get(@needs, type)
      test "metadata, elem, reshape and squeeze on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data(type, {2, 3})

        assert_same(fn x -> x end, [x])
        literal = Nx.reverse(x)
        assert_same(fn x -> Nx.add(x, literal) end, [x])
        assert_same(fn x -> Nx.Defn.Kernel.stop_grad(x) end, [x])

        # A tuple through `metadata`: `elem` picks each element.
        assert_same(
          fn x ->
            {a, b} = Nx.Defn.Kernel.stop_grad({x, Nx.reverse(x)})
            Nx.concatenate([a, b])
          end,
          [x]
        )

        assert_same(&Nx.reshape(&1, {3, 2}), [x])
        assert_same(fn x -> x |> Nx.reshape({1, 6, 1}) |> Nx.squeeze() end, [x])
      end
    end
  end

  describe "imag, real and conjugate of real tensors" do
    for type <-
          [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}, {:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}] ++
            [{:f, 16}, {:bf, 16}, {:f, 32}, {:f, 64}] do
      @tag requires: Map.get(@needs, type)
      test "#{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data(type, {65})
        # `Nx.imag` of a real tensor traces to a zero broadcast.
        assert_same(&Nx.imag/1, [x])
        assert_same(&Nx.imag/1, [data(type, {2, 3})])
        assert_same(&Nx.real/1, [x])
        assert_same(&Nx.conjugate/1, [x], signed_zero: true)
        assert_same(&Nx.phase/1, [x])
      end
    end
  end

  # -- Rows with explicit expected values -----------------------------------------

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  describe "dynamic starts clamp to [0, dim - length]" do
    # `BinaryBackend` may differ here, so the expected values are written out.
    @x Nx.iota({5, 6}, type: :s32)

    test "slice with a start below 0 starts at 0" do
      out = run(fn x, i, j -> Nx.slice(x, [i, j], [2, 3]) end, [@x, Nx.tensor(-4), Nx.tensor(-1)])
      assert out == Nx.tensor([[0, 1, 2], [6, 7, 8]])
    end

    test "slice with a start past the end ends at the last element" do
      out = run(fn x, i, j -> Nx.slice(x, [i, j], [2, 3]) end, [@x, Nx.tensor(9), Nx.tensor(6)])
      assert out == Nx.tensor([[21, 22, 23], [27, 28, 29]])
    end

    test "slice with a start of exactly dim - length keeps it" do
      out = run(fn x, i, j -> Nx.slice(x, [i, j], [2, 3]) end, [@x, Nx.tensor(3), Nx.tensor(3)])
      assert out == Nx.tensor([[21, 22, 23], [27, 28, 29]])
    end

    test "a strided slice clamps by its input span" do
      # Span 5 with stride 2 gives 3 elements; a start of 4 clamps to 6 - 5 = 1.
      out =
        run(fn x, i -> Nx.slice(x, [0, i], [1, 5], strides: [1, 2]) end, [@x, Nx.tensor(4)])

      assert out == Nx.tensor([[1, 3, 5]])
    end

    test "starts of every integer type clamp the same way" do
      for type <- [{:u, 8}, {:u, 64}, {:s, 8}, {:s, 64}] do
        i = Nx.tensor(100, type: type)
        out = run(fn x, i -> Nx.slice(x, [i, 0], [1, 2]) end, [@x, i])
        assert out == Nx.tensor([[24, 25]])
      end

      out = run(fn x, i -> Nx.slice(x, [i, 0], [1, 2]) end, [@x, Nx.tensor(-100, type: {:s, 8})])
      assert out == Nx.tensor([[0, 1]])
    end

    test "put_slice with a start below 0 writes at 0" do
      out =
        run(
          fn x, u, i, j -> Nx.put_slice(x, [i, j], u) end,
          [@x, Nx.tensor([[-1, -2], [-3, -4]]), Nx.tensor(-7), Nx.tensor(-7)]
        )

      expected = @x |> Nx.put_slice([0, 0], Nx.tensor([[-1, -2], [-3, -4]]))
      assert out == expected
      assert Nx.to_flat_list(out) |> Enum.take(7) == [-1, -2, 2, 3, 4, 5, -3]
    end

    test "put_slice with a start past the end writes at the last window" do
      out =
        run(
          fn x, u, i, j -> Nx.put_slice(x, [i, j], u) end,
          [@x, Nx.tensor([[-1, -2], [-3, -4]]), Nx.tensor(50), Nx.tensor(50)]
        )

      rows = Nx.to_list(out)
      assert Enum.at(rows, 3) == [18, 19, 20, 21, -1, -2]
      assert Enum.at(rows, 4) == [24, 25, 26, 27, -3, -4]
      assert Enum.at(rows, 2) == [12, 13, 14, 15, 16, 17]
    end

    test "put_slice does not change its input" do
      u = Nx.tensor([[9, 9], [9, 9]])
      out = run(fn x, u -> Nx.put_slice(x, [1, 1], u) end, [@x, u])
      assert out != @x
      assert run(fn x -> Nx.add(x, 0) end, [@x]) == @x
    end
  end

  describe "rank 0 and one element" do
    test "scalars go through every op that takes them" do
      s = Nx.tensor(3.5)
      assert run(&Nx.broadcast(&1, {2}), [s]) == Nx.tensor([3.5, 3.5])
      assert run(&Nx.reverse/1, [s]) == s
      assert run(&Nx.transpose/1, [s]) == s
      assert run(fn x -> Nx.stack([x, x]) end, [s]) == Nx.tensor([3.5, 3.5])
      assert run(fn -> Nx.iota({}, type: :s32) end, []) == Nx.tensor(0)
      assert run(&Nx.pad(&1, 0.0, []), [s]) == s
    end

    test "put_slice of a scalar replaces it" do
      assert run(&Nx.put_slice(&1, [], &2), [Nx.tensor(1.5), Nx.tensor(8.0)]) == Nx.tensor(8.0)
    end

    test "a one-element tensor slices, pads and joins" do
      x = Nx.tensor([4.0])
      assert run(&Nx.slice(&1, [0], [1]), [x]) == x
      assert run(&Nx.pad(&1, 0.0, [{1, 1, 0}]), [x]) == Nx.tensor([0.0, 4.0, 0.0])
      assert run(fn a -> Nx.concatenate([a, a]) end, [x]) == Nx.tensor([4.0, 4.0])
    end
  end

  describe "rank up to the cap" do
    test "a rank 8 tensor transposes (rank 9 is checked in NxCL.ShapeOpsTest)" do
      shape = {1, 2, 1, 2, 1, 2, 1, 2}
      x = Nx.iota(shape, type: :s32)
      axes = Enum.to_list(7..0//-1)
      out = run(&Nx.transpose(&1, axes: axes), [x])
      assert out == Nx.transpose(x, axes: axes)
    end
  end

  describe "a bias-add layer" do
    test "dot, broadcast, add and sigmoid run through the compiler on f32" do
      layer = fn x, w, b -> x |> Nx.dot(w) |> Nx.add(Nx.broadcast(b, {4, 3})) |> Nx.sigmoid() end

      assert_same(
        layer,
        [Gen.random({4, 5}, {:f, 32}), Gen.random({5, 3}, {:f, 32}), Gen.random({3}, {:f, 32})],
        rtol: 1.0e-4,
        atol: 1.0e-4
      )
    end
  end
end
