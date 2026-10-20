defmodule NxCL.CompilerTest do
  @moduledoc """
  Compiled results against `Nx.Defn.Evaluator` on `Nx.BinaryBackend`, through
  `NxCL.TestSupport.Compare`.
  """
  use ExUnit.Case, async: false
  import NxCL.TestSupport.Compare

  @moduletag :opencl

  describe "single ops" do
    test "exp" do
      assert_same(fn x -> Nx.exp(x) end, [Nx.tensor([0.0, 1.0, 2.0])])
    end

    test "log" do
      assert_same(fn x -> Nx.log(x) end, [Nx.tensor([1.0, 2.0, 3.0])])
    end

    test "negate" do
      assert_same(fn x -> Nx.negate(x) end, [Nx.tensor([1.0, -2.0, 3.0])])
    end

    test "sigmoid" do
      assert_same(fn x -> Nx.sigmoid(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "tanh" do
      assert_same(fn x -> Nx.tanh(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "abs" do
      assert_same(fn x -> Nx.abs(x) end, [Nx.tensor([-1.0, 2.0, -3.0])])
    end
  end

  describe "binary ops" do
    test "add" do
      assert_same(
        fn x, y -> Nx.add(x, y) end,
        [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([10.0, 20.0, 30.0])]
      )
    end

    test "subtract" do
      assert_same(
        fn x, y -> Nx.subtract(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([1.0, 2.0, 3.0])]
      )
    end

    test "multiply" do
      assert_same(
        fn x, y -> Nx.multiply(x, y) end,
        [Nx.tensor([2.0, 3.0, 4.0]), Nx.tensor([5.0, 5.0, 5.0])]
      )
    end

    test "divide" do
      assert_same(
        fn x, y -> Nx.divide(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([2.0, 4.0, 5.0])]
      )
    end
  end

  describe "shape ops" do
    test "reshape" do
      assert_same(fn x -> Nx.reshape(x, {2, 3}) end, [Nx.iota({6}, type: :f32)])
    end

    test "squeeze" do
      assert_same(fn x -> Nx.squeeze(x) end, [Nx.iota({1, 3, 1}, type: :f32)])
    end
  end

  describe "chains" do
    test "exp then log (identity)" do
      assert_same(fn x -> x |> Nx.exp() |> Nx.log() end, [Nx.tensor([1.0, 2.0, 3.0])])
    end

    test "add then negate" do
      assert_same(
        fn x, y -> Nx.add(x, y) |> Nx.negate() end,
        [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
      )
    end

    test "multiply then sigmoid" do
      assert_same(
        fn x, y -> Nx.multiply(x, y) |> Nx.sigmoid() end,
        [Nx.tensor([0.5, 1.0, 1.5]), Nx.tensor([2.0, 2.0, 2.0])]
      )
    end
  end

  describe "matmul" do
    test "simple 2x2 matmul" do
      assert_same(
        fn x, w -> Nx.dot(x, w) end,
        [
          Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
        ]
      )
    end

    test "non-square matmul" do
      key = Nx.Random.key(42)
      {a, key} = Nx.Random.uniform(key, shape: {4, 8})
      {b, _key} = Nx.Random.uniform(key, shape: {8, 6})

      assert_same(fn x, w -> Nx.dot(x, w) end, [a, b])
    end
  end

  describe "dense layer pattern" do
    test "dot + add" do
      assert_same(
        fn x, w, b -> x |> Nx.dot(w) |> Nx.add(b) end,
        [
          Nx.tensor([[1.0, 2.0]]),
          Nx.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
          Nx.tensor([[0.1, 0.2, 0.3]])
        ]
      )
    end
  end

  describe "native dtypes" do
    test "s32 add keeps the s32 type" do
      assert_same(
        fn x, y -> Nx.add(x, y) end,
        [Nx.tensor([1, 2, 3], type: :s32), Nx.tensor([10, 20, 30], type: :s32)]
      )
    end
  end

  describe "containers" do
    test "a tuple with a nested map keeps its shape" do
      assert_same(
        fn x -> {Nx.exp(x), %{b: Nx.negate(x)}} end,
        [Nx.tensor([0.0, 1.0])]
      )
    end

    test "an output that is an argument passes through" do
      assert_same(fn x -> {x, Nx.exp(x)} end, [Nx.tensor([0.0, 1.0])])
    end

    test "container arguments" do
      assert_same(
        fn %{a: a, b: b} -> Nx.add(a, b) end,
        [%{a: Nx.tensor([1.0, 2.0]), b: Nx.tensor([3.0, 4.0])}]
      )
    end

    test "an ignored first argument keeps the input positions" do
      assert_same(
        fn _a, b -> Nx.exp(b) end,
        [Nx.tensor([1.0, 2.0]), Nx.tensor([0.0, 1.0])]
      )
    end

    test "names survive" do
      assert_same(
        fn x -> Nx.exp(x) end,
        [Nx.iota({2, 2}, type: :f32, names: [:r, :c])]
      )
    end
  end

  describe "errors" do
    test "a dtype with no OpenCL type raises NxCL.CompilerError" do
      jitted = Nx.Defn.jit(fn x -> Nx.negate(x) end, compiler: NxCL.Compiler)

      assert_raise NxCL.CompilerError, ~r/no OpenCL type/, fn ->
        jitted.(Nx.tensor([1, 2], type: {:s, 4}))
      end
    end

    test "an unsupported op raises ArgumentError when compiled" do
      # Every Nx 1.0 op lowers, so the node carries an op Nx does not define.
      unknown = fn x ->
        %{x | data: %Nx.Defn.Expr{id: make_ref(), op: :not_an_nx_op, args: [x], context: nil}}
      end

      jitted = Nx.Defn.jit(unknown, compiler: NxCL.Compiler)

      assert_raise ArgumentError, ~r/^does not yet lower op/, fn ->
        jitted.(Nx.tensor([1.0, 2.0]))
      end
    end
  end

  describe "determinism" do
    test "same result on repeated calls" do
      fun = fn x -> Nx.sigmoid(x) end
      jitted = Nx.Defn.jit(fun, compiler: NxCL.Compiler)
      input = Nx.tensor([1.0, 2.0, 3.0])

      r1 = jitted.(input) |> Nx.to_binary()
      r2 = jitted.(input) |> Nx.to_binary()
      assert r1 == r2
    end
  end
end
