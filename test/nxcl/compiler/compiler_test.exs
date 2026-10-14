defmodule NxCL.CompilerTest do
  @moduledoc """
  Full compiler round-trip tests: trace → lower → execute → verify.
  Compares compiled results against `Nx.Defn.Evaluator` on the binary backend
  through `NxCL.Parity`, which also records the passing cells.
  """
  use ExUnit.Case, async: false
  import NxCL.Parity

  @moduletag :opencl

  describe "single ops" do
    test "exp" do
      assert_parity(:exp, fn x -> Nx.exp(x) end, [Nx.tensor([0.0, 1.0, 2.0])])
    end

    test "log" do
      assert_parity(:log, fn x -> Nx.log(x) end, [Nx.tensor([1.0, 2.0, 3.0])])
    end

    test "negate" do
      assert_parity(:negate, fn x -> Nx.negate(x) end, [Nx.tensor([1.0, -2.0, 3.0])])
    end

    test "sigmoid" do
      assert_parity(:sigmoid, fn x -> Nx.sigmoid(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "tanh" do
      assert_parity(:tanh, fn x -> Nx.tanh(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "abs" do
      assert_parity(:abs, fn x -> Nx.abs(x) end, [Nx.tensor([-1.0, 2.0, -3.0])])
    end
  end

  describe "binary ops" do
    test "add" do
      assert_parity(
        :add,
        fn x, y -> Nx.add(x, y) end,
        [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([10.0, 20.0, 30.0])]
      )
    end

    test "subtract" do
      assert_parity(
        :subtract,
        fn x, y -> Nx.subtract(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([1.0, 2.0, 3.0])]
      )
    end

    test "multiply" do
      assert_parity(
        :multiply,
        fn x, y -> Nx.multiply(x, y) end,
        [Nx.tensor([2.0, 3.0, 4.0]), Nx.tensor([5.0, 5.0, 5.0])]
      )
    end

    test "divide" do
      assert_parity(
        :divide,
        fn x, y -> Nx.divide(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([2.0, 4.0, 5.0])]
      )
    end
  end

  describe "shape ops" do
    test "reshape" do
      assert_parity(:reshape, fn x -> Nx.reshape(x, {2, 3}) end, [Nx.iota({6}, type: :f32)])
    end

    test "squeeze" do
      assert_parity(:squeeze, fn x -> Nx.squeeze(x) end, [Nx.iota({1, 3, 1}, type: :f32)])
    end
  end

  describe "chains" do
    test "exp then log (identity)" do
      assert_parity(:exp_log, fn x -> x |> Nx.exp() |> Nx.log() end, [Nx.tensor([1.0, 2.0, 3.0])],
        record: false
      )
    end

    test "add then negate" do
      assert_parity(
        :add_negate,
        fn x, y -> Nx.add(x, y) |> Nx.negate() end,
        [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])],
        record: false
      )
    end

    test "multiply then sigmoid" do
      assert_parity(
        :multiply_sigmoid,
        fn x, y -> Nx.multiply(x, y) |> Nx.sigmoid() end,
        [Nx.tensor([0.5, 1.0, 1.5]), Nx.tensor([2.0, 2.0, 2.0])],
        record: false
      )
    end
  end

  describe "matmul" do
    test "simple 2x2 matmul" do
      assert_parity(
        :dot,
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

      assert_parity(:dot, fn x, w -> Nx.dot(x, w) end, [a, b])
    end
  end

  describe "dense layer pattern" do
    test "dot + add" do
      assert_parity(
        :dot_add,
        fn x, w, b -> x |> Nx.dot(w) |> Nx.add(b) end,
        [
          Nx.tensor([[1.0, 2.0]]),
          Nx.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
          Nx.tensor([[0.1, 0.2, 0.3]])
        ],
        record: false
      )
    end
  end

  describe "native dtypes" do
    test "s32 add keeps the s32 type" do
      assert_parity(
        :add,
        fn x, y -> Nx.add(x, y) end,
        [Nx.tensor([1, 2, 3], type: :s32), Nx.tensor([10, 20, 30], type: :s32)]
      )
    end
  end

  describe "containers" do
    test "a tuple with a nested map keeps its shape" do
      assert_parity(
        :exp,
        fn x -> {Nx.exp(x), %{b: Nx.negate(x)}} end,
        [Nx.tensor([0.0, 1.0])],
        record: false
      )
    end

    test "an output that is an argument passes through" do
      assert_parity(:exp, fn x -> {x, Nx.exp(x)} end, [Nx.tensor([0.0, 1.0])], record: false)
    end

    test "container arguments" do
      assert_parity(
        :add,
        fn %{a: a, b: b} -> Nx.add(a, b) end,
        [%{a: Nx.tensor([1.0, 2.0]), b: Nx.tensor([3.0, 4.0])}],
        record: false
      )
    end

    test "an ignored first argument keeps the input positions" do
      assert_parity(
        :exp,
        fn _a, b -> Nx.exp(b) end,
        [Nx.tensor([1.0, 2.0]), Nx.tensor([0.0, 1.0])],
        record: false
      )
    end

    test "names survive" do
      assert_parity(
        :exp,
        fn x -> Nx.exp(x) end,
        [Nx.iota({2, 2}, type: :f32, names: [:r, :c])],
        record: false
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
      jitted = Nx.Defn.jit(fn x -> Nx.fft(x) end, compiler: NxCL.Compiler)

      assert_raise ArgumentError, ~r/^does not yet lower op/, fn ->
        jitted.(Nx.tensor([1.0, 2.0], type: :c64))
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
