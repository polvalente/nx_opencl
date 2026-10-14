defmodule NxCL.LowerTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias NxCL.Lower
  alias NxCL.Program
  alias NxCL.Program.Instruction

  defp trace(fun, args), do: Nx.Defn.debug_expr_apply(fun, args)

  defp lower(fun, args) do
    flat = args |> Nx.Defn.Composite.flatten_list() |> length()
    fun |> trace(args) |> Lower.lower(flat)
  end

  @f32 {:f, 32}

  defn count_up(n) do
    while n, Nx.less(n, 10) do
      n + 1
    end
  end

  test "dot + add + sigmoid lowers instruction by instruction" do
    program =
      lower(
        fn x, w, b -> x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid() end,
        [Nx.iota({1, 2}, type: :f32), Nx.iota({2, 3}, type: :f32), Nx.iota({1, 3}, type: :f32)]
      )

    assert %Program{
             inputs: [{@f32, {1, 2}}, {@f32, {2, 3}}, {@f32, {1, 3}}],
             captures: [],
             constants: [],
             outputs: [{:result, 2}],
             ref_counts: %{
               {:input, 0} => 1,
               {:input, 1} => 1,
               {:input, 2} => 1,
               {:result, 0} => 1,
               {:result, 1} => 1,
               {:result, 2} => 1
             }
           } = program

    assert [
             %Instruction{
               op: :dot,
               operands: [{:input, 0}, {:input, 1}],
               out_type: @f32,
               out_shape: {1, 3},
               attrs: [2]
             },
             %Instruction{
               op: :add,
               operands: [{:result, 0}, {:input, 2}],
               out_type: @f32,
               out_shape: {1, 3},
               attrs: []
             },
             %Instruction{
               op: :sigmoid,
               operands: [{:result, 1}],
               out_type: @f32,
               out_shape: {1, 3},
               attrs: []
             }
           ] = program.instructions
  end

  test "an ignored first argument keeps the input positions" do
    program = lower(fn _a, b -> Nx.exp(b) end, [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])])

    assert [nil, {@f32, {2}}] = program.inputs
    assert [%Instruction{op: :exp, operands: [{:input, 1}]}] = program.instructions
  end

  test "trailing unused arguments are kept" do
    program = lower(fn a, _b -> Nx.exp(a) end, [Nx.tensor([1.0]), Nx.tensor([2.0])])
    assert [{@f32, {1}}, nil] = program.inputs
  end

  test "a scalar constant goes in the constants table" do
    program = lower(fn x -> Nx.add(x, 2.0) end, [Nx.tensor(1.0)])

    assert [{2.0, @f32}] = program.constants
    assert [%Instruction{op: :add, operands: [{:const, 0}, {:input, 0}]}] = program.instructions
  end

  test "a literal tensor goes in the captures" do
    literal = Nx.tensor([1.0, 2.0])
    program = lower(fn x -> Nx.add(x, literal) end, [Nx.tensor([3.0, 4.0])])

    assert [^literal] = program.captures
    assert [%Instruction{op: :add, operands: [{:input, 0}, {:capture, 0}]}] = program.instructions
  end

  test "reshape and squeeze carry the alias attr" do
    program =
      lower(fn x -> x |> Nx.reshape({1, 3}) |> Nx.squeeze() end, [Nx.iota({3}, type: :f32)])

    assert [
             %Instruction{op: :reshape, attrs: [:alias], out_shape: {1, 3}},
             %Instruction{op: :squeeze, attrs: [:alias], out_shape: {3}}
           ] = program.instructions
  end

  test "a shared node is lowered once and counted twice" do
    program =
      lower(
        fn x ->
          y = Nx.negate(x)
          Nx.add(y, y)
        end,
        [Nx.tensor([1.0])]
      )

    assert [%Instruction{op: :negate}, %Instruction{op: :add}] = program.instructions
    assert program.ref_counts[{:result, 0}] == 2
  end

  describe "containers" do
    test "outputs are the flattened leaves and the template keeps the shape" do
      program =
        lower(fn x -> {Nx.exp(x), %{b: Nx.negate(x)}} end, [Nx.tensor([1.0, 2.0])])

      assert [{:result, 0}, {:result, 1}] = program.outputs
      assert {%Nx.Tensor{}, %{b: %Nx.Tensor{}}} = program.output_template
    end

    test "an output that is an input passes through as a ref" do
      program = lower(fn x -> {x, Nx.exp(x)} end, [Nx.tensor([1.0])])
      assert [{:input, 0}, {:result, 0}] = program.outputs
      assert program.ref_counts[{:input, 0}] == 2
    end

    test "container arguments are flattened in order" do
      program =
        lower(
          fn %{a: a, b: b} -> Nx.add(a, b) end,
          [%{a: Nx.tensor([1.0]), b: Nx.tensor([2.0])}]
        )

      assert [{@f32, {1}}, {@f32, {1}}] = program.inputs
    end
  end

  describe "unsupported input" do
    test "an op that is not lowered raises with the fixed prefix" do
      assert_raise ArgumentError, ~r/^does not yet lower op :sin/, fn ->
        lower(fn x -> Nx.sin(x) end, [Nx.tensor([1.0])])
      end

      assert_raise ArgumentError, ~r/^does not yet lower op :sum/, fn ->
        lower(fn x -> Nx.sum(x) end, [Nx.tensor([1.0])])
      end
    end

    test "a scalar mixed with a tensor raises (broadcast is not lowered)" do
      assert_raise ArgumentError, ~r/^does not yet lower op :add/, fn ->
        lower(fn x -> Nx.add(x, 1.0) end, [Nx.tensor([1.0, 2.0])])
      end
    end

    test "a dot that is not a 2D matmul raises" do
      assert_raise ArgumentError, ~r/^does not yet lower op :dot/, fn ->
        lower(fn x, y -> Nx.dot(x, y) end, [Nx.tensor([1.0, 2.0]), Nx.tensor([1.0, 2.0])])
      end
    end

    test "a while loop raises" do
      assert_raise ArgumentError, ~r/^does not yet lower op :while/, fn ->
        lower(&count_up/1, [Nx.tensor(0)])
      end
    end
  end
end
