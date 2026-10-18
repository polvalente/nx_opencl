defmodule NxCL.ReduceOpsTest do
  @moduledoc """
  Lowering and runtime behaviour of reductions, scans, windows and sort that the
  differential rows cannot show: which instructions lowering emits, that
  `reduce` with an add body shares the kernel of `sum`, which bodies raise,
  softmax on a batch, and zero-size inputs, which go through the native wire
  because Nx 1.0 cannot build them.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Lower
  alias NxCL.Native

  @moduletag :opencl

  @f32 {:f, 32}

  defp lower(fun, args) do
    flat = args |> Nx.Defn.Composite.flatten_list() |> length()
    fun |> Nx.Defn.debug_expr_apply(args) |> Lower.lower(flat)
  end

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  describe "lowering" do
    test "reduce with an add body lowers to semantic steps" do
      x = Nx.iota({3, 7}, type: :f32)

      sum = lower(&Nx.sum(&1, axes: [1]), [x]).instructions

      generic =
        lower(
          &Nx.reduce(&1, Nx.tensor(0.0), [axes: [1]], fn a, b -> Nx.add(a, b) end),
          [x]
        ).instructions

      assert [%Instruction{op: :sum, attrs: [1]}] = sum
      # A body of one add is the steps `[add(0, 1)]`; the native code folds that into the
      # same kernel as `sum` (next test).
      assert [%Instruction{op: :reduce, attrs: [1, :add, 0, 1, 1]}] = generic
    end

    test "each reduction, scan, window and sort lowers to one instruction" do
      x = Nx.iota({4, 6}, type: :f32)

      one = fn fun ->
        [%Instruction{} = i] = lower(fun, [x]).instructions
        i
      end

      assert %{op: :reduce_max, attrs: [0, 1]} = one.(&Nx.reduce_max(&1, axes: [0, 1]))
      assert %{op: :argmax, attrs: [:high, 1]} = one.(&Nx.argmax(&1, axis: 1, tie_break: :high))

      assert %{op: :cumulative_sum, attrs: [1, 1]} =
               one.(&Nx.cumulative_sum(&1, axis: 1, reverse: true))

      assert %{op: :window_max} = one.(&Nx.window_max(&1, {2, 2}))
      assert %{op: :sort, attrs: [0, :desc]} = one.(&Nx.sort(&1, axis: 0, direction: :desc))
      assert %{op: :argsort, attrs: [1, :asc]} = one.(&Nx.argsort(&1, axis: 1))
    end

    test "a body outside the supported steps raises, naming the op" do
      x = Nx.iota({3, 7}, type: :f32)

      assert_raise ArgumentError, ~r/does not yet lower op :sin/, fn ->
        lower(&Nx.reduce(&1, Nx.tensor(0.0), [axes: [1]], fn a, b -> Nx.add(Nx.sin(a), b) end), [
          x
        ])
      end

      assert_raise ArgumentError, ~r/does not yet lower op :constant/, fn ->
        lower(
          &Nx.reduce(&1, Nx.tensor(0.0), [axes: [1]], fn a, b -> Nx.add(Nx.add(a, b), 1) end),
          [x]
        )
      end

      assert_raise ArgumentError, ~r/does not yet lower op :sin/, fn ->
        lower(
          &Nx.window_reduce(&1, Nx.tensor(0.0), {1, 2}, fn a, b -> Nx.add(Nx.sin(a), b) end),
          [x]
        )
      end
    end
  end

  describe "kernels" do
    test "reduce(&Nx.add/2) gives the result of sum" do
      x = Nx.iota({3, 41}, type: :f32)
      sum = &Nx.sum(&1, axes: [1])
      add = &Nx.reduce(&1, Nx.tensor(0.0), [axes: [1]], fn a, b -> Nx.add(a, b) end)

      assert Nx.to_binary(run(sum, [x])) == Nx.to_binary(run(add, [x]))
    end

    test "a thread-sized line, a long line and many outputs reduce correctly" do
      x = Nx.iota({1, 20}, type: :f32)
      assert Nx.to_number(run(&Nx.sum/1, [x])) == 190.0

      # Few outputs and a long line: partial sums, then a fold of the partials.
      y = Nx.broadcast(Nx.tensor(1.0), {2, 100_000})
      out = run(&Nx.sum(&1, axes: [1]), [y])
      assert Nx.to_flat_list(out) == [100_000.0, 100_000.0]

      # Many outputs: one group per output, one pass.
      z = Nx.broadcast(Nx.tensor(1.0), {128, 1000})
      out = run(&Nx.sum(&1, axes: [1]), [z])
      assert Nx.to_flat_list(out) == List.duplicate(1000.0, 128)
    end

    test "softmax on a 128 by 1000 batch matches the reference" do
      softmax = fn x ->
        e = Nx.exp(Nx.subtract(x, Nx.reduce_max(x, axes: [1], keep_axes: true)))
        Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
      end

      x = NxCL.TestSupport.Gen.random({128, 1000}, @f32)
      assert_same(softmax, [x], rtol: 1.0e-4, atol: 1.0e-6)
    end
  end

  describe "argmax and argmin ties and NaN" do
    test "tie_break picks the first or last of equal maxima and minima" do
      x = Nx.tensor([[1, 5, 5, 2, 5], [3, 3, 1, 1, 3]], type: :s32)

      assert run(&Nx.argmax(&1, axis: 1, tie_break: :low), [x]) |> Nx.to_flat_list() == [1, 0]
      assert run(&Nx.argmax(&1, axis: 1, tie_break: :high), [x]) |> Nx.to_flat_list() == [4, 4]
      assert run(&Nx.argmin(&1, axis: 1, tie_break: :low), [x]) |> Nx.to_flat_list() == [0, 2]
      assert run(&Nx.argmin(&1, axis: 1, tie_break: :high), [x]) |> Nx.to_flat_list() == [0, 3]
    end

    test "NaN beats every number; of two NaNs the later one wins, as in the reference" do
      x = Nx.tensor([[1.0, :nan, 7.0, :nan, 2.0]], type: :f32)

      for tie <- [:low, :high] do
        for fun <- [&Nx.argmax/2, &Nx.argmin/2] do
          out = run(&fun.(&1, axis: 1, tie_break: tie), [x])
          assert Nx.to_flat_list(out) == [3]
        end
      end
    end
  end

  describe "zero-size inputs, through the wire" do
    setup do
      {:ok, ctx: Native.device_ctx_create(0)}
    end

    defp wire(inputs, instr), do: {inputs, [], [], [instr], [{:result, 0}]}

    defp exec(ctx, wire, buffers) do
      with {:ok, program} <- Native.compile_program(ctx, wire) do
        {:ok, [out]} = Native.eval_program(program, buffers)
        out
      end
    end

    test "an empty output reads back as empty", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)

      cases = [
        {{@f32, {0, 3}}, {:sum, [{:input, 0}], @f32, {0}, [1]}},
        {{@f32, {0}}, {:cumulative_sum, [{:input, 0}], @f32, {0}, [0, 0]}},
        {{@f32, {0}}, {:sort, [{:input, 0}], @f32, {0}, [0, :asc]}},
        {{@f32, {0}}, {:argsort, [{:input, 0}], {:s, 32}, {0}, [0, :desc]}},
        {{@f32, {0}}, {:window_sum, [{:input, 0}], @f32, {0}, [1, 1, 1, 0, 0]}},
        {{@f32, {0, 3}}, {:argmax, [{:input, 0}], {:s, 32}, {0}, [:low, 1]}}
      ]

      for {input, instr} <- cases do
        out = exec(ctx, wire([input], instr), [empty])
        assert Native.buffer_read(out, 0) == <<>>, inspect(instr)
      end
    end

    test "an empty reduced axis fills with the identity of the op", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)
      input = {@f32, {3, 0}}

      cases = [
        {:sum, @f32, <<0, 0, 0, 0>>},
        {:product, @f32, <<0, 0, 128, 63>>},
        {:reduce_max, @f32, <<0, 0, 128, 255>>},
        {:reduce_min, @f32, <<0, 0, 128, 127>>}
      ]

      for {op, type, identity} <- cases do
        out = exec(ctx, wire([input], {op, [{:input, 0}], type, {3}, [1]}), [empty])
        assert Native.buffer_read(out, 12) == :binary.copy(identity, 3), inspect(op)
      end

      # `all` of nothing is true and `any` of nothing is false.
      out = exec(ctx, wire([input], {:all, [{:input, 0}], {:u, 8}, {3}, [1]}), [empty])
      assert Native.buffer_read(out, 3) == <<1, 1, 1>>
      out = exec(ctx, wire([input], {:any, [{:input, 0}], {:u, 8}, {3}, [1]}), [empty])
      assert Native.buffer_read(out, 3) == <<0, 0, 0>>
    end

    test "an arg reduction over an empty axis is an error", %{ctx: ctx} do
      assert {:error, reason} =
               Native.compile_program(
                 ctx,
                 wire([{@f32, {3, 0}}], {:argmax, [{:input, 0}], {:s, 32}, {3}, [:low, 1]})
               )

      assert reason =~ "empty axis"
    end
  end
end
