defmodule NxCL.FusionTest do
  @moduledoc """
  The fusion pass on its own (which instructions group, where it stops, the
  option), then fused programs on the device: launch counts, the argument
  limit, and the kernel cache.
  """
  use ExUnit.Case, async: false
  import Nx.Defn

  alias NxCL.Executor
  alias NxCL.Fusion
  alias NxCL.Lower
  alias NxCL.Native
  alias NxCL.Program
  alias NxCL.Program.Instruction

  defp lower(fun, args) do
    flat = args |> Nx.Defn.Composite.flatten_list() |> length()
    fun |> Nx.Defn.debug_expr_apply(args) |> Lower.lower(flat)
  end

  defp ops(%Program{instructions: instructions}), do: Enum.map(instructions, & &1.op)

  defp fused(%Program{instructions: instructions}),
    do: Enum.filter(instructions, &(&1.op == :fused))

  defp steps(%Instruction{attrs: [_maps, steps]}), do: Enum.map(steps, &elem(&1, 0))

  defp chain(x, n) do
    Enum.reduce(1..n, x, fn k, acc -> if rem(k, 2) == 0, do: Nx.exp(acc), else: Nx.negate(acc) end)
  end

  @opts [max_args: 512]

  describe "grouping" do
    test "a 20-op chain becomes one fused instruction of 20 steps" do
      program = lower(&chain(&1, 20), [Nx.iota({8}, type: :f32)])
      assert length(program.instructions) == 20

      fused = Fusion.fuse(program, @opts)

      assert [%Instruction{op: :fused, operands: [{:input, 0}], out_shape: {8}} = instr] =
               fused.instructions

      assert length(steps(instr)) == 20
      assert fused.outputs == [{:result, 0}]
    end

    test "fusion: false returns the program unchanged" do
      program = lower(&chain(&1, 20), [Nx.iota({8}, type: :f32)])
      assert Fusion.fuse(program, fusion: false) == program
    end

    test "a value another instruction reads stays its own result" do
      program =
        lower(
          fn x ->
            a = Nx.exp(x)
            {a, Nx.negate(Nx.sin(a))}
          end,
          [Nx.iota({8}, type: :f32)]
        )

      fused = Fusion.fuse(program, @opts)
      assert ops(fused) == [:exp, :fused]
      assert [%Instruction{operands: [{:result, 0}]} = instr] = fused(fused)
      assert steps(instr) == [:sin, :negate]
      assert fused.outputs == [{:result, 0}, {:result, 1}]
    end

    test "a value read twice inside the group joins it" do
      program = lower(fn x -> Nx.multiply(Nx.exp(x), Nx.exp(x)) end, [Nx.iota({8}, type: :f32)])
      program = Fusion.fuse(program, @opts)
      assert [%Instruction{op: :fused, operands: [{:input, 0}]}] = program.instructions
    end

    test "a broadcast input is read through an index mapping, not a step" do
      program =
        lower(
          fn x, b -> Nx.add(Nx.exp(x), b) end,
          [Nx.iota({2, 3}, type: :f32), Nx.iota({3}, type: :f32)]
        )

      assert :broadcast in ops(program)
      fused = Fusion.fuse(program, @opts)

      assert [%Instruction{op: :fused, operands: [{:input, 0}, {:input, 1}]} = instr] =
               fused.instructions

      assert instr.attrs |> hd() == [:direct, {:broadcast, [1]}]
      assert steps(instr) == [:exp, :add]
    end

    test "a scalar constant is an operand of the group" do
      program = lower(fn x -> Nx.add(Nx.exp(x), 1.0) end, [Nx.iota({8}, type: :f32)])
      fused = Fusion.fuse(program, @opts)
      assert [%Instruction{op: :fused, operands: [{:input, 0}, {:const, 0}]}] = fused.instructions
    end

    test "a different shape starts a new group" do
      program =
        lower(
          fn x ->
            x |> Nx.exp() |> Nx.negate() |> Nx.sum(axes: [1]) |> Nx.exp() |> Nx.negate()
          end,
          [Nx.iota({2, 3}, type: :f32)]
        )

      fused = Fusion.fuse(program, @opts)
      assert [:fused, :sum, :fused] = ops(fused)
    end

    test "an instruction that is not elementwise is left alone" do
      program =
        lower(fn x -> x |> Nx.exp() |> Nx.transpose() |> Nx.negate() end, [
          Nx.iota({2, 3}, type: :f32)
        ])

      assert ops(Fusion.fuse(program, @opts)) == [:exp, :transpose, :negate]
    end

    test "groups split at the argument limit" do
      args = for _ <- 1..5, do: Nx.iota({4}, type: :f32)
      sum = fn a, b, c, d, e -> a |> Nx.add(b) |> Nx.add(c) |> Nx.add(d) |> Nx.add(e) end
      program = lower(sum, args)

      assert [instr] = Fusion.fuse(program, max_args: 6).instructions
      assert length(instr.operands) == 5

      split = Fusion.fuse(program, max_args: 4)
      assert length(split.instructions) > 1
      assert Enum.all?(fused(split), &(length(&1.operands) + 1 <= 4))

      assert [%Instruction{op: :fused, operands: operands}] =
               Fusion.fuse(program, max_args: 3, split: false).instructions

      assert length(operands) == 5
    end
  end

  describe "control flow and host calls" do
    defn loop_body(x) do
      {_i, y} =
        while {i = 0, y = x}, i < 3 do
          {i + 1, Nx.negate(Nx.exp(y))}
        end

      Nx.sin(Nx.cos(y))
    end

    defn branch(p, x) do
      cond do
        p > 0 -> Nx.negate(Nx.exp(x))
        true -> Nx.sin(Nx.cos(x))
      end
    end

    test "a while keeps its place; its body and the code after it fuse on their own" do
      program = lower(&loop_body/1, [Nx.iota({4}, type: :f32)])
      fused = Fusion.fuse(program, @opts)

      assert :while in ops(fused)
      [while] = Enum.filter(fused.instructions, &(&1.op == :while))
      [_cond, body] = while.subprograms
      body_ops = Enum.map(body.instructions, & &1.op)
      assert :fused in body_ops
      refute :exp in body_ops
      assert [%Instruction{op: :fused}] = fused(fused)
      assert ops(fused) |> Enum.reverse() |> hd() == :fused
    end

    test "a cond keeps its place; its clauses fuse on their own" do
      program = lower(&branch/2, [Nx.tensor(1), Nx.iota({4}, type: :f32)])
      fused = Fusion.fuse(program, @opts)
      [cond_instr] = Enum.filter(fused.instructions, &(&1.op == :cond))

      bodies = Enum.flat_map(cond_instr.subprograms, &Enum.map(&1.instructions, fn i -> i.op end))
      assert :fused in bodies
      refute :exp in bodies
    end

    test "a host call splits the groups around it" do
      program =
        lower(
          fn x ->
            y = x |> Nx.exp() |> Nx.negate() |> Nx.io_call(fn _ -> :ok end)
            y |> Nx.sin() |> Nx.cos()
          end,
          [Nx.iota({4}, type: :f32)]
        )

      fused = Fusion.fuse(program, @opts)
      assert ops(fused) == [:fused, :host_call, :fused]
    end
  end

  describe "on the device" do
    @describetag :opencl

    defp launches() do
      {_builds, _hits, launches} = Native.kernel_cache_stats(Executor.ctx())
      launches
    end

    defp run_counting(fun, args, opts) do
      jitted = Nx.Defn.jit(fun, [compiler: NxCL.Compiler] ++ opts)
      before = launches()
      result = apply(jitted, args)
      {result, launches() - before}
    end

    test "a 20-op chain runs one kernel with fusion on and 20 with it off" do
      x = Nx.iota({100}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
      {on, one} = run_counting(&chain(&1, 20), [x], fusion: true)
      {off, twenty} = run_counting(&chain(&1, 20), [x], fusion: false)
      assert {one, twenty} == {1, 20}
      assert Nx.all_close(on, off, rtol: 1.0e-5, atol: 1.0e-5) |> Nx.to_number() == 1
    end

    test "an input a later instruction still reads stays intact after the fused kernel" do
      fun = fn x ->
        a = x |> Nx.exp() |> Nx.negate()
        {a, Nx.reduce_max(x), a |> Nx.sin() |> Nx.cos() |> Nx.add(x)}
      end

      x = Nx.iota({6}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(6)
      expected = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), [x])
      actual = apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler, fusion: true), [x])

      for {e, a} <- Enum.zip(Tuple.to_list(expected), Tuple.to_list(actual)) do
        assert Nx.all_close(e, a, rtol: 1.0e-5, atol: 1.0e-5) |> Nx.to_number() == 1
      end
    end

    test "a group over the device argument limit splits and still runs" do
      limit = Fusion.device_max_args()
      n = limit + 1
      args = for k <- 1..n, do: Nx.tensor([k, k + 1], type: :f32)
      sum = fn xs -> xs |> Tuple.to_list() |> Enum.reduce(&Nx.add/2) end
      program = lower(sum, [List.to_tuple(args)])

      assert length(Fusion.fuse(program).instructions) > 1
      lazy = Enum.map(args, fn t -> fn -> t end end)
      result = Executor.run(Fusion.fuse(program), lazy)
      assert Nx.to_flat_list(result) == [Enum.sum(1..n) * 1.0, Enum.sum(2..(n + 1)) * 1.0]
    end

    test "a group sent unsplit gets the Rust limit error" do
      n = Fusion.device_max_args() + 1
      args = for k <- 1..n, do: Nx.tensor([k], type: :f32)
      sum = fn xs -> xs |> Tuple.to_list() |> Enum.reduce(&Nx.add/2) end
      program = lower(sum, [List.to_tuple(args)])

      unsplit = Fusion.fuse(program, split: false)
      assert [%Instruction{op: :fused}] = unsplit.instructions
      lazy = Enum.map(args, fn t -> fn -> t end end)

      error = assert_raise NxCL.CompilerError, fn -> Executor.run(unsplit, lazy) end
      assert error.message =~ "argument limit"
      refute error.message =~ "Failed to compile"
    end

    test "the second call finds the fused source in the kernel cache" do
      # A constant no other test uses gives this chain its own source.
      constant = 1.0 + :erlang.unique_integer([:positive]) / 1.0e9
      fun = fn x -> x |> Nx.multiply(constant) |> Nx.exp() |> Nx.negate() end
      ctx = Executor.ctx()

      first = Nx.iota({5}, type: :f32, backend: Nx.BinaryBackend)
      second = Nx.iota({9}, type: :f32, backend: Nx.BinaryBackend)

      {builds0, hits0, _} = Native.kernel_cache_stats(ctx)
      Nx.Defn.jit(fun, compiler: NxCL.Compiler, fusion: true).(first)
      {builds1, hits1, _} = Native.kernel_cache_stats(ctx)
      # A new shape is a new program, so it is decoded again; its fused source is the same.
      Nx.Defn.jit(fun, compiler: NxCL.Compiler, fusion: true).(second)
      {builds2, hits2, _} = Native.kernel_cache_stats(ctx)

      assert builds1 - builds0 == 1
      assert hits1 == hits0
      assert builds2 == builds1
      assert hits2 - hits1 == 1
    end
  end
end
