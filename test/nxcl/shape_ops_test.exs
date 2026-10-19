defmodule NxCL.ShapeOpsTest do
  @moduledoc """
  Lowering and runtime behaviour of the shape ops that the differential rows
  cannot show: which instructions lowering emits, the values that reshape,
  squeeze, metadata and elem pass on, zero-size inputs (Nx 1.0 cannot build them,
  so they go through the native wire), and the rank cap.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport, only: [lower: 2, ops: 1]
  import NxCL.TestSupport.Compare

  alias NxCL.Compiler.Instruction
  alias NxCL.Native

  @moduletag :opencl

  @f32 {:f, 32}

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  describe "lowering" do
    test "each movement op lowers to one instruction with semantic attrs" do
      x = Nx.iota({2, 3}, type: :f32)

      assert [%Instruction{op: :transpose, attrs: [1, 0], out_shape: {3, 2}}] =
               lower(&Nx.transpose/1, [x]).instructions

      assert [%Instruction{op: :reverse, attrs: [1]}] =
               lower(&Nx.reverse(&1, axes: [1]), [x]).instructions

      assert [%Instruction{op: :broadcast, attrs: [1, 2], out_shape: {2, 3, 4}}] =
               lower(&Nx.broadcast(&1, {2, 3, 4}, axes: [1, 2]), [Nx.iota({3, 1})]).instructions

      assert [%Instruction{op: :pad, attrs: [1, 2, 0, -1, 0, 3]}] =
               lower(&Nx.pad(&1, 0.0, [{1, 2, 0}, {-1, 0, 3}]), [x]).instructions

      assert [%Instruction{op: :concatenate, attrs: [1], operands: [_, _]}] =
               lower(&Nx.concatenate([&1, &1], axis: 1), [x]).instructions

      assert [%Instruction{op: :iota, attrs: [nil], operands: []}] =
               lower(fn -> Nx.iota({2, 3}) end, []).instructions

      assert [%Instruction{op: :eye, operands: []}] =
               lower(fn -> Nx.eye({3, 3}) end, []).instructions
    end

    test "slice starts are operands: static integers become s64 constants, tensors stay refs" do
      program = lower(fn x, i -> Nx.slice(x, [i, 1], [1, 2]) end, [Nx.iota({3, 4}), Nx.tensor(1)])

      assert [%Instruction{op: :slice, operands: [{:input, 0}, {:input, 1}, {:const, 0}]}] =
               program.instructions

      # Nx makes the integer next to a tensor start an `s32` constant.
      assert [{1, {:s, 32}}] = program.constants

      program = lower(&Nx.slice(&1, [1, 0], [1, 2]), [Nx.iota({3, 4})])

      assert [%Instruction{op: :slice, operands: [{:input, 0}, {:const, 0}, {:const, 1}]}] =
               program.instructions

      assert [{1, {:s, 64}}, {0, {:s, 64}}] = program.constants
    end

    test "a scalar tensor or a lower-rank tensor next to a tensor gets a broadcast" do
      program = lower(fn x, s -> Nx.add(x, s) end, [Nx.iota({2}, type: :f32), Nx.tensor(1.0)])

      assert [%Instruction{op: :broadcast, attrs: [], out_shape: {2}}, %Instruction{op: :add}] =
               program.instructions

      program = lower(fn x, y -> Nx.add(x, y) end, [Nx.iota({2, 3}), Nx.iota({3})])

      assert [%Instruction{op: :broadcast, attrs: [1], out_shape: {2, 3}}, _] =
               program.instructions
    end

    test "a constant of a shape lowers to a broadcast of a scalar constant" do
      program = lower(fn -> Nx.broadcast(Nx.tensor(2.0), {2, 2}) end, [])

      assert [%Instruction{op: :broadcast, operands: [{:const, 0}], out_shape: {2, 2}}] =
               program.instructions

      assert [{2.0, @f32}] = program.constants
    end

    test "metadata and elem emit no instruction" do
      program =
        lower(
          fn x ->
            {a, b} = Nx.Defn.Kernel.stop_grad({x, Nx.negate(x)})
            Nx.add(a, b)
          end,
          [Nx.iota({2}, type: :f32)]
        )

      assert [:negate, :add] = ops(program)
    end

    test "operands of another type are cast with as_type" do
      program =
        lower(&Nx.pad(&1, 0.5, [{1, 0, 0}]), [Nx.iota({3}, type: :s32)])

      assert [:as_type, :pad] = ops(program)
    end
  end

  describe "reshape, squeeze, metadata and elem" do
    test "give the input's values" do
      x = Nx.iota({6}, type: :f32)

      out =
        run(
          fn x ->
            {a, b} = Nx.Defn.Kernel.stop_grad({x, x})
            y = a |> Nx.reshape({1, 6, 1}) |> Nx.squeeze() |> Nx.reshape({2, 3})
            {y, b}
          end,
          [x]
        )

      assert {y, b} = out
      assert y == Nx.reshape(x, {2, 3})
      assert b == x
    end

    test "a reshaped value can feed later ops" do
      x = Nx.iota({6}, type: :f32)

      assert_same(
        fn x ->
          y = Nx.reshape(x, {2, 3})
          {y, Nx.add(y, Nx.reverse(y))}
        end,
        [x],
        backend: false
      )
    end
  end

  describe "zero-size inputs, through the wire" do
    # Nx 1.0 has no zero-size tensors, so each program is written as a wire term.
    setup do
      ctx = Native.device_ctx_create(0)
      {:ok, ctx: ctx}
    end

    defp wire(inputs, consts, instr), do: {inputs, [], consts, [instr], [{:result, 0}]}

    defp exec(ctx, wire, buffers) do
      {:ok, program} = Native.compile_program(ctx, wire)
      {:ok, [out]} = Native.eval_program(program, buffers)
      out
    end

    defp floats(buf, n),
      do: buf |> Native.buffer_read(n * 4) |> NxCL.TestHelpers.binary_to_floats()

    test "an empty output reads back as empty", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)
      s64 = {:s, 64}

      cases = [
        {[{@f32, {0}}], [], {:reverse, [{:input, 0}], @f32, {0}, [0]}},
        {[{@f32, {0, 3}}], [], {:transpose, [{:input, 0}], @f32, {3, 0}, [1, 0]}},
        {[{@f32, {0}}], [], {:broadcast, [{:input, 0}], @f32, {2, 0}, [1]}},
        {[{@f32, {4}}], [{s64, 1}], {:slice, [{:input, 0}, {:const, 0}], @f32, {0}, [0, 1]}},
        {[], [], {:iota, [], @f32, {0}, [0]}},
        {[], [], {:eye, [], @f32, {0, 3}, []}},
        {[{@f32, {0}}], [], {:negate, [{:input, 0}], @f32, {0}, []}}
      ]

      for {inputs, consts, instr} <- cases do
        buffers =
          Enum.map(inputs, fn {_t, shape} ->
            if elem(shape, 0) == 0, do: empty, else: Native.buffer_write(ctx, <<0::128>>)
          end) ++ Enum.map(consts, fn _ -> nil end)

        out = exec(ctx, wire(inputs, consts, instr), buffers)
        assert Native.buffer_read(out, 0) == <<>>, inspect(instr)
      end
    end

    test "an empty input beside a non-empty output still fills", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)
      full = Native.buffer_write(ctx, NxCL.TestHelpers.floats_to_binary([1.0, 2.0]))

      # concatenate of {0} and {2}
      out =
        exec(
          ctx,
          wire(
            [{@f32, {0}}, {@f32, {2}}],
            [],
            {:concatenate, [{:input, 0}, {:input, 1}], @f32, {2}, [0]}
          ),
          [empty, full]
        )

      assert floats(out, 2) == [1.0, 2.0]

      # pad of {0} by 1 on each side: both elements are the pad value
      out =
        exec(
          ctx,
          wire(
            [{@f32, {0}}],
            [{@f32, 7.0}],
            {:pad, [{:input, 0}, {:const, 0}], @f32, {2}, [1, 1, 0]}
          ),
          [empty, nil]
        )

      assert floats(out, 2) == [7.0, 7.0]

      # put_slice of an empty update leaves the tensor as it is
      out =
        exec(
          ctx,
          wire(
            [{@f32, {2}}, {@f32, {0}}],
            [{{:s, 64}, 1}],
            {:put_slice, [{:input, 0}, {:input, 1}, {:const, 0}], @f32, {2}, []}
          ),
          [full, empty, nil]
        )

      assert floats(out, 2) == [1.0, 2.0]
    end
  end

  describe "rank cap" do
    test "rank 9 returns an error from the decoder, rank 8 compiles" do
      ctx = Native.device_ctx_create(0)

      wire = fn rank ->
        shape = List.to_tuple(List.duplicate(1, rank))
        axes = Enum.to_list(0..(rank - 1))

        {[{@f32, shape}], [], [], [{:transpose, [{:input, 0}], @f32, shape, axes}],
         [{:result, 0}]}
      end

      assert {:error, reason} = Native.compile_program(ctx, wire.(9))
      assert reason =~ "rank 9"
      assert {:ok, _} = Native.compile_program(ctx, wire.(8))
    end
  end

  describe "bias-add layer" do
    test "dot, bias add through an explicit broadcast, and sigmoid" do
      x = Nx.iota({4, 5}, type: :f32) |> Nx.divide(10)
      w = Nx.iota({5, 3}, type: :f32) |> Nx.divide(20)
      b = Nx.tensor([0.1, -0.2, 0.3])

      layer = fn x, w, b ->
        x |> Nx.dot(w) |> Nx.add(Nx.broadcast(b, {4, 3})) |> Nx.sigmoid()
      end

      assert ops(lower(layer, [x, w, b])) ==
               [:reshape, :reshape, :batched_gemm, :reshape, :broadcast, :add, :sigmoid]

      assert_same(layer, [x, w, b], rtol: 1.0e-4, atol: 1.0e-4)
    end
  end
end
