defmodule NxCL.NativeRuntimeTest do
  @moduledoc """
  Tests of the native program executor: decoding, kernel cache, outputs,
  buffer liveness. They use a fresh device context where a test reads counters.
  """
  use ExUnit.Case, async: false

  alias NxCL.Native
  alias NxCL.Program.Builder

  @moduletag :opencl

  @f32 {:f, 32}

  @add_one """
  __kernel void add_one(__global const float* a, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] + 1.0f;
  }
  """

  defp upload(ctx, floats),
    do: Native.buffer_write(ctx, NxCL.TestHelpers.floats_to_binary(floats))

  defp read(buf, n), do: buf |> Native.buffer_read(n * 4) |> NxCL.TestHelpers.binary_to_floats()

  # `shape` is the shape of every input; the tests use f32 vectors.
  defp wire(instrs, outputs, n_inputs \\ 1, shape \\ {3}),
    do: {List.duplicate({@f32, shape}, n_inputs), [], [], instrs, outputs}

  defp instr(op, operands, shape, attrs \\ []), do: {op, operands, @f32, shape, attrs}

  defp run!(ctx, wire, buffers) do
    {:ok, program} = Native.compile_program(ctx, wire)
    {:ok, outputs} = Native.eval_program(program, buffers)
    outputs
  end

  defp wait_for(fun, tries \\ 50) do
    :erlang.garbage_collect()

    cond do
      fun.() -> :ok
      tries == 0 -> flunk("condition did not hold after garbage collection")
      true -> Process.sleep(20) && wait_for(fun, tries - 1)
    end
  end

  describe "outputs" do
    test "two outputs come back in order" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0, 3.0])

      instrs = [instr(:negate, [{:input, 0}], {3}), instr(:add, [{:input, 0}, {:result, 0}], {3})]
      [a, b] = run!(ctx, wire(instrs, [{:result, 0}, {:result, 1}]), [x])

      assert read(a, 3) == [-1.0, -2.0, -3.0]
      assert read(b, 3) == [0.0, 0.0, 0.0]
    end

    test "an input returned as output comes back unchanged" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [4.0, 5.0])

      [out] = run!(ctx, wire([], [{:input, 0}], 1, {2}), [x])
      assert read(out, 2) == [4.0, 5.0]
      assert read(x, 2) == [4.0, 5.0]
    end

    test "the same ref as output twice, and an alias as output" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0])

      instrs = [
        instr(:negate, [{:input, 0}], {2}),
        instr(:reshape, [{:result, 0}], {2}, [:alias])
      ]

      outs = run!(ctx, wire(instrs, [{:result, 0}, {:result, 0}, {:result, 1}], 1, {2}), [x])

      assert Enum.map(outs, &read(&1, 2)) == List.duplicate([-1.0, -2.0], 3)
    end

    test "an unread input may be nil" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0])

      instrs = [instr(:exp, [{:input, 1}], {1})]
      [out] = run!(ctx, wire(instrs, [{:result, 0}], 2, {1}), [nil, x])
      assert_in_delta hd(read(out, 1)), :math.exp(1.0), 1.0e-5
    end

    test "a generic launch runs caller source" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0, 3.0])

      launch =
        Builder.launch([{:input, 0}], @f32, {3}, @add_one, "add_one",
          global: [64],
          local: [64],
          scalars: [3]
        )

      encoded = {launch.op, launch.operands, @f32, {3}, launch.attrs}

      [out] = run!(ctx, wire([encoded], [{:result, 0}]), [x])
      assert read(out, 3) == [2.0, 3.0, 4.0]
    end
  end

  describe "bad programs" do
    setup do
      ctx = Native.device_ctx_create(0)
      %{ctx: ctx}
    end

    test "return an error term and leave the VM up", %{ctx: ctx} do
      i1 = fn op, ops, attrs -> {op, ops, @f32, {1}, attrs} end
      prog = fn instrs, outs -> {[{@f32, {1}}], [], [], instrs, outs} end
      out0 = [{:result, 0}]

      bad = [
        :not_a_program,
        prog.([i1.(:frobnicate, [{:input, 0}], [])], out0),
        prog.([i1.(:negate, [{:input, 5}], [])], out0),
        prog.([i1.(:negate, [{:result, 0}], [])], out0),
        prog.([i1.(:negate, [{:carry, 0}], [])], out0),
        prog.([i1.(:negate, [{:weird, 0}], [])], out0),
        prog.([i1.(:negate, [{:input, 0}], [])], [{:result, 3}]),
        prog.([i1.(:add, [{:input, 0}], [])], out0),
        prog.([i1.(:negate, :oops, [])], out0),
        prog.([i1.(:launch, [{:input, 0}], [<<"not C">>, "k", "", 1, 1, 0])], out0),
        prog.([i1.(:launch, [{:input, 0}], [1, 2])], out0),
        # a type the table does not have, an unknown kind, a bad shape, a bad constant
        {[{{:s, 4}, {1}}], [], [], [i1.(:negate, [{:input, 0}], [])], [{:result, 0}]},
        {[{{:q, 8}, {1}}], [], [], [], []},
        {[{@f32, :not_a_shape}], [], [], [], []},
        {[], [], [{@f32, :oops}], [], []},
        {[{@f32, {1}}], [], [], [{:negate, [{:input, 0}], {:f, 32}, {1}}], [{:result, 0}]},
        # operands of different shapes need broadcast
        {[{@f32, {2}}, {@f32, {3}}], [], [], [{:add, [{:input, 0}, {:input, 1}], @f32, {3}, []}],
         [{:result, 0}]},
        # f64 may not exist on the device; an f32 op fed an op that has no f32 form
        {[{@f32, {1}}], [], [], [{:divide, [{:input, 0}, {:input, 0}], {:s, 32}, {1}, []}],
         [{:result, 0}]}
      ]

      for wire <- bad do
        assert {:error, reason} = Native.compile_program(ctx, wire), inspect(wire)
        assert is_binary(reason)
      end

      x = upload(ctx, [1.0])
      [out] = run!(ctx, wire([instr(:negate, [{:input, 0}], {1})], [{:result, 0}], 1, {1}), [x])
      assert read(out, 1) == [-1.0]
    end

    test "eval rejects the wrong buffer count and a missing input", %{ctx: ctx} do
      {:ok, program} =
        Native.compile_program(
          ctx,
          wire([instr(:negate, [{:input, 0}], {1})], [{:result, 0}], 1, {1})
        )

      assert {:error, _} = Native.eval_program(program, [])
      assert {:error, _} = Native.eval_program(program, [nil])
    end
  end

  describe "kernel cache" do
    test "builds each distinct kernel once" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0])

      instrs = [
        instr(:add, [{:input, 0}, {:input, 0}], {2}),
        instr(:exp, [{:result, 0}], {2}),
        instr(:add, [{:result, 1}, {:input, 0}], {2})
      ]

      w = wire(instrs, [{:result, 2}], 1, {2})
      {:ok, program} = Native.compile_program(ctx, w)
      assert {2, 1, 0} = Native.kernel_cache_stats(ctx)

      for _ <- 1..100, do: {:ok, [_]} = Native.eval_program(program, [x])
      assert {2, 1, 300} = Native.kernel_cache_stats(ctx)

      {:ok, _} = Native.compile_program(ctx, w)
      assert {2, 4, 300} = Native.kernel_cache_stats(ctx)
    end

    test "a jitted function builds nothing after its first call" do
      jitted =
        Nx.Defn.jit(fn x -> x |> Nx.tanh() |> Nx.multiply(x) |> Nx.tanh() end,
          compiler: NxCL.Compiler
        )

      x = Nx.tensor([0.5, 1.0, 1.5])
      ctx = NxCL.Executor.ctx()

      jitted.(x)
      {builds, hits, launches} = Native.kernel_cache_stats(ctx)

      for _ <- 1..99, do: jitted.(x)
      assert {^builds, ^hits, after_launches} = Native.kernel_cache_stats(ctx)
      assert after_launches - launches == 99 * 3
    end
  end

  describe "buffer liveness" do
    test "10 000 evals leave the live buffer count where it started" do
      ctx = Native.device_ctx_create(0)
      start = Native.live_buffer_count(ctx)
      x = upload(ctx, [1.0, 2.0, 3.0, 4.0])

      instrs = [
        instr(:negate, [{:input, 0}], {4}),
        instr(:reshape, [{:result, 0}], {4}, [:alias]),
        instr(:add, [{:result, 1}, {:input, 0}], {4}),
        instr(:exp, [{:result, 2}], {4})
      ]

      {:ok, program} =
        Native.compile_program(
          ctx,
          wire(instrs, [{:result, 3}, {:result, 0}, {:input, 0}], 1, {4})
        )

      for _ <- 1..10_000 do
        {:ok, [_, _, _]} = Native.eval_program(program, [x])
      end

      wait_for(fn -> Native.live_buffer_count(ctx) == start + 1 end)
      assert read(x, 4) == [1.0, 2.0, 3.0, 4.0]
    end

    test "the program frees intermediates before it returns" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0])
      start = Native.live_buffer_count(ctx)

      instrs =
        for i <- 0..9,
            do: instr(:negate, [if(i == 0, do: {:input, 0}, else: {:result, i - 1})], {2})

      {:ok, program} = Native.compile_program(ctx, wire(instrs, [{:result, 9}], 1, {2}))
      {:ok, [out]} = Native.eval_program(program, [x])

      assert Native.live_buffer_count(ctx) == start + 1
      assert read(out, 2) == [1.0, 2.0]
    end
  end
end
