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

  defp wire(instrs, outputs, n_inputs \\ 1), do: {n_inputs, 0, 0, instrs, outputs}

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

      instrs = [{:negate, [{:input, 0}], 12, []}, {:add, [{:input, 0}, {:result, 0}], 12, []}]
      [a, b] = run!(ctx, wire(instrs, [{:result, 0}, {:result, 1}]), [x])

      assert read(a, 3) == [-1.0, -2.0, -3.0]
      assert read(b, 3) == [0.0, 0.0, 0.0]
    end

    test "an input returned as output comes back unchanged" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [4.0, 5.0])

      [out] = run!(ctx, wire([], [{:input, 0}]), [x])
      assert read(out, 2) == [4.0, 5.0]
      assert read(x, 2) == [4.0, 5.0]
    end

    test "the same ref as output twice, and an alias as output" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0])

      instrs = [{:negate, [{:input, 0}], 8, []}, {:reshape, [{:result, 0}], 8, [:alias]}]
      outs = run!(ctx, wire(instrs, [{:result, 0}, {:result, 0}, {:result, 1}]), [x])

      assert Enum.map(outs, &read(&1, 2)) == List.duplicate([-1.0, -2.0], 3)
    end

    test "an unread input may be nil" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0])

      instrs = [{:exp, [{:input, 1}], 4, []}]
      [out] = run!(ctx, wire(instrs, [{:result, 0}], 2), [nil, x])
      assert_in_delta hd(read(out, 1)), :math.exp(1.0), 1.0e-5
    end

    test "a generic launch runs caller source" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0, 3.0])

      instr =
        Builder.launch([{:input, 0}], @f32, {3}, @add_one, "add_one",
          global: [64],
          local: [64],
          scalars: [3]
        )

      encoded = {instr.op, instr.operands, 12, instr.attrs}

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
      bad = [
        :not_a_program,
        {1, 0, 0, [{:frobnicate, [{:input, 0}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, [{:input, 5}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, [{:result, 0}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, [{:carry, 0}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, [{:weird, 0}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, [{:input, 0}], 4, []}], [{:result, 3}]},
        {1, 0, 0, [{:add, [{:input, 0}], 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:negate, :oops, 4, []}], [{:result, 0}]},
        {1, 0, 0, [{:launch, [{:input, 0}], 4, [<<"not C">>, "k", "", 1, 1, 0]}], [{:result, 0}]},
        {1, 0, 0, [{:launch, [{:input, 0}], 4, [1, 2]}], [{:result, 0}]}
      ]

      for wire <- bad do
        assert {:error, reason} = Native.compile_program(ctx, wire), inspect(wire)
        assert is_binary(reason)
      end

      x = upload(ctx, [1.0])
      [out] = run!(ctx, wire([{:negate, [{:input, 0}], 4, []}], [{:result, 0}]), [x])
      assert read(out, 1) == [-1.0]
    end

    test "eval rejects the wrong buffer count and a missing input", %{ctx: ctx} do
      {:ok, program} =
        Native.compile_program(ctx, wire([{:negate, [{:input, 0}], 4, []}], [{:result, 0}]))

      assert {:error, _} = Native.eval_program(program, [])
      assert {:error, _} = Native.eval_program(program, [nil])
    end
  end

  describe "kernel cache" do
    test "builds each distinct kernel once" do
      ctx = Native.device_ctx_create(0)
      x = upload(ctx, [1.0, 2.0])

      instrs = [
        {:add, [{:input, 0}, {:input, 0}], 8, []},
        {:exp, [{:result, 0}], 8, []},
        {:add, [{:result, 1}, {:input, 0}], 8, []}
      ]

      {:ok, program} = Native.compile_program(ctx, wire(instrs, [{:result, 2}]))
      assert {2, 1, 0} = Native.kernel_cache_stats(ctx)

      for _ <- 1..100, do: {:ok, [_]} = Native.eval_program(program, [x])
      assert {2, 1, 300} = Native.kernel_cache_stats(ctx)

      {:ok, _} = Native.compile_program(ctx, wire(instrs, [{:result, 2}]))
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
        {:negate, [{:input, 0}], 16, []},
        {:reshape, [{:result, 0}], 16, [:alias]},
        {:add, [{:result, 1}, {:input, 0}], 16, []},
        {:exp, [{:result, 2}], 16, []}
      ]

      {:ok, program} =
        Native.compile_program(ctx, wire(instrs, [{:result, 3}, {:result, 0}, {:input, 0}]))

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
        for i <- 0..9, do: {:negate, [if(i == 0, do: {:input, 0}, else: {:result, i - 1})], 8, []}

      {:ok, program} = Native.compile_program(ctx, wire(instrs, [{:result, 9}]))
      {:ok, [out]} = Native.eval_program(program, [x])

      assert Native.live_buffer_count(ctx) == start + 1
      assert read(out, 2) == [1.0, 2.0]
    end
  end
end
