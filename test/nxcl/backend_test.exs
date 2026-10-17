defmodule NxCL.BackendTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  describe "from_binary / to_binary round-trip" do
    test "f32 tensor round-trip" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      bin = Nx.to_binary(t)

      gpu_t = Nx.backend_transfer(t, NxCL.Backend)
      assert %NxCL.Backend{} = gpu_t.data

      result_bin = Nx.to_binary(gpu_t)
      assert result_bin == bin
    end

    test "2D tensor round-trip" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      gpu_t = Nx.backend_transfer(t, NxCL.Backend)
      cpu_t = Nx.backend_transfer(gpu_t, Nx.BinaryBackend)

      assert_tensors_close(cpu_t, t)
    end
  end

  describe "elementwise binary ops via backend" do
    test "add" do
      a = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.add(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([11.0, 22.0, 33.0])

      assert_tensors_close(result, expected)
    end

    test "subtract" do
      a = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.subtract(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([9.0, 18.0, 27.0])

      assert_tensors_close(result, expected)
    end

    test "multiply" do
      a = Nx.tensor([2.0, 3.0, 4.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([5.0, 5.0, 5.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.multiply(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([10.0, 15.0, 20.0])

      assert_tensors_close(result, expected)
    end

    test "divide" do
      a = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([2.0, 4.0, 5.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.divide(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([5.0, 5.0, 6.0])

      assert_tensors_close(result, expected)
    end
  end

  describe "elementwise unary ops via backend" do
    test "negate" do
      t = Nx.tensor([1.0, -2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.negate(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([-1.0, 2.0, -3.0])
      assert_tensors_close(result, expected)
    end

    test "exp" do
      t = Nx.tensor([0.0, 1.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.exp(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.exp(Nx.tensor([0.0, 1.0, 2.0]))
      assert_tensors_close(result, expected, 1.0e-5)
    end

    test "sigmoid" do
      t = Nx.tensor([0.0, 1.0, -1.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.sigmoid(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.sigmoid(Nx.tensor([0.0, 1.0, -1.0]))
      assert_tensors_close(result, expected, 1.0e-5)
    end
  end

  describe "dot (matmul) via backend" do
    test "2D matmul matches CPU" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.dot(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.dot(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))

      assert_tensors_close(result, expected)
    end

    test "non-square matmul" do
      key = Nx.Random.key(42)
      {a_cpu, key} = Nx.Random.uniform(key, shape: {4, 8})
      {b_cpu, _key} = Nx.Random.uniform(key, shape: {8, 6})

      a = Nx.backend_transfer(a_cpu, NxCL.Backend)
      b = Nx.backend_transfer(b_cpu, NxCL.Backend)

      result = Nx.dot(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.dot(a_cpu, b_cpu)

      assert_tensors_close(result, expected, 1.0e-4)
    end
  end

  describe "reductions via backend" do
    test "sum all elements" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.sum(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(10.0)
      assert_tensors_close(result, expected, 1.0e-4)
    end

    test "reduce_max" do
      t = Nx.tensor([3.0, 1.0, 5.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.reduce_max(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(5.0)
      assert_tensors_close(result, expected, 1.0e-5)
    end

    test "reduce_min" do
      t = Nx.tensor([3.0, 1.0, 5.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.reduce_min(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(1.0)
      assert_tensors_close(result, expected, 1.0e-5)
    end
  end

  describe "ops with no CPU copy" do
    test "comparison and sort run as programs on the device" do
      a = Nx.tensor([1.0, 5.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([2.0, 4.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)

      greater = Nx.greater(a, b)
      assert %NxCL.Backend{} = greater.data
      assert Nx.backend_transfer(greater, Nx.BinaryBackend) == Nx.tensor([0, 1, 0], type: :u8)

      sorted = Nx.sort(Nx.tensor([3.0, 1.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend))
      assert %NxCL.Backend{} = sorted.data

      assert_tensors_close(
        Nx.backend_transfer(sorted, Nx.BinaryBackend),
        Nx.tensor([1.0, 2.0, 3.0])
      )
    end

    test "an op with no lowering raises ArgumentError" do
      t = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)

      assert_raise ArgumentError, ~r/^does not yet lower op :exp/, fn ->
        Nx.reduce(t, 0.0, fn a, b -> Nx.add(Nx.exp(a), b) end)
      end
    end

    test "from_pointer and to_pointer raise with a message" do
      t = Nx.tensor([1.0]) |> Nx.backend_transfer(NxCL.Backend)
      assert_raise ArgumentError, ~r/does not support to_pointer/, fn -> Nx.to_pointer(t) end

      assert_raise ArgumentError, ~r/does not support from_pointer/, fn ->
        NxCL.Backend.from_pointer(t, :local, [], 0, 4)
      end
    end

    test "every Nx.Backend callback is implemented" do
      Code.ensure_loaded!(NxCL.Backend)

      missing =
        for {name, arity} <- Nx.Backend.behaviour_info(:callbacks),
            not function_exported?(NxCL.Backend, name, arity),
            do: {name, arity}

      assert missing == []
    end

    test "qr, eigh, svd and lu run as a visible host call" do
      m = Nx.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
      gpu = Nx.backend_transfer(m, NxCL.Backend)

      for {name, fun} <- [
            qr: &Nx.LinAlg.qr/1,
            eigh: &Nx.LinAlg.eigh/1,
            svd: &Nx.LinAlg.svd/1,
            lu: &Nx.LinAlg.lu/1
          ] do
        {result, counts} = NxCL.TestSupport.Trace.run(fn -> fun.(gpu) end)
        assert %{host_calls: 1, host_reads: 1} = counts, "#{name}"

        for leaf <- Nx.Defn.Composite.flatten_list([result]) do
          assert %NxCL.Backend{} = leaf.data
        end
      end
    end
  end

  describe "callbacks that take a function, a constant or another backend's tensor" do
    test "reduce and window_reduce read the body of the function" do
      t = Nx.iota({2, 3}, type: :f32) |> Nx.backend_transfer(NxCL.Backend)

      assert Nx.reduce(t, 10.0, [axes: [1]], fn a, b -> Nx.add(a, b) end) |> Nx.to_flat_list() ==
               [13.0, 22.0]

      assert Nx.window_reduce(t, 0.0, {1, 2}, fn a, b -> Nx.max(a, b) end) |> Nx.to_flat_list() ==
               [1.0, 2.0, 4.0, 5.0]
    end

    test "a constant of a shape, with nan, and a slice with integer starts" do
      c = Nx.broadcast(Nx.tensor(:nan, type: :f32, backend: NxCL.Backend), {2, 2})
      assert Nx.to_flat_list(c) == [:nan, :nan, :nan, :nan]

      t = Nx.iota({4, 4}, type: :s32) |> Nx.backend_transfer(NxCL.Backend)
      assert Nx.slice(t, [1, 2], [2, 2]) |> Nx.to_flat_list() == [6, 7, 10, 11]
    end

    test "a tensor from another backend goes up and the result stays on the device" do
      a = Nx.iota({3}, type: :f32, backend: NxCL.Backend)
      result = Nx.add(a, Nx.tensor([1.0, 1.0, 1.0], backend: Nx.BinaryBackend))
      assert %NxCL.Backend{} = result.data
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end
  end

  describe "memory" do
    test "10 000 adds with backend_deallocate leave live_buffer_count unchanged" do
      a = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0]) |> Nx.backend_transfer(NxCL.Backend)
      ctx = NxCL.Executor.ctx()

      :erlang.garbage_collect()
      before = NxCL.Native.live_buffer_count(ctx)

      for _ <- 1..10_000 do
        assert :ok = a |> Nx.add(b) |> Nx.backend_deallocate()
      end

      assert NxCL.Native.live_buffer_count(ctx) == before
    end

    test "reading a freed tensor fails; a second free reports it" do
      t = Nx.tensor([1.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      assert :ok = Nx.backend_deallocate(t)
      assert :already_deallocated = Nx.backend_deallocate(t)
      assert_raise ErlangError, fn -> Nx.to_binary(t) end
    end

    test "a reshape shares memory; freeing the source keeps the result" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0]) |> Nx.backend_transfer(NxCL.Backend)
      r = Nx.reshape(t, {2, 2})
      ctx = NxCL.Executor.ctx()
      before = NxCL.Native.live_buffer_count(ctx)

      Nx.backend_deallocate(t)
      assert NxCL.Native.live_buffer_count(ctx) == before
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0]

      Nx.backend_deallocate(r)
      assert NxCL.Native.live_buffer_count(ctx) == before - 1
    end

    test "to_binary honours a limit" do
      t = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      assert Nx.to_binary(t, limit: 2) == Nx.to_binary(Nx.tensor([1.0, 2.0]))
    end
  end

  describe "program cache" do
    test "a repeat call with the same types and shapes builds no kernel" do
      a = Nx.iota({64}, type: :f32, backend: NxCL.Backend)
      b = Nx.iota({64}, type: :f32, backend: NxCL.Backend)
      ctx = NxCL.Executor.ctx()

      Nx.add(a, b)
      {builds, _hits, _launches} = NxCL.Native.kernel_cache_stats(ctx)
      Nx.add(a, b)
      assert {^builds, _, _} = NxCL.Native.kernel_cache_stats(ctx)
    end
  end

  describe "to_batched" do
    test "cuts batches and repeats the leftover on request" do
      t = Nx.iota({5, 2}, type: :s32) |> Nx.backend_transfer(NxCL.Backend)
      batches = t |> Nx.to_batched(2, leftover: :repeat) |> Enum.map(&Nx.to_flat_list/1)
      assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]
      assert t |> Nx.to_batched(2, leftover: :discard) |> Enum.count() == 2
    end
  end

  describe "differential test: GPU vs CPU" do
    test "chain of ops matches CPU" do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {4, 4})

      # CPU reference
      cpu_result =
        input
        |> Nx.multiply(Nx.tensor(2.0))
        |> Nx.add(Nx.tensor(1.0))
        |> Nx.negate()

      # GPU computation
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)

      gpu_result =
        gpu_input
        |> Nx.multiply(Nx.backend_transfer(Nx.tensor(2.0), NxCL.Backend))
        |> Nx.add(Nx.backend_transfer(Nx.tensor(1.0), NxCL.Backend))
        |> Nx.negate()
        |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-5)
    end
  end
end
