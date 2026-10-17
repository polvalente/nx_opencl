defmodule NxCL.CompileCacheTest do
  @moduledoc """
  Closure cache, structural cache, options, device registry, queue partitions,
  eviction and a concurrency stress run. Run on one GPU, alone: two suites at
  once flake.
  """
  use ExUnit.Case, async: false

  alias NxCL.CompileCache
  alias NxCL.Devices
  alias NxCL.Native
  alias NxCL.ProgramCache
  alias NxCL.TestSupport.Calls

  @moduletag :opencl

  defp jit(fun, opts \\ []), do: Nx.Defn.jit(fun, [compiler: NxCL.Compiler] ++ opts)

  @lower {NxCL.Lower, :lower, 2}

  defp with_env(module, value) do
    Application.put_env(:nx_opencl, module, value)
    on_exit(fn -> Application.delete_env(:nx_opencl, module) end)
  end

  describe "closure cache" do
    test "a second call of the same function and templates does not trace or lower" do
      calls = :counters.new(1, [])

      fun = fn x ->
        :counters.add(calls, 1, 1)
        Nx.add(Nx.multiply(x, 3), 1)
      end

      x = Nx.iota({4}, type: :f32)

      {first, %{@lower => lowered}} = Calls.count([@lower], fn -> jit(fun).(x) end)
      assert first == Nx.tensor([1.0, 4.0, 7.0, 10.0])
      assert :counters.get(calls, 1) == 1
      assert lowered == 1

      {second, %{@lower => lowered}} = Calls.count([@lower], fn -> jit(fun).(x) end)
      assert second == Nx.tensor([1.0, 4.0, 7.0, 10.0])
      assert :counters.get(calls, 1) == 1
      assert lowered == 0
    end

    test "other templates compile again" do
      calls = :counters.new(1, [])

      fun = fn x ->
        :counters.add(calls, 1, 1)
        Nx.negate(x)
      end

      jit(fun).(Nx.iota({4}, type: :f32))
      jit(fun).(Nx.iota({5}, type: :f32))
      jit(fun).(Nx.iota({4}, type: :s32))
      assert :counters.get(calls, 1) == 3
    end

    test "the device is part of the key" do
      calls = :counters.new(1, [])

      fun = fn x ->
        :counters.add(calls, 1, 1)
        Nx.negate(x)
      end

      x = Nx.iota({4}, type: :f32)
      jit(fun).(x)
      jit(fun, device: 0).(x)
      jit(fun, device: {0, 0}).(x)
      assert :counters.get(calls, 1) == 3
    end
  end

  describe "structural cache" do
    test "two functions that lower alike share one compiled program" do
      x = Nx.iota({6}, type: :f32)
      # A constant no other test uses, so the first call adds an entry.
      c = System.unique_integer([:positive]) / 1000
      # A full cache evicts as it adds, so start empty.
      ProgramCache.clear()
      size = ProgramCache.size()

      {a, %{@lower => lowered_a}} =
        Calls.count([@lower], fn -> jit(fn x -> Nx.exp(Nx.add(x, c)) end).(x) end)

      assert ProgramCache.size() == size + 1

      {b, %{@lower => lowered_b}} =
        Calls.count([@lower], fn -> jit(fn y -> Nx.exp(Nx.add(y, c)) end).(x) end)

      assert a == b
      assert {lowered_a, lowered_b} == {1, 1}
      assert ProgramCache.size() == size + 1
    end

    test "the signature holds no expression ids" do
      expr = fn -> Nx.Defn.debug_expr(fn x -> Nx.sin(x) end).(Nx.iota({3}, type: :f32)) end
      one = NxCL.Lower.lower(expr.(), 1)
      two = NxCL.Lower.lower(expr.(), 1)
      assert NxCL.Program.signature(one) == NxCL.Program.signature(two)
    end

    test "a different shape is a different program" do
      c = System.unique_integer([:positive]) / 1000
      ProgramCache.clear()
      size = ProgramCache.size()
      jit(fn x -> Nx.add(Nx.sin(x), c) end).(Nx.iota({3}, type: :f32))
      jit(fn x -> Nx.add(Nx.sin(x), c) end).(Nx.iota({4}, type: :f32))
      assert ProgramCache.size() == size + 2
    end

    test "the first writer wins a race" do
      key = {:race, make_ref()}
      parent = self()

      winners =
        for i <- 1..8 do
          Task.async(fn ->
            send(parent, {:ready, self()})

            receive do
              :go -> ProgramCache.fetch_or_build(key, fn -> {:built, i} end)
            end
          end)
        end

      pids = for _ <- 1..8, do: receive(do: ({:ready, pid} -> pid))
      Enum.each(pids, &send(&1, :go))

      results = Enum.map(winners, &Task.await/1)
      assert results |> Enum.uniq() |> length() == 1
      assert ProgramCache.fetch_or_build(key, fn -> :never end) == hd(results)
    end

    test "the bound evicts the oldest entry first" do
      with_env(ProgramCache, max_items: 3)
      keys = for i <- 1..5, do: {:bound, make_ref(), i}
      for key <- keys, do: ProgramCache.fetch_or_build(key, fn -> :value end)

      assert ProgramCache.size() <= 3
      [old, newest] = [hd(keys), List.last(keys)]
      assert ProgramCache.fetch_or_build(old, fn -> :rebuilt end) == :rebuilt
      assert ProgramCache.fetch_or_build(newest, fn -> :rebuilt end) == :value
    end
  end

  describe "NxCL.CompileCache" do
    test "max_items evicts the oldest entry first" do
      start_supervised!({CompileCache, name: :cache_fifo, max_items: 2})
      CompileCache.put(:a, 1, :cache_fifo)
      CompileCache.put(:b, 2, :cache_fifo)
      CompileCache.put(:c, 3, :cache_fifo)

      assert CompileCache.fetch(:a, :cache_fifo) == {:error, :cache_miss}
      assert CompileCache.fetch(:b, :cache_fifo) == {:ok, 2}
      assert CompileCache.fetch(:c, :cache_fifo) == {:ok, 3}
      assert CompileCache.size(:cache_fifo) == 2
    end

    test "an entry past its ttl misses on read, then the sweep removes it" do
      start_supervised!({CompileCache, name: :cache_ttl, ttl: 100})
      CompileCache.put(:a, 1, :cache_ttl)
      assert CompileCache.fetch(:a, :cache_ttl) == {:ok, 1}

      # The sweep runs every 50 ms; read before it can run, past the ttl.
      Process.sleep(110)
      assert CompileCache.fetch(:a, :cache_ttl) == {:error, :cache_miss}

      Process.sleep(100)
      assert CompileCache.size(:cache_ttl) == 0
    end

    test "a bad bound raises" do
      Process.flag(:trap_exit, true)

      assert {:error, {%ArgumentError{message: message}, _}} =
               CompileCache.start_link(name: :cache_bad, max_items: 0)

      assert message =~ "max_items"
    end
  end

  describe "options" do
    test "an unknown option raises ArgumentError and names the key" do
      error =
        assert_raise ArgumentError, fn ->
          jit(fn x -> Nx.negate(x) end, devise: 0).(Nx.tensor([1.0]))
        end

      assert error.message =~ ":devise"
    end

    test "a non-empty :hooks raises" do
      error =
        assert_raise ArgumentError, fn ->
          jit(fn x -> Nx.negate(x) end, hooks: %{foo: fn _ -> :ok end}).(Nx.tensor([1.0]))
        end

      assert error.message =~ ":foo"
    end

    test "a bad device raises or errors before the function runs" do
      assert_raise ArgumentError, ~r/:device/, fn ->
        jit(fn x -> Nx.negate(x) end, device: :gpu).(Nx.tensor([1.0]))
      end

      assert_raise NxCL.CompilerError, ~r/out of range/, fn ->
        jit(fn x -> Nx.negate(x) end, device: 999).(Nx.tensor([1.0]))
      end
    end

    test "__to_backend__ passes the device on" do
      assert NxCL.Compiler.__to_backend__(device: 0) == {NxCL.Backend, device: 0}
    end
  end

  describe "device registry" do
    test "one context per device, shared by every caller" do
      assert Devices.ctx() == NxCL.Executor.ctx()
      assert {:ok, %{ctx: ctx, key: {nil, 0}}} = Devices.fetch(device: 0)
      assert ctx == Devices.ctx()
      assert %{key: {nil, 0}} = Devices.for_ctx(ctx)
    end

    test "a context that cannot open gives an error tuple" do
      assert {:error, message} = Devices.fetch(device: 999)
      assert message =~ "out of range"
      assert {:error, message} = Devices.fetch(device: {99, 0})
      assert message =~ "Platform index 99"
    end

    test "persistent_term holds no device context" do
      jit(fn x -> Nx.negate(x) end).(Nx.tensor([1.0]))
      ctx = Devices.ctx()
      entries = :persistent_term.get()

      refute Enum.any?(entries, fn {key, value} -> key == :nxcl_device_ctx or value == ctx end)

      refute Enum.any?(entries, fn {key, _value} ->
               match?({NxCL.Executor, _}, key) or match?({NxCL.Executor, _, _}, key) or
                 match?({NxCL.Executor, _, _, _}, key)
             end)
    end

    test "the backend honors :device" do
      t = Nx.tensor([1.0, 2.0], backend: {NxCL.Backend, device: 0})
      assert t.data.ctx == Devices.ctx()
    end
  end

  describe "partitions" do
    test "max_concurrency: 2 gives two distinct queues" do
      assert [one, two] = NxCL.Compiler.__partitions_options__(max_concurrency: 2)
      assert is_reference(one[:command_queue])
      assert is_reference(two[:command_queue])
      assert one[:command_queue] != two[:command_queue]
      refute Keyword.has_key?(one, :max_concurrency)
    end

    test "without max_concurrency there is one partition" do
      assert [opts] = NxCL.Compiler.__partitions_options__([])
      assert is_reference(opts[:command_queue])
    end

    test "a queue only runs programs of its own context" do
      ctx = Native.device_ctx_create(0)
      other = Native.queue_create(ctx)

      wire =
        {[{{:f, 32}, {2}}], [], [], [{:negate, [{:input, 0}], {:f, 32}, {2}, []}], [{:result, 0}]}

      {:ok, program} = Native.compile_program(Devices.ctx(), wire)
      buf = Native.buffer_write(Devices.ctx(), <<0::64>>)
      assert {:error, message} = Native.eval_program_on(program, [buf], other)
      assert message =~ "another device context"
    end

    test "two processes run on two queues at once" do
      [one, two] = NxCL.Compiler.__partitions_options__(max_concurrency: 2)
      x = Nx.iota({64, 64}, type: :f32) |> Nx.divide(100)
      expected = Nx.dot(x, x)

      run = fn opts ->
        Task.async(fn ->
          fun = fn a -> Nx.dot(a, a) end
          for _ <- 1..20, do: jit(fun, opts).(x)
        end)
      end

      for results <- Task.await_many([run.(one), run.(two)], 60_000),
          result <- results do
        assert Nx.all_close(result, expected, atol: 1.0e-3) |> Nx.to_number() == 1
      end
    end
  end

  describe "eviction and lifetime" do
    test "a closure whose cache entries are gone still runs" do
      with_env(ProgramCache, max_items: 1)
      x = Nx.iota({8}, type: :f32)
      closure = jit(fn x -> Nx.add(Nx.exp(x), 1) end)
      expected = closure.(x)

      # Push the first program out of the cache.
      for n <- 1..4, do: jit(fn x -> Nx.add(x, n) end).(x)

      assert closure.(x) == expected
    end

    test "a cache entry dropped during a run does not free what the run uses" do
      with_env(ProgramCache, max_items: 1)
      x = Nx.iota({256, 256}, type: :f32) |> Nx.divide(1000)
      fun = fn a -> a |> Nx.dot(a) |> Nx.dot(a) |> Nx.dot(a) end
      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x)

      runner = Task.async(fn -> for _ <- 1..10, do: jit(fun).(x) end)
      evictor = Task.async(fn -> for n <- 1..30, do: jit(fn y -> Nx.add(y, n) end).(x) end)

      Task.await(evictor, 60_000)

      for result <- Task.await(runner, 60_000),
          do: assert(Nx.all_close(result, expected, atol: 1.0e-2) |> Nx.to_number() == 1)
    end
  end

  describe "concurrency" do
    @describetag timeout: 300_000

    defp reference_fun(0, x), do: x |> Nx.multiply(2) |> Nx.add(1)
    defp reference_fun(1, x), do: Nx.dot(x, x)
    defp reference_fun(2, x), do: x |> Nx.exp() |> Nx.sum(axes: [1])

    test "8 processes x 100 calls over 3 functions and 2 queues" do
      queues = NxCL.Compiler.__partitions_options__(max_concurrency: 2)
      x = Nx.iota({16, 16}, type: :f32) |> Nx.divide(64)

      funs = [
        fn a -> reference_fun(0, a) end,
        fn a -> reference_fun(1, a) end,
        fn a -> reference_fun(2, a) end
      ]

      expected = for i <- 0..2, do: reference_fun(i, Nx.backend_copy(x, Nx.BinaryBackend))

      tasks =
        for p <- 0..7 do
          opts = Enum.at(queues, rem(p, 2))

          Task.async(fn ->
            for call <- 1..100 do
              i = rem(call + p, 3)
              result = jit(Enum.at(funs, i), opts).(x)
              {i, Nx.all_close(result, Enum.at(expected, i), atol: 1.0e-3) |> Nx.to_number()}
            end
          end)
        end

      results = tasks |> Task.await_many(280_000) |> List.flatten()
      assert length(results) == 800
      assert Enum.all?(results, fn {_i, ok} -> ok == 1 end)
    end
  end
end
