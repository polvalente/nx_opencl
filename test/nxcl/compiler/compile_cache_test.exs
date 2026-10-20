defmodule NxCL.Compiler.CompileCacheTest do
  @moduledoc """
  The closure cache, the structural cache, options, the device registry, queue
  partitions, eviction and a concurrent stress run.
  """
  use ExUnit.Case, async: false

  alias NxCL.Compiler.CompileCache
  alias NxCL.Compiler.ProgramCache
  alias NxCL.Devices
  alias NxCL.Native
  alias NxCL.TestSupport
  alias NxCL.TestSupport.Calls

  @moduletag :opencl

  defp jit(fun, opts \\ []), do: Nx.Defn.jit(fun, [compiler: NxCL.Compiler] ++ opts)

  @lower {NxCL.Compiler.Lower, :lower, 2}

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
      one = NxCL.Compiler.Lower.lower(expr.(), 1)
      two = NxCL.Compiler.Lower.lower(expr.(), 1)
      assert NxCL.Compiler.Program.signature(one) == NxCL.Compiler.Program.signature(two)
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

  describe "NxCL.Compiler.CompileCache" do
    # The cache reads its bounds from the app env; a restart clears it and
    # starts the sweep timer under the new value.
    defp restart_cache(env) do
      Application.put_env(:nx_opencl, CompileCache, env)
      reset_cache()

      on_exit(fn ->
        Application.delete_env(:nx_opencl, CompileCache)
        reset_cache()
      end)
    end

    defp reset_cache do
      :ok = Supervisor.terminate_child(NxCL.Supervisor, CompileCache)
      {:ok, _pid} = Supervisor.restart_child(NxCL.Supervisor, CompileCache)
    end

    test "max_items evicts the oldest entry first" do
      restart_cache(max_items: 2)
      CompileCache.put(:a, 1)
      CompileCache.put(:b, 2)
      CompileCache.put(:c, 3)

      assert CompileCache.fetch(:a) == {:error, :cache_miss}
      assert CompileCache.fetch(:b) == {:ok, 2}
      assert CompileCache.fetch(:c) == {:ok, 3}
      assert CompileCache.size() == 2
    end

    test "an entry past its ttl misses on read, and the sweep removes it later" do
      restart_cache(ttl: 200)
      CompileCache.put(:a, 1)
      assert CompileCache.fetch(:a) == {:ok, 1}

      # A suspended cache cannot sweep, so only the read can hide the entry.
      pid = Process.whereis(CompileCache)
      :sys.suspend(pid)

      TestSupport.wait_until(fn -> CompileCache.fetch(:a) == {:error, :cache_miss} end)

      assert CompileCache.size() == 1

      :sys.resume(pid)
      TestSupport.wait_until(fn -> CompileCache.size() == 0 end)
    end

    test "a bad bound raises when the cache starts" do
      Application.put_env(:nx_opencl, CompileCache, max_items: 0)
      on_exit(fn -> Application.delete_env(:nx_opencl, CompileCache) end)
      Process.flag(:trap_exit, true)
      :ok = Supervisor.terminate_child(NxCL.Supervisor, CompileCache)

      assert {:error, {%ArgumentError{message: message}, _}} =
               Supervisor.restart_child(NxCL.Supervisor, CompileCache)

      assert message =~ "max_items"

      Application.delete_env(:nx_opencl, CompileCache)
      {:ok, _pid} = Supervisor.restart_child(NxCL.Supervisor, CompileCache)
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
      assert {:ok, %{ctx: ctx, key: {nil, 0}}} = Devices.fetch(device: 0)
      assert ctx == Devices.fetch!().ctx
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
      ctx = Devices.fetch!().ctx
      entries = :persistent_term.get()

      refute Enum.any?(entries, fn {key, value} -> key == :nxcl_device_ctx or value == ctx end)

      refute Enum.any?(entries, fn {key, _value} ->
               match?({NxCL.Compiler.Executor, _}, key) or
                 match?({NxCL.Compiler.Executor, _, _}, key) or
                 match?({NxCL.Compiler.Executor, _, _, _}, key)
             end)
    end

    test "the backend honors :device" do
      t = Nx.tensor([1.0, 2.0], backend: {NxCL.Backend, device: 0})
      assert t.data.ctx == Devices.fetch!().ctx
    end
  end

  describe "a crash of the registry" do
    test "restarts the caches with it and opens a fresh context" do
      old_ctx = Devices.fetch!().ctx
      jit(fn x -> Nx.negate(x) end).(Nx.tensor([1.0]))
      assert ProgramCache.size() > 0
      assert CompileCache.size() > 0

      pids = for mod <- [Devices, ProgramCache, CompileCache], do: Process.whereis(mod)
      ref = Process.monitor(hd(pids))
      Process.exit(hd(pids), :kill)
      assert_receive {:DOWN, ^ref, :process, _pid, :killed}

      TestSupport.wait_until(fn ->
        new = for mod <- [Devices, ProgramCache, CompileCache], do: Process.whereis(mod)
        Enum.all?(new, &is_pid/1) and Enum.zip(pids, new) |> Enum.all?(fn {a, b} -> a != b end)
      end)

      assert ProgramCache.size() == 0
      assert CompileCache.size() == 0
      assert Devices.for_ctx(old_ctx) == nil

      new_ctx = Devices.fetch!().ctx
      assert %{key: {nil, 0}} = Devices.for_ctx(new_ctx)
      assert Nx.to_flat_list(jit(fn x -> Nx.negate(x) end).(Nx.tensor([1.0]))) == [-1.0]
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

    test "max_concurrency must be a positive integer" do
      for bad <- [0, -1, 1.5, :two] do
        assert_raise ArgumentError, ~r/:max_concurrency must be a positive integer/, fn ->
          NxCL.Compiler.__partitions_options__(max_concurrency: bad)
        end
      end
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

      {:ok, program} = Native.compile_program(Devices.fetch!().ctx, wire)
      buf = Native.buffer_write(Devices.fetch!().ctx, <<0::64>>)
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
      x = Nx.iota({64, 64}, type: :f32) |> Nx.divide(1000)
      fun = fn a -> Nx.dot(a, a) end
      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x)

      runner = Task.async(fn -> for _ <- 1..50, do: jit(fun).(x) end)
      evictor = Task.async(fn -> for n <- 1..30, do: jit(fn y -> Nx.add(y, n) end).(x) end)

      Task.await(evictor, 60_000)

      for result <- Task.await(runner, 60_000),
          do:
            assert(
              Nx.all_close(result, expected, rtol: 1.0e-4, atol: 1.0e-4) |> Nx.to_number() == 1
            )
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
