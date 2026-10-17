defmodule NxCL.ControlFlowTest do
  @moduledoc """
  Checks of the control-flow executor that a differential row cannot state: which
  kernels launch, how many predicates the executor reads, device buffer counts
  after a loop, the zero-iteration result, and the lowering error for a value an
  inner scope takes from its parent.
  """
  use ExUnit.Case, async: false

  import Nx.Defn

  alias NxCL.Executor
  alias NxCL.Lower
  alias NxCL.Native
  alias NxCL.TestSupport.Trace

  @moduletag :opencl

  defp run_jit(fun), do: Nx.Defn.jit(fun, compiler: NxCL.Compiler)

  defp launches() do
    {_builds, _hits, launches} = Native.kernel_cache_stats(Executor.ctx())
    launches
  end

  defp live_buffers(), do: Native.live_buffer_count(Executor.ctx())

  # The live count after garbage collection stops freeing handles.
  defp settled_buffers(previous \\ nil) do
    :erlang.garbage_collect()
    Process.sleep(30)
    count = live_buffers()
    if count == previous, do: count, else: settled_buffers(count)
  end

  # Resources free when the BEAM collects their handles.
  defp wait_for_buffers(expected, tries \\ 100) do
    :erlang.garbage_collect()

    cond do
      live_buffers() == expected -> :ok
      tries == 0 -> flunk("live buffers #{live_buffers()}, expected #{expected}")
      true -> Process.sleep(20) && wait_for_buffers(expected, tries - 1)
    end
  end

  defn pick(p, x, idx) do
    cond do
      p ->
        x + 1

      true ->
        # Never taken in the first test: out of range, and costly.
        sorted = Nx.sort(Nx.dot(x, Nx.transpose(x)), axis: 0)
        Nx.sum(Nx.gather(sorted, idx)) + Nx.sum(sorted)
    end
  end

  defn counter(n) do
    {i, _n} =
      while {i = 0, n}, i < n do
        {i + 1, n}
      end

    i
  end

  defn counter_static() do
    while i = 0, i < 10 do
      i + 1
    end
  end

  defn counter_tuple(x) do
    {i, acc, _x} =
      while {i = 0, acc = x, x}, i < 1000 do
        {i + 1, acc + x, x}
      end

    {i, acc}
  end

  defn zero_trips(x, n) do
    {i, acc, _x, _n} =
      while {i = 0, acc = x, x, n}, i < n do
        {i + 1, acc + x, x, n}
      end

    {i, acc}
  end

  defn outer_and_loop(x) do
    y = x * 2

    {_i, acc} =
      while {i = 0, acc = x}, i < 3 do
        {i + 1, acc + 1}
      end

    {y, acc}
  end

  describe "branches" do
    test "an untaken branch launches no kernel and never fails" do
      x = Nx.iota({16, 4}, type: :f32)
      # Out of range for the sorted 16 x 16 matrix: this branch would raise if it ran.
      idx = Nx.tensor([[99, 99]])
      p = Nx.tensor(1, type: :u8)

      assert Nx.to_flat_list(run_jit(&pick/3).(p, x, idx)) == Nx.to_flat_list(Nx.add(x, 1))

      before = launches()
      run_jit(&pick/3).(p, x, idx)
      taken = launches() - before

      # `x + 1` is one launch; a plain jit of the same add shows the cost of the branch.
      before = launches()
      run_jit(&Nx.add(&1, 1)).(x)
      assert taken == launches() - before
    end

    test "the branch that runs may fail, and only that one" do
      x = Nx.iota({4, 4}, type: :f32)
      idx = Nx.tensor([[99, 99]])

      assert_raise NxCL.CompilerError, ~r/index out of bounds in gather/, fn ->
        run_jit(&pick/3).(Nx.tensor(0, type: :u8), x, idx)
      end
    end
  end

  describe "loops" do
    test "a zero-iteration loop returns its initial carry" do
      x = Nx.tensor([1.5, -2.0, 3.25])
      {i, acc} = run_jit(&zero_trips/2).(x, Nx.tensor(0))

      assert Nx.to_number(i) == 0
      assert Nx.to_flat_list(acc) == Nx.to_flat_list(x)
    end

    test "a 1000-iteration loop leaves the live buffer count where it was" do
      # Warm run: builds the kernels and the plan.
      assert Nx.to_number(run_jit(&counter/1).(Nx.tensor(3))) == 3
      start = settled_buffers()

      assert Nx.to_number(run_jit(&counter/1).(Nx.tensor(1000))) == 1000
      wait_for_buffers(start)

      {i, _acc} = run_jit(&counter_tuple/1).(Nx.tensor([1.0, 2.0]))
      assert Nx.to_number(i) == 1000
      wait_for_buffers(start)
    end

    test "a static trip count reads no predicate" do
      run_jit(&counter_static/0).()

      {result, counts} = Trace.run(fn -> run_jit(&counter_static/0).() end)

      assert Nx.to_number(result) == 10
      # The loop has no cond and a static count, so every scope run is a body run.
      assert %{predicate_reads: 0, scope_runs: 10} = counts
    end

    test "a dynamic loop reads one predicate per iteration, and one more to stop" do
      run_jit(&counter/1).(Nx.tensor(1))

      {result, counts} = Trace.run(fn -> run_jit(&counter/1).(Nx.tensor(7)) end)

      assert Nx.to_number(result) == 7
      # Scope runs are the predicate scopes plus the body runs (7 iterations).
      assert %{predicate_reads: 8} = counts
      assert counts.scope_runs - counts.predicate_reads == 7
    end

    test "a cond reads predicates in order and stops at the first true one" do
      fun = fn p, q, x -> pick_two(p, q, x) end
      x = Nx.tensor([1.0])
      run_jit(fun).(Nx.tensor(1), Nx.tensor(1), x)

      {_, counts} = Trace.run(fn -> run_jit(fun).(Nx.tensor(1), Nx.tensor(1), x) end)
      assert %{predicate_reads: 1} = counts

      {_, counts} = Trace.run(fn -> run_jit(fun).(Nx.tensor(0), Nx.tensor(1), x) end)
      assert %{predicate_reads: 2} = counts
    end
  end

  defn pick_two(p, q, x) do
    cond do
      p -> x + 1
      q -> x + 2
      true -> x + 3
    end
  end

  describe "lowering errors" do
    test "an inner scope that reads a value of its parent fails at lowering" do
      {y, acc} = Nx.Defn.debug_expr_apply(&outer_and_loop/1, [Nx.tensor([1.0, 2.0])])

      %Nx.Tensor{data: %Nx.Defn.Expr{op: :elem, args: [loop, pos]}} = acc
      %Nx.Tensor{data: %Nx.Defn.Expr{op: :while, args: [initial, arg, pred, block]}} = loop
      {i_next, acc_next} = block
      %Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [carry, _one]}} = acc_next

      # `y` lives in the parent scope. Nx refuses to build this graph, so patch it in.
      bad_next = Nx.Defn.Tree.put_args(acc_next, [carry, y])
      bad_loop = Nx.Defn.Tree.put_args(loop, [initial, arg, pred, {i_next, bad_next}])
      bad_acc = Nx.Defn.Tree.put_args(acc, [bad_loop, pos])

      assert_raise ArgumentError,
                   ~r/inner scope reads :(multiply|parameter) from its parent scope/,
                   fn ->
                     Lower.lower({y, bad_acc}, 1)
                   end
    end

    test "nesting past the limit raises" do
      assert_raise ArgumentError, ~r/nest more than 64 levels/, fn ->
        state = %{
          depth: 64,
          instrs: [],
          refs: %{},
          captures: [],
          constants: [],
          avail: %{},
          outer: %{},
          scope: :root
        }

        Lower.lower_scope(state, [], %{}, :while)
      end
    end
  end
end
