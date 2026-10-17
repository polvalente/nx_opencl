defmodule NxCL.HostCallTest.Tag do
  @moduledoc false
  # A callback that sends `{label, flat values}` to the process that runs it.
  def fun(label), do: fn t -> send(self(), {label, Nx.to_flat_list(t)}) end
end

defmodule NxCL.HostCallTest do
  @moduledoc """
  Host callbacks (`io_call`, `runtime_call`, `print_value`) through `NxCL.Compiler`.

  What `Nx.Defn.Evaluator` does, probed on Nx 1.0 (stage 11), and what we match:

    * Nx 1.0 has no `:hook` node. `print_value` is an `io_call`, and a named
      `io_call` runs its own callback (the `:hooks` override is out of scope).
    * A callback runs in the calling process, once for each time its node is
      evaluated: once per execution, only in the `cond` clause taken, once per
      `while` iteration. A node used twice fires once.
    * The evaluator walks from the outputs, so an effect whose result no output
      reads never fires. We lower from the outputs too, so it does not fire.
    * Effects fire in argument order, depth first: `io_call(io_call(x, f1) + 1, f2)`
      runs `f1` then `f2`, and a tuple output `{io_call(a, f1), io_call(b, f2)}` runs
      `f1` then `f2`.
    * The callback receives its input container with tensors on the evaluator's
      backend (`Nx.BinaryBackend` here). We pass tensors on the default backend,
      made with `Nx.from_binary/2`, so the two agree. The return value of an
      `io_call` callback is ignored; its result is the input, unchanged.
    * A `runtime_call` result must be `Nx.compatible?/2` with the template. The
      evaluator raises a `RuntimeError`; we raise an `ArgumentError` that names
      both templates.
  """
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO
  import Nx.Defn
  import NxCL.Parity

  alias NxCL.Lower
  alias NxCL.TestSupport.Trace

  @moduletag :opencl

  defp run_jit(fun, compiler \\ NxCL.Compiler), do: Nx.Defn.jit(fun, compiler: compiler)

  defp flush(acc \\ []) do
    receive do
      msg -> flush([msg | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  # Runs `fun` on both compilers and returns the messages each one sent.
  defp both(fun, args) do
    flush()
    expected = apply(run_jit(fun, Nx.Defn.Evaluator), args)
    reference = flush()
    actual = apply(run_jit(fun), args)
    {{expected, reference}, {actual, flush()}}
  end

  defp tag(label), do: NxCL.HostCallTest.Tag.fun(label)

  defn branches(p, x) do
    cond do
      p > 0 -> io_call(x + 1, NxCL.HostCallTest.Tag.fun(:yes))
      true -> io_call(x - 1, NxCL.HostCallTest.Tag.fun(:no))
    end
  end

  defn counted_dynamic(x, n) do
    {_i, acc, _n} =
      while {i = 0, acc = x, n}, i < n do
        acc = io_call(acc, NxCL.HostCallTest.Tag.fun(:i))
        {i + 1, acc + 1, n}
      end

    acc
  end

  defn counted_static(x) do
    {_i, acc} =
      while {i = 0, acc = x}, i < 5 do
        acc = io_call(acc, NxCL.HostCallTest.Tag.fun(:i))
        {i + 1, acc + 1}
      end

    acc
  end

  defn cond_in_while(x) do
    {_i, acc} =
      while {i = 0, acc = x}, i < 4 do
        acc =
          cond do
            rem(i, 2) == 0 -> io_call(acc + 1, NxCL.HostCallTest.Tag.fun(:even))
            true -> io_call(acc + 10, NxCL.HostCallTest.Tag.fun(:odd))
          end

        {i + 1, acc}
      end

    acc
  end

  describe "runtime_call" do
    test "a result that feeds a dot matches the reference" do
      x = Nx.iota({3, 4}, type: :f32)
      w = Nx.iota({4, 2}, type: :f32)

      fun = fn x, w ->
        x = Nx.add(x, 1)
        y = Nx.runtime_call(x, x, fn t, _ -> Nx.multiply(t, 2) end)
        z = Nx.dot(y, w)
        Nx.sum(z) |> Nx.add(z)
      end

      assert_parity(:runtime_call, fun, [x, w])
    end

    test "passes options and returns tuples and maps" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      fun = fn x ->
        out = {Nx.template({3}, :f32), Nx.template({}, :s32)}

        {a, b} =
          Nx.runtime_call(out, x, [scale: 3], fn t, opts ->
            {Nx.multiply(t, opts[:scale]), Nx.tensor(Nx.size(t))}
          end)

        m =
          Nx.runtime_call(
            %{s: Nx.template({}, :f32), d: Nx.template({3}, :f32)},
            {a, x},
            [],
            fn {p, q}, _ -> %{s: Nx.sum(p), d: Nx.subtract(p, q)} end
          )

        {Nx.add(a, 1), Nx.multiply(b, 2), m}
      end

      assert_parity(:runtime_call, fun, [x])
    end

    test "a wrong shape raises and names both templates" do
      fun = fn x -> Nx.runtime_call(x, x, fn t, _ -> Nx.concatenate([t, t]) end) end

      error =
        assert_raise ArgumentError, fn -> run_jit(fun).(Nx.tensor([1.0, 2.0])) end

      assert error.message =~ "f32[2]"
      assert error.message =~ "f32[4]"
      assert error.message =~ "output template"
    end

    test "a wrong type raises and names both templates" do
      fun = fn x -> Nx.runtime_call(x, x, fn t, _ -> Nx.as_type(t, :s32) end) end

      error = assert_raise ArgumentError, fn -> run_jit(fun).(Nx.tensor([1.0, 2.0])) end

      assert error.message =~ "f32[2]"
      assert error.message =~ "s32[2]"
    end
  end

  describe "io_call" do
    test "fires once per execution, with the right values, in the calling process" do
      fun = fn x -> Nx.io_call(Nx.add(x, 1), tag(:a)) |> Nx.multiply(2) end
      x = Nx.tensor([1.0, 2.0])
      compiled = run_jit(fun)

      assert_parity(:io_call, fun, [x])
      flush()

      assert Nx.to_flat_list(compiled.(x)) == [4.0, 6.0]
      assert flush() == [{:a, [2.0, 3.0]}]
      compiled.(Nx.tensor([5.0, 6.0]))
      assert flush() == [{:a, [6.0, 7.0]}]
    end

    test "inside a cond only the taken branch fires" do
      x = Nx.tensor([1.0, 2.0])

      for {p, label} <- [{1, :yes}, {0, :no}] do
        {{e, e_msgs}, {a, a_msgs}} = both(&branches/2, [Nx.tensor(p), x])
        assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
        assert [{^label, _}] = a_msgs
        assert a_msgs == e_msgs
        assert_parity(:io_call, &branches/2, [Nx.tensor(p), x], record: false)
        flush()
      end
    end

    test "inside a 5-iteration while it fires 5 times" do
      for {fun, args} <- [
            {&counted_static/1, [Nx.tensor(0.0)]},
            {&counted_dynamic/2, [Nx.tensor(0.0), Nx.tensor(5)]}
          ] do
        {{e, e_msgs}, {a, a_msgs}} = both(fun, args)
        assert Nx.to_number(a) == Nx.to_number(e)
        assert length(a_msgs) == 5
        assert a_msgs == e_msgs
        assert Enum.map(a_msgs, fn {:i, [v]} -> v end) == [0.0, 1.0, 2.0, 3.0, 4.0]
      end
    end

    test "a zero-trip while never fires" do
      {{_e, e_msgs}, {_a, a_msgs}} = both(&counted_dynamic/2, [Nx.tensor(0.0), Nx.tensor(0)])
      assert a_msgs == [] and e_msgs == []
    end

    test "a callback in a cond inside a while fires for the taken clauses only" do
      {{e, e_msgs}, {a, a_msgs}} = both(&cond_in_while/1, [Nx.tensor(0.0)])
      assert Nx.to_number(a) == Nx.to_number(e)
      assert Enum.map(a_msgs, &elem(&1, 0)) == [:even, :odd, :even, :odd]
      assert a_msgs == e_msgs
    end

    test "tuple and map inputs arrive as the same container" do
      pid = self()

      fun = fn a, b ->
        {c, d} =
          Nx.io_call({Nx.add(a, 1), %{k: b}}, fn {x, %{k: y}} ->
            send(pid, {:got, Nx.to_flat_list(x), Nx.to_flat_list(y)})
          end)

        Nx.add(c, d.k)
      end

      args = [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
      {{e, e_msgs}, {a, a_msgs}} = both(fun, args)
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
      assert a_msgs == [{:got, [2.0, 3.0], [3.0, 4.0]}]
      assert a_msgs == e_msgs
    end

    test "a named io_call runs its callback, and one with no callback does nothing" do
      pid = self()

      fun = fn x ->
        y =
          Nx.io_call(Nx.add(x, 1), :my_hook, fn t -> send(pid, {:named, Nx.to_flat_list(t)}) end)

        Nx.io_call(y, :silent)
      end

      {{e, e_msgs}, {a, a_msgs}} = both(fun, [Nx.tensor([1.0])])
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
      assert a_msgs == [{:named, [2.0]}]
      assert a_msgs == e_msgs
    end

    test "print_value prints what the evaluator prints" do
      fun = fn x -> Nx.Defn.Kernel.print_value(Nx.multiply(x, 2), label: "v") end
      x = Nx.tensor([1.0, 2.0])

      expected = capture_io(fn -> run_jit(fun, Nx.Defn.Evaluator).(x) end)
      actual = capture_io(fn -> run_jit(fun).(x) end)

      assert actual =~ "v: #Nx.Tensor<"
      assert actual == expected
    end
  end

  describe "order and reach" do
    test "two chained effects fire in the evaluator's order" do
      fun = fn x ->
        y = Nx.io_call(Nx.add(x, 1), tag(:first))
        z = Nx.io_call(Nx.multiply(y, 3), tag(:second))
        w = Nx.io_call(Nx.subtract(z, 1), tag(:third))
        Nx.add(w, y)
      end

      {{e, e_msgs}, {a, a_msgs}} = both(fun, [Nx.tensor([1.0, 2.0])])
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
      assert Enum.map(a_msgs, &elem(&1, 0)) == [:first, :second, :third]
      assert a_msgs == e_msgs
    end

    test "independent effects in a tuple output fire in output order" do
      fun = fn a, b ->
        {Nx.io_call(Nx.add(a, 1), tag(:one)), Nx.io_call(Nx.add(b, 1), tag(:two))}
      end

      {{_e, e_msgs}, {_a, a_msgs}} = both(fun, [Nx.tensor([1.0]), Nx.tensor([2.0])])
      assert Enum.map(a_msgs, &elem(&1, 0)) == [:one, :two]
      assert a_msgs == e_msgs
    end

    test "an effect fires once however many nodes read it" do
      fun = fn x ->
        y = Nx.io_call(Nx.add(x, 1), tag(:once))
        Nx.add(Nx.multiply(y, y), y)
      end

      {{_e, e_msgs}, {_a, a_msgs}} = both(fun, [Nx.tensor([1.0])])
      assert length(a_msgs) == 1
      assert a_msgs == e_msgs
    end

    test "an effect whose result no output reads fires in neither" do
      fun = fn x ->
        _dropped = Nx.io_call(Nx.add(x, 1), tag(:dropped))
        Nx.multiply(x, 2)
      end

      {{_e, e_msgs}, {_a, a_msgs}} = both(fun, [Nx.tensor([1.0])])
      assert e_msgs == []
      assert a_msgs == []
    end

    test "an effect whose result only feeds another effect still fires first" do
      fun = fn x ->
        y = Nx.io_call(Nx.add(x, 1), tag(:inner))
        Nx.io_call(Nx.negate(y), tag(:outer))
      end

      {{e, e_msgs}, {a, a_msgs}} = both(fun, [Nx.tensor([1.0])])
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
      assert Enum.map(a_msgs, &elem(&1, 0)) == [:inner, :outer]
      assert a_msgs == e_msgs
    end
  end

  describe "transfers" do
    test "tensors a callback does not take never leave the device" do
      fun = fn a, b ->
        c = Nx.add(b, 1)
        a = Nx.io_call(Nx.add(a, 1), tag(:a))
        {a, c}
      end

      a = Nx.iota({4}, type: :f32)
      b = Nx.iota({1024, 64}, type: :f32)
      {{a2, c}, stats} = Trace.run(fn -> run_jit(fun).(a, b) end)

      assert Nx.to_flat_list(a2) == [1.0, 2.0, 3.0, 4.0]
      assert Nx.shape(c) == {1024, 64}
      assert stats.host_calls == 1
      assert stats.host_reads == 1
      assert stats.host_read_bytes == 16
      # A side-effect call returns its input buffer; nothing is uploaded.
      assert stats.host_uploads == 0
    end

    test "a runtime_call reads and uploads only its own tensors" do
      fun = fn a, b ->
        big = Nx.multiply(b, 2)
        r = Nx.runtime_call(a, a, fn t, _ -> Nx.add(t, 1) end)
        {r, big}
      end

      {_, stats} =
        Trace.run(fn ->
          run_jit(fun).(Nx.iota({3}, type: :f32), Nx.iota({512, 512}, type: :f32))
        end)

      assert %{host_calls: 1, host_reads: 1, host_read_bytes: 12, host_uploads: 1} = stats
    end

    test "a container input counts each leaf it takes" do
      fun = fn a, b, c ->
        {x, y} = Nx.io_call({Nx.add(a, 1), Nx.add(b, 1)}, fn {_p, _q} -> :ok end)
        {x, y, Nx.add(c, 1)}
      end

      {_, stats} =
        Trace.run(fn -> run_jit(fun).(Nx.iota({2}), Nx.iota({3}), Nx.iota({100_000})) end)

      assert %{host_reads: 2, host_read_bytes: 20} = stats
    end

    test "a callback in a while reads once per iteration" do
      {_, stats} =
        Trace.run(fn -> run_jit(&counted_dynamic/2).(Nx.tensor(0.0), Nx.tensor(5)) end)

      assert %{host_calls: 5, host_reads: 5, host_read_bytes: 20} = stats
    end
  end

  describe "the :hooks option" do
    test "a non-empty map raises ArgumentError" do
      fun = fn x -> Nx.io_call(x, :named, fn _ -> :ok end) end
      hooks = %{named: fn _ -> :ok end}

      error =
        assert_raise ArgumentError, fn ->
          Nx.Defn.jit(fun, compiler: NxCL.Compiler, hooks: hooks).(Nx.tensor(1.0))
        end

      assert error.message =~ ":hooks"
      assert error.message =~ ":named"
    end

    test "an empty map is accepted" do
      fun = fn x -> Nx.add(x, 1) end

      assert Nx.to_number(Nx.Defn.jit(fun, compiler: NxCL.Compiler, hooks: %{}).(Nx.tensor(1.0))) ==
               2.0
    end
  end

  describe "lowering" do
    test "a host call is one instruction with list-valued types and a side table" do
      fun = fn x ->
        {a, b} =
          Nx.runtime_call({Nx.template({2}, :f32), Nx.template({}, :s32)}, x, fn t, _ ->
            {t, Nx.tensor(1)}
          end)

        Nx.add(a, 1) |> Nx.add(Nx.as_type(b, :f32))
      end

      expr = Nx.Defn.debug_expr_apply(fun, [Nx.iota({2}, type: :f32)])
      program = Lower.lower(expr, 1)

      assert [%{op: :host_call} = call | _] = program.instructions
      assert call.operands == [{:input, 0}]
      assert call.out_type == [{:f, 32}, {:s, 32}]
      assert call.out_shape == [{2}, {}]
      assert call.attrs == [0]
      assert [%{kind: :runtime_call, fun: f}] = program.host_calls
      assert is_function(f, 2)
    end

    test "a callback in a branch lowers into the sub-program" do
      expr = Nx.Defn.debug_expr_apply(&branches/2, [Nx.tensor(1), Nx.tensor([1.0])])
      program = Lower.lower(expr, 2)

      assert [_pred, %{op: :cond, subprograms: subs}] = program.instructions
      assert Enum.any?(subs, fn sub -> Enum.any?(sub.instructions, &(&1.op == :host_call)) end)
      assert length(program.host_calls) == 2
    end
  end
end
