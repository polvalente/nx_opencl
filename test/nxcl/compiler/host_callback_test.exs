defmodule NxCL.HostCallbackTest do
  @moduledoc """
  Differential rows for `runtime_call` and `io_call` against `Nx.Defn.Evaluator`.
  Each dtype gets a `runtime_call` between device ops (before, after, and feeding
  a `dot`) and an `io_call` between device ops; the `io_call` rows also check that
  the callback saw the same tensor on both compilers. Order, counts, errors and
  transfers are in `NxCL.HostCallTest`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl

  @types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 8},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:f, 16},
    {:bf, 16},
    {:f, 32},
    {:f, 64},
    {:c, 64},
    {:c, 128}
  ]

  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  defp flush(acc \\ []) do
    receive do
      msg -> flush([msg | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  defp seen(t), do: {t.type, t.shape, Nx.to_binary(t)}

  # The reference runs first, then the compiler, then the eager backend; each should
  # send the same message.
  defp assert_same_messages(fun, args) do
    flush()
    assert_same(fun, args)
    assert [reference, actual, eager] = flush()
    assert reference == actual
    assert reference == eager
  end

  for type <- @types do
    @tag requires: Map.get(@needs, type)
    test "runtime_call between device ops on #{Nx.Type.to_string(type)}" do
      type = unquote(Macro.escape(type))
      x = Gen.random({3, 4}, type)
      w = Gen.random({4, 2}, type)

      twice = fn t, _opts -> Nx.add(t, t) end

      before_and_after = fn x ->
        y = Nx.add(x, x)
        z = Nx.runtime_call(y, y, twice)
        Nx.multiply(z, x)
      end

      into_dot = fn x, w ->
        z = Nx.runtime_call(x, x, twice)
        Nx.dot(z, w)
      end

      assert_same(before_and_after, [x])
      assert_same(into_dot, [x, w])
    end

    @tag requires: Map.get(@needs, type)
    test "io_call between device ops on #{Nx.Type.to_string(type)}" do
      type = unquote(Macro.escape(type))
      x = Gen.random({3, 4}, type)
      pid = self()

      fun = fn x ->
        y = Nx.add(x, x)
        z = Nx.io_call(y, fn t -> send(pid, {:io, seen(t)}) end)
        Nx.multiply(z, x)
      end

      assert_same_messages(fun, [x])
    end
  end
end
