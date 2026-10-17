defmodule NxCL.NoFallbackTest do
  @moduledoc """
  Ops run on the device, not on the host CPU: a backend that raises on every
  compute callback is the default while a program runs, and calls into
  `Nx.BinaryBackend` are traced (`NxCL.TestSupport.NoFallback`). Only a host call
  (`Nx.runtime_call/4`, `Nx.io_call/2`, and for now `qr`, `eigh`, `svd` and `lu`)
  may compute on the host.
  """
  use ExUnit.Case, async: false

  alias NxCL.TestSupport.NoFallback

  @moduletag :opencl

  # Representative ops by family, built at run time so each test owns its tensors.
  @op_names ~w(add exp sum cumulative_sum dot transpose sort top_k take select fft cholesky
               triangular_solve determinant)a

  defp spd, do: Nx.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
  defp vec, do: Nx.tensor([1.0, 2.0, 3.0])

  defp op(:add), do: {&Nx.add/2, [vec(), vec()]}
  defp op(:exp), do: {&Nx.exp/1, [vec()]}
  defp op(:sum), do: {&Nx.sum/1, [spd()]}
  defp op(:cumulative_sum), do: {&Nx.cumulative_sum/1, [vec()]}
  defp op(:dot), do: {&Nx.dot/2, [spd(), spd()]}
  defp op(:transpose), do: {&Nx.transpose/1, [spd()]}
  defp op(:sort), do: {&Nx.sort/1, [vec()]}
  defp op(:top_k), do: {&Nx.top_k(&1, k: 2), [vec()]}
  defp op(:take), do: {&Nx.take(&1, Nx.tensor([2, 0])), [vec()]}
  defp op(:select), do: {&Nx.select(Nx.greater(&1, 1.5), &1, 0.0), [vec()]}
  defp op(:fft), do: {&Nx.fft/1, [vec()]}
  defp op(:cholesky), do: {&Nx.LinAlg.cholesky/1, [spd()]}
  defp op(:triangular_solve), do: {&Nx.LinAlg.triangular_solve/2, [spd(), vec()]}
  defp op(:determinant), do: {&Nx.LinAlg.determinant/1, [spd()]}

  describe "the backend" do
    test "moves data and raises on compute" do
      t = Nx.tensor([1.0, 2.0], backend: NoFallback)

      assert Nx.to_flat_list(t) == [1.0, 2.0]
      assert Nx.to_flat_list(Nx.reshape(t, {2, 1})) == [1.0, 2.0]
      assert_raise NoFallback.Fallback, ~r/add\/3/, fn -> Nx.add(t, t) end
    end
  end

  describe "guard/1" do
    test "an intermediate built on the default backend raises" do
      assert_raise NoFallback.Fallback, ~r/iota/, fn ->
        NoFallback.guard(fn -> Nx.iota({3}) end)
      end
    end

    test "a call into Nx.BinaryBackend with no host call raises" do
      a = Nx.tensor([1.0, 2.0], backend: Nx.BinaryBackend)

      assert_raise NoFallback.Fallback, ~r/Nx.BinaryBackend computed with no host call/, fn ->
        NoFallback.guard(fn -> Nx.add(a, a) end)
      end
    end

    test "the trace stops after a raise" do
      assert_raise NoFallback.Fallback, fn -> NoFallback.guard(fn -> Nx.iota({3}) end) end
      a = Nx.tensor([1.0], backend: Nx.BinaryBackend)
      assert {_, %{binary: [{:add, 3, _}]}} = NoFallback.trace(fn -> Nx.add(a, a) end)
      assert Process.get({NoFallback, :collector}) == nil
    end
  end

  describe "the compiler" do
    for name <- @op_names do
      test "#{name} computes on the device" do
        {fun, args} = op(unquote(name))

        NoFallback.guard(fn -> apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args) end)
      end
    end

    test "qr runs as a host call, which may compute on the host" do
      NoFallback.guard(fn ->
        Nx.Defn.jit_apply(&Nx.LinAlg.qr/1, [spd()], compiler: NxCL.Compiler)
      end)
    end

    test "a runtime callback may compute on the host" do
      fun = fn x -> Nx.runtime_call(x, x, fn t, _opts -> Nx.add(t, 1) end) end

      result =
        NoFallback.guard(fn -> Nx.Defn.jit_apply(fun, [vec()], compiler: NxCL.Compiler) end)

      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end
  end

  describe "the eager backend" do
    for name <- @op_names do
      test "#{name} computes on the device" do
        {fun, args} = op(unquote(name))

        {_, seen} =
          NoFallback.trace(fn ->
            Nx.with_default_backend(NxCL.Backend, fn ->
              args = Enum.map(args, &Nx.backend_copy(&1, NxCL.Backend))
              apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)
            end)
          end)

        assert NoFallback.compute_calls(seen.binary) == []
      end
    end
  end
end
