defmodule NxCL.TripwireTest do
  @moduledoc """
  The no-fallback tripwire (`NxCL.Parity.Tripwire`): it raises on a compute
  callback, catches a call into `Nx.BinaryBackend`, and counts host calls per
  row. Only `qr`, `eigh`, `svd` and `lu` make a host call on their own; the
  whole matrix runs under the same guard (`NxCL.Parity`), and
  `check_parity.exs` reads the counts back from the results file.
  """
  use ExUnit.Case, async: false

  alias NxCL.Parity.Tripwire

  @moduletag :opencl

  defp run_compiled(fun, args) do
    Tripwire.guard(:probe, fn ->
      jit = Nx.Defn.jit(fun, compiler: Tripwire.Compiler, inner: NxCL.Compiler)
      apply(jit, args)
    end)
  end

  setup do
    Tripwire.reset()
    :ok
  end

  describe "the backend" do
    test "moves data and raises on compute" do
      t = Nx.tensor([1.0, 2.0], backend: Tripwire)

      assert Nx.to_flat_list(t) == [1.0, 2.0]
      assert Nx.to_flat_list(Nx.reshape(t, {2, 1})) == [1.0, 2.0]
      assert_raise Tripwire.Fallback, ~r/add\/3/, fn -> Nx.add(t, t) end
    end
  end

  describe "guard/2" do
    test "an intermediate built on the default backend raises" do
      assert_raise Tripwire.Fallback, ~r/iota/, fn ->
        Tripwire.guard(:probe, fn -> Nx.iota({3}) end)
      end
    end

    test "a call into Nx.BinaryBackend with no host call raises" do
      a = Nx.tensor([1.0, 2.0], backend: Nx.BinaryBackend)

      assert_raise Tripwire.Fallback, ~r/Nx.BinaryBackend computed with no host call/, fn ->
        Tripwire.guard(:probe, fn -> Nx.add(a, a) end)
      end
    end

    test "the trace stops after a raise" do
      assert_raise Tripwire.Fallback, fn -> Tripwire.guard(:probe, fn -> Nx.iota({3}) end) end
      a = Nx.tensor([1.0], backend: Nx.BinaryBackend)
      assert {_, %{binary: [{:add, 3, _}]}} = Tripwire.trace(fn -> Nx.add(a, a) end)
      assert Process.get({Tripwire, :collector}) == nil
    end

    test "the user's own function is not on the compute path" do
      literal = Nx.tensor(2.0, backend: Nx.BinaryBackend)

      result =
        run_compiled(fn x -> Nx.add(x, Nx.multiply(literal, literal)) end, [Nx.tensor(1.0)])

      assert Nx.to_number(result) == 5.0
    end
  end

  describe "host calls" do
    @g5 [
      block_lin_alg_qr: &Nx.LinAlg.qr/1,
      block_lin_alg_eigh: &Nx.LinAlg.eigh/1,
      block_lin_alg_svd: &Nx.LinAlg.svd/1,
      block_lin_alg_lu: &Nx.LinAlg.lu/1
    ]

    @spd Nx.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])

    for {kind, fun} <- @g5 do
      test "#{kind} makes one host call, counted under its block" do
        run_compiled(unquote(fun), [@spd])

        assert Tripwire.host_calls() == %{unquote(kind) => 1}
      end
    end

    test "solve runs its lu on the host and counts under lu" do
      run_compiled(&Nx.LinAlg.solve/2, [@spd, Nx.tensor([1.0, 2.0, 3.0])])

      assert Tripwire.host_calls() == %{block_lin_alg_lu: 1}
    end

    test "a user callback counts as :callback" do
      fun = fn x ->
        Nx.runtime_call(x, x, fn t, _opts -> Nx.add(t, 1) end)
      end

      run_compiled(fun, [Nx.tensor([1.0, 2.0, 3.0])])

      assert Tripwire.host_calls() == %{callback: 1}
    end

    test "ops with a device kernel make none" do
      vec = Nx.tensor([1.0, 2.0, 3.0])

      ops = [
        cholesky: {&Nx.LinAlg.cholesky/1, [@spd]},
        triangular_solve: {&Nx.LinAlg.triangular_solve/2, [@spd, vec]},
        determinant: {&Nx.LinAlg.determinant/1, [@spd]},
        fft: {&Nx.fft/1, [vec]},
        sort: {&Nx.sort/1, [vec]},
        dot: {&Nx.dot/2, [@spd, @spd]},
        top_k: {&Nx.top_k(&1, k: 2), [vec]}
      ]

      for {_name, {fun, args}} <- ops, do: run_compiled(fun, args)

      assert Tripwire.host_calls() == %{}
    end
  end
end
