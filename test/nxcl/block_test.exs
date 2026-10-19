defmodule NxCL.BlockTest.Custom do
  @moduledoc false
  # A block struct outside `Nx.Block.*`: it has no table entry and lowers its default.
  defstruct []
end

defmodule NxCL.BlockTest do
  @moduledoc """
  The block hook (`NxCL.Compiler.Lower.Block`): every `Nx.Block.*` struct has a decision,
  each decision shows in the lowered program, and no block lowers to a `:host_call`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport, only: [lower: 2, ops: 1]

  alias NxCL.Compiler.Lower.Block
  alias NxCL.TestSupport.Trace

  @moduletag :opencl

  defp block_structs do
    Application.spec(:nx, :modules)
    |> Enum.filter(&String.starts_with?(Atom.to_string(&1), "Elixir.Nx.Block."))
    |> Enum.filter(&(Code.ensure_loaded?(&1) and function_exported?(&1, :__struct__, 0)))
    |> Enum.sort()
  end

  defp spd(n),
    do: Nx.add(Nx.multiply(Nx.eye(n), n), Nx.iota({n, n}, type: :f32) |> Nx.divide(100))

  defp sym(n) do
    a = spd(n)
    Nx.divide(Nx.add(a, Nx.transpose(a)), 2)
  end

  # One call per struct, through the public Nx function that builds the block.
  defp cases do
    x = Nx.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    z = Nx.complex(x, Nx.negate(x))
    idx = Nx.tensor([1, 0, 1])
    a4 = Nx.add(spd(4), Nx.iota({4, 4}, type: :f32))

    %{
      Nx.Block.LogicalNot => {&Nx.logical_not/1, [x]},
      Nx.Block.Phase => {&Nx.phase/1, [z]},
      Nx.Block.CumulativeSum => {&Nx.cumulative_sum/1, [x]},
      Nx.Block.CumulativeProduct => {&Nx.cumulative_product/1, [x]},
      Nx.Block.CumulativeMin => {&Nx.cumulative_min/1, [x]},
      Nx.Block.CumulativeMax => {&Nx.cumulative_max/1, [x]},
      Nx.Block.LinAlg.Cholesky => {&Nx.LinAlg.cholesky/1, [spd(3)]},
      Nx.Block.LinAlg.Solve => {&Nx.LinAlg.solve/2, [spd(3), Nx.iota({3})]},
      Nx.Block.LinAlg.Determinant => {&Nx.LinAlg.determinant/1, [a4]},
      Nx.Block.AllClose => {&Nx.all_close/2, [x, x]},
      Nx.Block.Take => {&Nx.take(&1, &2, axis: 1), [x, idx]},
      Nx.Block.TakeAlongAxis =>
        {&Nx.take_along_axis(&1, &2, axis: 1), [x, Nx.tensor([[0, 1, 2], [2, 1, 0]])]},
      Nx.Block.TopK => {&Nx.top_k(&1, k: 2), [x]},
      Nx.Block.FFT2 => {&Nx.fft2/1, [x]},
      Nx.Block.IFFT2 => {&Nx.ifft2/1, [z]},
      Nx.Block.RFFT => {&Nx.rfft/1, [x]},
      Nx.Block.IRFFT => {&Nx.irfft/1, [z]},
      Nx.Block.LinAlg.QR => {&Nx.LinAlg.qr/1, [spd(3)]},
      Nx.Block.LinAlg.Eigh => {&Nx.LinAlg.eigh/1, [sym(3)]},
      Nx.Block.LinAlg.SVD => {&Nx.LinAlg.svd/1, [spd(3)]},
      Nx.Block.LinAlg.LU => {&Nx.LinAlg.lu/1, [spd(3)]}
    }
  end

  test "every Nx.Block struct has a decision" do
    structs = block_structs()
    assert structs != []

    missing = structs -- Map.keys(Block.decisions())
    assert missing == [], "no decision for #{inspect(missing)}"

    extra = Map.keys(Block.decisions()) -- structs
    assert extra == [], "decision for a struct Nx no longer defines: #{inspect(extra)}"

    for {mod, decision} <- Block.decisions() do
      assert decision in [:native, :default],
             "bad decision for #{inspect(mod)}: #{inspect(decision)}"
    end

    for mod <- [Nx.Block.LinAlg.QR, Nx.Block.LinAlg.Eigh, Nx.Block.LinAlg.SVD, Nx.Block.LinAlg.LU] do
      assert Block.decisions()[mod] == :default
    end
  end

  test "every struct has a case, and every case lowers" do
    assert Enum.sort(Map.keys(cases())) == block_structs()

    for {mod, {fun, args}} <- cases() do
      assert %NxCL.Compiler.Program{} = lower(fun, args), "#{inspect(mod)} did not lower"
    end
  end

  test "no block lowers to a host call" do
    for {mod, {fun, args}} <- cases() do
      program = lower(fun, args)
      refute :host_call in ops(program), "#{inspect(mod)} has a host call"
      assert program.host_calls == [], "#{inspect(mod)} has a host call entry"
    end
  end

  test "a native block lowers to its own instruction" do
    native = %{
      Nx.Block.LogicalNot => :logical_not,
      Nx.Block.Phase => :phase,
      Nx.Block.CumulativeSum => :cumulative_sum,
      Nx.Block.CumulativeProduct => :cumulative_product,
      Nx.Block.CumulativeMin => :cumulative_min,
      Nx.Block.CumulativeMax => :cumulative_max,
      Nx.Block.LinAlg.Cholesky => :cholesky
    }

    assert Enum.sort(Map.keys(native)) ==
             Enum.sort(for {mod, :native} <- Block.decisions(), do: mod)

    for {mod, op} <- native do
      {fun, args} = Map.fetch!(cases(), mod)
      assert ops(lower(fun, args)) == [op], inspect(mod)
    end
  end

  test "a block outside the table lowers its default" do
    fun = fn x ->
      Nx.block(%NxCL.BlockTest.Custom{}, [x], x, fn _struct, t -> Nx.multiply(Nx.add(t, 1), 2) end)
    end

    program = lower(fun, [Nx.tensor([1.0, 2.0])])
    assert ops(program) == [:add, :multiply]
    assert program.host_calls == []
    assert Block.decision(%NxCL.BlockTest.Custom{}) == :default
  end

  describe "linear algebra runs on the device" do
    defp run_counted(fun, args) do
      {result, counts} =
        Trace.run(fn -> apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args) end)

      # Segments ran, so a host-call count of zero is a real zero.
      assert counts.segment_runs > 0
      {result, counts.host_calls}
    end

    test "cholesky, triangular_solve and fft run with no host call" do
      a = spd(5)
      l = Nx.LinAlg.cholesky(a)
      x = Nx.iota({2, 6}, type: :f32)

      for {fun, args, device_op} <- [
            {&Nx.LinAlg.cholesky/1, [a], :cholesky},
            {&Nx.LinAlg.triangular_solve(&1, &2), [l, Nx.iota({5, 2})], :triangular_solve},
            {&Nx.fft/1, [x], :fft},
            {&Nx.ifft/1, [x], :ifft},
            {&Nx.fft(&1, length: 5), [x], :fft}
          ] do
        assert device_op in ops(lower(fun, args))
        refute :host_call in ops(lower(fun, args))
        assert {_, 0} = run_counted(fun, args)
      end
    end

    test "solve runs on the device: two triangular_solve instructions, no host call" do
      args = [spd(4), Nx.iota({4, 2})]
      program = lower(&Nx.LinAlg.solve/2, args)
      assert Enum.count(ops(program), &(&1 == :triangular_solve)) == 2
      refute :host_call in ops(program)
      assert {_, 0} = run_counted(&Nx.LinAlg.solve/2, args)
    end

    test "qr, eigh, svd and lu run with no host call and satisfy their identities" do
      for {kind, fun, arg} <- [
            {:qr, &Nx.LinAlg.qr/1, spd(4)},
            {:eigh, &Nx.LinAlg.eigh/1, sym(4)},
            {:svd, &Nx.LinAlg.svd/1, spd(4)},
            {:lu, &Nx.LinAlg.lu/1, spd(4)}
          ] do
        refute :host_call in ops(lower(fun, [arg])), "#{kind}"
        assert {_, 0} = run_counted(fun, [arg])
        NxCL.TestSupport.Factors.assert_factors(kind, fun, arg)
      end
    end

    test "with this compiler as the default, qr still runs on the device" do
      Nx.Defn.default_options(compiler: NxCL.Compiler)

      try do
        {{q, r}, 0} = run_counted(&Nx.LinAlg.qr/1, [spd(3)])
        assert Nx.all_close(Nx.dot(q, r), spd(3), atol: 1.0e-5) |> Nx.to_number() == 1
      after
        Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
      end
    end
  end
end
