defmodule NxCL.TestSupportTest do
  @moduledoc "Tests of the test helpers: the comparison rules, device gating, and generators."
  use ExUnit.Case, async: false

  alias NxCL.TestSupport.{Calls, Compare, Device, Factors, Gen, Trace}

  # Runs the evaluator, then breaks the result in the way `:mode` names.
  defmodule BrokenCompiler do
    @behaviour Nx.Defn.Compiler

    @impl true
    def __jit__(key, vars, fun, args_list, opts) do
      mode = Keyword.fetch!(opts, :mode)
      [result] = Nx.Defn.Evaluator.__jit__(key, vars, fun, args_list, [])
      [break(result, mode)]
    end

    @impl true
    defdelegate __compile__(key, vars, fun, opts), to: Nx.Defn.Evaluator

    @impl true
    defdelegate __partitions_options__(opts), to: Nx.Defn.Evaluator

    @impl true
    defdelegate __to_backend__(opts), to: Nx.Defn.Evaluator

    defp break(t, :value), do: Nx.add(t, 1.0)
    defp break(t, :type), do: Nx.as_type(t, :f64)
    defp break(t, :shape), do: Nx.reshape(t, {1, :auto})
    defp break(t, :nan), do: Nx.put_slice(t, [0], Nx.tensor([:nan], type: Nx.type(t)))
    defp break(t, :names), do: Nx.rename(t, [:x])
    defp break(t, :ok), do: t
  end

  defp run(mode, args \\ [Nx.tensor([1.0, 2.0, 3.0])]) do
    Compare.assert_same(&Nx.negate/1, args, compiler: BrokenCompiler, compiler_opts: [mode: mode])
  end

  describe "Compare.assert_same/3" do
    test "passes on an identical result" do
      assert :ok = run(:ok)
    end

    test "fails on a wrong value" do
      assert_raise ExUnit.AssertionError, ~r/value mismatch/, fn -> run(:value) end
    end

    test "fails on a wrong type" do
      assert_raise ExUnit.AssertionError, ~r/type/, fn -> run(:type) end
    end

    test "fails on a wrong shape" do
      assert_raise ExUnit.AssertionError, ~r/shape/, fn -> run(:shape) end
    end

    test "fails on a wrong name" do
      assert_raise ExUnit.AssertionError, ~r/names/, fn -> run(:names) end
    end

    test "fails on NaN where a number is expected" do
      assert_raise ExUnit.AssertionError, ~r/NaN mismatch/, fn -> run(:nan) end
    end

    test "NaN equals NaN" do
      assert :ok = run(:ok, [Nx.tensor([:nan, 1.0])])
    end

    test "failure message carries the seed" do
      e = assert_raise ExUnit.AssertionError, fn -> run(:value) end
      assert e.message =~ "seed: #{Compare.seed()}"
    end
  end

  describe "capability gating" do
    # The tag excludes the row on a device without the extension, so these rows run
    # only where the device has it.
    @tag requires: :fp64
    test "f64 rows run only where the device has cl_khr_fp64" do
      assert Device.capability?(:fp64)
    end

    @tag requires: :fp16
    test "f16 rows run only where the device has cl_khr_fp16" do
      assert Device.capability?(:fp16)
    end

    setup do
      old = Application.fetch_env(:nx_opencl, :test_extensions)
      on_exit(fn -> restore(old) end)
    end

    test "a device without cl_khr_fp64 gives a skip message and an exclude filter" do
      Application.put_env(:nx_opencl, :test_extensions, "cl_khr_fp16")

      refute Device.capability?(:fp64)
      assert Device.skip_reason(:fp64) =~ "lacks cl_khr_fp64"
      assert {:requires, :fp64} in Device.unsupported_filters()
      refute {:requires, :fp16} in Device.unsupported_filters()
    end

    test "a device with the extension is not skipped" do
      Application.put_env(
        :nx_opencl,
        :test_extensions,
        "cl_khr_fp64 cl_khr_fp16 cl_khr_int64_base_atomics"
      )

      assert Device.skip_reason(:fp64) == nil
      assert Device.unsupported_filters() == []
    end

    test "a device without 64-bit atomics gives a skip message and an exclude filter" do
      Application.put_env(:nx_opencl, :test_extensions, "cl_khr_fp64 cl_khr_fp16")

      refute Device.capability?(:int64_atomics)
      assert Device.skip_reason(:int64_atomics) =~ "lacks cl_khr_int64_base_atomics"
      assert Device.unsupported_filters() == [requires: :int64_atomics]
    end

    defp restore({:ok, v}), do: Application.put_env(:nx_opencl, :test_extensions, v)
    defp restore(:error), do: Application.delete_env(:nx_opencl, :test_extensions)
  end

  describe "call tracers" do
    # A raise in the traced function must leave the process untraced and kill the
    # collector, or the next tracer in the same process fails.
    defp raise_with_collector(run) do
      parent = self()

      assert_raise RuntimeError, "boom", fn ->
        run.(fn ->
          {:tracer, collector} = :erlang.trace_info(self(), :tracer)
          send(parent, {:collector, collector})
          raise "boom"
        end)
      end

      assert_received {:collector, collector}
      ref = Process.monitor(collector)
      assert_receive {:DOWN, ^ref, :process, _pid, _reason}
      assert {:flags, []} = :erlang.trace_info(self(), :flags)
    end

    test "Trace.run cleans up after a raise" do
      raise_with_collector(&Trace.run/1)
      assert {:ok, %{host_calls: 0}} = Trace.run(fn -> :ok end)
    end

    test "Calls.count cleans up after a raise" do
      mfa = {Enum, :map, 2}
      raise_with_collector(&Calls.count([mfa], &1))
      assert {2, %{^mfa => 1}} = Calls.count([mfa], fn -> length(Enum.map([1, 2], & &1)) end)
    end
  end

  describe "Gen" do
    test "same seed gives the same tensor" do
      a = Gen.random({65}, {:f, 32}, seed: 7)
      assert Nx.to_binary(a) == Nx.to_binary(Gen.random({65}, {:f, 32}, seed: 7))
    end

    test "two unseeded calls draw different data" do
      a = Gen.random({65}, {:f, 32})
      b = Gen.random({65}, {:f, 32})
      refute Nx.to_binary(a) == Nx.to_binary(b)
    end

    test "random tensors have the asked shape and type" do
      for shape <- Gen.shapes(), type <- [{:f, 32}, {:s, 8}, {:u, 16}, {:c, 64}] do
        t = Gen.random(shape, type, seed: 1)
        assert t.shape == shape
        assert t.type == type
      end
    end

    test "edge values build for every real dtype" do
      for {kind, _} = type <- Gen.dtypes(), kind in [:s, :u, :f, :bf] do
        assert Gen.edge_values(type).type == type
      end
    end
  end

  describe "Factors on a half type" do
    @describetag :opencl

    # max |A| is 4, so a `bf16` tolerance is 8 * 2^-7 * 3 * 4 = 0.75. Each wrong factor
    # is off by 4 or more in the identity it breaks.
    defp half do
      Nx.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 2.0]], type: :bf16)
    end

    test "right factors pass" do
      x = half()
      assert :ok = Factors.assert_factors(:qr, &Nx.LinAlg.qr/1, x)
      assert :ok = Factors.assert_factors(:eigh, &Nx.LinAlg.eigh/1, x)
      assert :ok = Factors.assert_factors(:svd, &Nx.LinAlg.svd/1, x)
      assert :ok = Factors.assert_factors(:lu, &Nx.LinAlg.lu/1, x)
    end

    test "an R with an entry below the diagonal fails" do
      wrong = fn x ->
        {q, r} = Nx.LinAlg.qr(x)
        {q, Nx.put_slice(r, [2, 0], Nx.tensor([[4.0]], type: :bf16))}
      end

      # `Q R == A` is checked first and breaks too; either failure shows the check works.
      assert_raise ExUnit.AssertionError, ~r/(Q R == A|R upper triangular)/, fn ->
        Factors.assert_factors(:qr, wrong, half())
      end
    end

    test "an R with a flipped row fails" do
      wrong = fn x ->
        {q, r} = Nx.LinAlg.qr(x)
        {q, Nx.put_slice(r, [0, 0], Nx.negate(r[0]) |> Nx.new_axis(0))}
      end

      assert_raise ExUnit.AssertionError, ~r/Q R == A/, fn ->
        Factors.assert_factors(:qr, wrong, half())
      end
    end

    test "wrong eigenvalues fail" do
      wrong = fn x ->
        {w, v} = Nx.LinAlg.eigh(x)
        {Nx.add(w, Nx.tensor(4.0, type: :bf16)), v}
      end

      assert_raise ExUnit.AssertionError, ~r/A V == V diag\(w\)/, fn ->
        Factors.assert_factors(:eigh, wrong, half())
      end
    end

    test "wrong singular values fail" do
      wrong = fn x ->
        {u, s, vt} = Nx.LinAlg.svd(x)
        {u, Nx.multiply(s, Nx.tensor(2.0, type: :bf16)), vt}
      end

      assert_raise ExUnit.AssertionError, ~r/U S V\^H == A/, fn ->
        Factors.assert_factors(:svd, wrong, half())
      end
    end

    test "wrong singular values fail on f32, past the svd floor" do
      x = Nx.as_type(half(), :f32)
      assert :ok = Factors.assert_factors(:svd, &Nx.LinAlg.svd/1, x)

      # An `f32` check allows the reference's own residual, so a factor that is wrong in
      # the reference too would pass. The right function runs the reference.
      wrong = fn x ->
        {u, s, vt} = Nx.LinAlg.svd(x)
        # 1.001 times S is off by about 5e-3, over the 1.6e-3 floor at max |A| = 4.
        {u, Nx.multiply(s, 1.001), vt}
      end

      assert_raise ExUnit.AssertionError, ~r/U S V\^H == A/, fn ->
        Factors.assert_factors(:svd, wrong, x, reference_fun: &Nx.LinAlg.svd/1)
      end
    end

    test "a P that is not 0 or 1 fails" do
      # `P` doubled and `U` halved keep `P L U == A`, `L` and `U` right.
      wrong = fn x ->
        {p, l, u} = Nx.LinAlg.lu(x)
        {Nx.multiply(p, 2), l, Nx.divide(u, 2)}
      end

      assert_raise ExUnit.AssertionError, ~r/P entries are 0 or 1/, fn ->
        Factors.assert_factors(:lu, wrong, half(), reference_fun: &Nx.LinAlg.lu/1)
      end
    end

    test "a P with two ones in a row fails" do
      # For this `A`, `P = [[1, 0], [1, 0]]` still gives `P L U == A` with a unit lower
      # `L` and an upper `U`; only `P^T P == I` shows that `P` is not a permutation.
      a = Nx.tensor([[1.0, 0.0], [1.0, 0.0]])

      wrong = fn x ->
        {_p, l, u} = Nx.LinAlg.lu(x)
        {Nx.tensor([[1.0, 0.0], [1.0, 0.0]]), l, u}
      end

      assert :ok = Factors.assert_factors(:lu, &Nx.LinAlg.lu/1, a)

      assert_raise ExUnit.AssertionError, ~r/P\^T P == I/, fn ->
        Factors.assert_factors(:lu, wrong, a, reference_fun: &Nx.LinAlg.lu/1)
      end
    end

    test "singular values in ascending order fail" do
      # The same factors in reverse order still multiply to `A` and stay orthonormal.
      wrong = fn x ->
        {u, s, vt} = Nx.LinAlg.svd(x)
        {Nx.reverse(u, axes: [1]), Nx.reverse(s), Nx.reverse(vt, axes: [0])}
      end

      assert_raise ExUnit.AssertionError, ~r/S descending/, fn ->
        Factors.assert_factors(:svd, wrong, Nx.as_type(half(), :f32),
          reference_fun: &Nx.LinAlg.svd/1
        )
      end
    end

    test "an L with a wrong entry above the diagonal fails" do
      wrong = fn x ->
        {p, l, u} = Nx.LinAlg.lu(x)
        {p, Nx.put_slice(l, [0, 2], Nx.tensor([[4.0]], type: :bf16)), u}
      end

      assert_raise ExUnit.AssertionError, ~r/(P L U == A|L unit lower triangular)/, fn ->
        Factors.assert_factors(:lu, wrong, half())
      end
    end
  end
end
