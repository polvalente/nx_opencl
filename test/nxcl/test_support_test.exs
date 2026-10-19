defmodule NxCL.TestSupportTest do
  @moduledoc "Tests of the test helpers: the comparison rules, device gating, and generators."
  use ExUnit.Case, async: false

  alias NxCL.TestSupport.{Compare, Device, Factors, Gen}

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
    # These run the evaluator on both sides. They prove the tag skips a device
    # without the extension and runs on one that has it.
    @tag requires: :fp64
    test "f64 rows run only where the device has cl_khr_fp64" do
      assert Device.capability?(:fp64)

      assert :ok =
               Compare.assert_same(&Nx.negate/1, [Nx.tensor([1.0, 2.0], type: :f64)],
                 compiler: Nx.Defn.Evaluator
               )
    end

    @tag requires: :fp16
    test "f16 rows run only where the device has cl_khr_fp16" do
      assert Device.capability?(:fp16)

      assert :ok =
               Compare.assert_same(&Nx.negate/1, [Nx.tensor([1.0, 2.0], type: :f16)],
                 compiler: Nx.Defn.Evaluator
               )
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

  describe "Gen" do
    test "same seed gives the same tensor" do
      a = Gen.random({65}, {:f, 32}, seed: 7)
      assert Nx.to_binary(a) == Nx.to_binary(Gen.random({65}, {:f, 32}, seed: 7))
    end

    test "two unseeded calls draw different data, and the same seed gives the same data" do
      a = Gen.random({65}, {:f, 32})
      b = Gen.random({65}, {:f, 32})
      refute Nx.to_binary(a) == Nx.to_binary(b)

      assert Nx.to_binary(Gen.random({65}, {:s, 8}, seed: 3)) ==
               Nx.to_binary(Gen.random({65}, {:s, 8}, seed: 3))
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
        if type != {:f, 64}, do: assert(tuple_size(Gen.overflow_values(type).shape) == 1)
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
