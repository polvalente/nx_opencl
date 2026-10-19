defmodule NxCL.NoFallbackTest do
  @moduledoc """
  Ops run on the device, not on the host CPU: a backend that raises on every
  compute callback is the default while a program runs, and calls into
  `Nx.BinaryBackend` are traced (`NxCL.TestSupport.NoFallback`). Only a host call
  (`Nx.runtime_call/4` and `Nx.io_call/2`) may compute on the host; `qr`, `eigh`,
  `svd` and `lu` run on the device.
  """
  use ExUnit.Case, async: false

  alias NxCL.TestSupport.NoFallback

  @moduletag :opencl

  # Ops by family, built at run time so each test owns its tensors. The list covers
  # every elementwise op, the reductions, scans, windows and sorts, the shape ops, the
  # indexing ops, `conv`, `fft` and the linear algebra ops. The ops of a family that
  # share a lowering (the comparison ops, say) are all in it, and the integer and
  # complex ops have lists of their own (`c_` marks a complex input).
  @float_unary ~w(exp expm1 log log1p sqrt rsqrt cbrt sin cos tan asin acos atan sinh cosh tanh
                  asinh acosh atanh erf erfc erf_inv sigmoid negate abs sign floor ceil round
                  is_nan is_infinity logical_not)a
  @float_binary ~w(add subtract multiply divide pow atan2 max min remainder equal not_equal
                   greater less greater_equal less_equal logical_and logical_or logical_xor)a
  @int_unary ~w(bitwise_not population_count count_leading_zeros)a
  @int_binary ~w(quotient bitwise_and bitwise_or bitwise_xor left_shift right_shift)a
  @reduce ~w(sum product reduce_max reduce_min argmax argmin all any cumulative_sum
             cumulative_product cumulative_max cumulative_min sort argsort)a
  @window ~w(window_sum window_product window_max window_min)a
  @complex_unary ~w(c_real c_imag c_conjugate c_abs c_exp)a
  @named ~w(top_k transpose reshape squeeze broadcast reverse pad concatenate stack slice
            put_slice take take_along_axis gather indexed_add indexed_put select clip as_type
            dot conv iota eye tile new_axis fft ifft fft2 ifft2 rfft irfft cholesky
            triangular_solve determinant qr eigh svd lu solve window_scatter_max)a

  @op_names @float_unary ++
              @float_binary ++
              @int_unary ++ @int_binary ++ @reduce ++ @window ++ @complex_unary ++ @named

  defp spd, do: Nx.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
  defp vec, do: Nx.tensor([1.0, 2.0, 3.0])
  defp ints, do: Nx.tensor([1, 2, 3], type: :s32)
  defp cplx, do: Nx.complex(vec(), Nx.tensor([0.5, -1.0, 2.0]))

  defp op(name) when name in @float_unary, do: {&apply(Nx, name, [&1]), [vec()]}
  defp op(name) when name in @float_binary, do: {&apply(Nx, name, [&1, &2]), [vec(), vec()]}
  defp op(name) when name in @int_unary, do: {&apply(Nx, name, [&1]), [ints()]}
  defp op(name) when name in @int_binary, do: {&apply(Nx, name, [&1, &2]), [ints(), ints()]}
  defp op(name) when name in @reduce, do: {&apply(Nx, name, [&1]), [spd()]}
  defp op(name) when name in @window, do: {&apply(Nx, name, [&1, {2}]), [vec()]}

  defp op(name) when name in @complex_unary do
    "c_" <> nx_name = Atom.to_string(name)
    {&apply(Nx, String.to_existing_atom(nx_name), [&1]), [cplx()]}
  end

  defp op(:top_k), do: {&Nx.top_k(&1, k: 2), [vec()]}
  defp op(:transpose), do: {&Nx.transpose/1, [spd()]}
  defp op(:reshape), do: {&Nx.reshape(&1, {1, 9}), [spd()]}
  defp op(:squeeze), do: {&Nx.squeeze(Nx.reshape(&1, {1, 3})), [vec()]}
  defp op(:broadcast), do: {&Nx.broadcast(&1, {2, 3}), [vec()]}
  defp op(:reverse), do: {&Nx.reverse/1, [spd()]}
  defp op(:pad), do: {&Nx.pad(&1, 0.0, [{1, 1, 0}]), [vec()]}
  defp op(:concatenate), do: {&Nx.concatenate([&1, &1]), [vec()]}
  defp op(:stack), do: {&Nx.stack([&1, &1]), [vec()]}
  defp op(:slice), do: {&Nx.slice(&1, [1], [2]), [vec()]}
  defp op(:put_slice), do: {&Nx.put_slice(&1, [1], Nx.tensor([9.0])), [vec()]}
  defp op(:take), do: {&Nx.take(&1, Nx.tensor([2, 0])), [vec()]}

  defp op(:take_along_axis),
    do: {&Nx.take_along_axis(&1, Nx.tensor([[0], [1], [2]]), axis: 1), [spd()]}

  defp op(:gather), do: {&Nx.gather(&1, Nx.tensor([[0, 1], [2, 2]])), [spd()]}

  defp op(:indexed_add),
    do: {&Nx.indexed_add(&1, Nx.tensor([[0], [2]]), Nx.tensor([1.0, 1.0])), [vec()]}

  defp op(:indexed_put),
    do: {&Nx.indexed_put(&1, Nx.tensor([[0], [2]]), Nx.tensor([1.0, 1.0])), [vec()]}

  defp op(:select), do: {&Nx.select(Nx.greater(&1, 1.5), &1, 0.0), [vec()]}
  defp op(:clip), do: {&Nx.clip(&1, 1.5, 2.5), [vec()]}
  defp op(:as_type), do: {&Nx.as_type(&1, :s32), [vec()]}
  defp op(:dot), do: {&Nx.dot/2, [spd(), spd()]}

  defp op(:conv),
    do:
      {&Nx.conv(&1, Nx.reshape(&2, {1, 1, 2}), padding: :same),
       [Nx.reshape(vec(), {1, 1, 3}), Nx.tensor([1.0, 2.0])]}

  defp op(:iota), do: {fn -> Nx.iota({3, 3}) end, []}
  defp op(:eye), do: {fn -> Nx.eye(3) end, []}
  defp op(:tile), do: {&Nx.tile(&1, [2]), [vec()]}
  defp op(:new_axis), do: {&Nx.new_axis(&1, 0), [vec()]}
  defp op(:fft), do: {&Nx.fft/1, [vec()]}
  defp op(:ifft), do: {&Nx.ifft/1, [vec()]}
  defp op(:fft2), do: {&Nx.fft2/1, [spd()]}
  defp op(:ifft2), do: {&Nx.ifft2/1, [spd()]}
  defp op(:rfft), do: {&Nx.rfft/1, [vec()]}
  defp op(:irfft), do: {&Nx.irfft/1, [vec()]}
  defp op(:cholesky), do: {&Nx.LinAlg.cholesky/1, [spd()]}
  defp op(:triangular_solve), do: {&Nx.LinAlg.triangular_solve/2, [spd(), vec()]}
  defp op(:determinant), do: {&Nx.LinAlg.determinant/1, [spd()]}
  defp op(:qr), do: {&Nx.LinAlg.qr/1, [spd()]}
  defp op(:eigh), do: {&Nx.LinAlg.eigh/1, [spd()]}
  defp op(:svd), do: {&Nx.LinAlg.svd/1, [spd()]}
  defp op(:lu), do: {&Nx.LinAlg.lu/1, [spd()]}
  defp op(:solve), do: {&Nx.LinAlg.solve/2, [spd(), vec()]}

  defp op(:window_scatter_max),
    do:
      {&Nx.window_scatter_max(&1, &2, 0.0, {2}, padding: :valid), [vec(), Nx.tensor([1.0, 2.0])]}

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

      assert_raise NoFallback.Fallback, ~r/Nx.BinaryBackend computed outside a host call/, fn ->
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

    test "a runtime callback may compute on the host" do
      fun = fn x -> Nx.runtime_call(x, x, fn t, _opts -> Nx.add(t, 1) end) end

      result =
        NoFallback.guard(fn -> Nx.Defn.jit_apply(fun, [vec()], compiler: NxCL.Compiler) end)

      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end

    test "a host call does not hide a call into Nx.BinaryBackend made outside it" do
      fun = fn x -> Nx.runtime_call(x, x, fn t, _opts -> Nx.add(t, 1) end) end
      x = vec()
      b = Nx.tensor([1.0, 2.0], backend: Nx.BinaryBackend)

      assert_raise NoFallback.Fallback, ~r/computed outside a host call/, fn ->
        NoFallback.guard(fn ->
          Nx.Defn.jit_apply(fun, [x], compiler: NxCL.Compiler)
          Nx.add(b, b)
        end)
      end
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
