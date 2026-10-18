defmodule NxCL.GatherScatterTest do
  @moduledoc """
  Differential rows for `gather`, `take`, `take_along_axis`, `indexed_add` and
  `indexed_put` through `NxCL.TestSupport.Compare` against `Nx.BinaryBackend`.

  ## Reference rules (read from `Nx.BinaryBackend` and the Nx docs, Nx 1.0)

    * **Out of bounds.** An index below 0 or at or above the axis size raises
      `ArgumentError` ("index 5 is out of bounds for axis 1 ...") in `gather`,
      `indexed_add` and `indexed_put`; `take` and `take_along_axis` are blocks
      whose default is a `gather`, so they raise too. A kernel cannot raise, so
      the device flags the index, clamps or skips it, and the run raises
      `NxCL.CompilerError` ("index out of bounds in gather") after the queue
      drains. `test/nxcl/indexing_test.exs` shows it for every op and index type.
    * **`axes`.** `gather` and `indexed_*` take a sorted list of axes. The last
      axis of `indices` holds one index per listed axis. The listed axes act as
      the leading axes, and the other axes keep their order. `gather` returns
      `[indices batch..., remaining axes...]`. An update row of `indexed_*` is a
      window over the remaining axes, starting at 0.
    * **Duplicates.** `indexed_add` sums every update for an index.
      `indexed_put` keeps the last update (the highest row): the reference
      groups updates by offset and takes the last. The Nx docs call this
      non-deterministic, but the binary backend is not, and the device matches it
      with a winner pass (`codegen/indexing.rs`). Rows below have duplicates.
    * **Index dtype.** Any integer type. Floats raise in Nx.
    * **Types.** `indexed_*` return the merged type of target and updates.
    * **Windows.** `BinaryBackend` treats an update row as one contiguous chunk of
      the (permuted) target, so it agrees with a window only when every window
      axis after the first is the full target axis. Rows use such windows;
      `indexing_test.exs` states the answer for a window that is not contiguous.

  `f64`/`c128` rows carry `requires: :fp64`; `f16` rows carry `requires: :fp16`.
  `indexed_add` on `s64`, `u64`, `f64` and `c128` needs `cl_khr_int64_base_atomics`;
  on a device without it the rows check the named error instead of a sum.
  """
  use ExUnit.Case, async: false

  import Bitwise
  import NxCL.TestSupport.Compare
  import NxCL.TestSupport.Device, only: [capability?: 1]

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 900_000

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

  @wide_add [{:s, 64}, {:u, 64}, {:f, 64}, {:c, 128}]

  @shapes [{7}, {5, 4}, {3, 4, 5}]

  # -- Generators ----------------------------------------------------------

  defp subsets(rank) do
    for mask <- 1..((1 <<< rank) - 1)//1,
        do: for(a <- 0..(rank - 1), (mask &&& 1 <<< a) != 0, do: a)
  end

  # Integer indices of shape `batch ++ [length(dims)]`; column `c` is in `[0, dims[c])`.
  defp indices(batch, dims, type, salt) do
    key = Nx.Random.key(seed() + :erlang.phash2(salt))

    {cols, _key} =
      Enum.map_reduce(dims, key, fn dim, key ->
        Nx.Random.randint(key, 0, dim, shape: batch, type: :s32)
      end)

    cols |> Nx.stack(axis: -1) |> Nx.as_type(type)
  end

  defp axis_dims(shape, axes), do: Enum.map(axes, &elem(shape, &1))

  defp values(shape, type, salt),
    do: Gen.random(shape, type, seed: seed() + :erlang.phash2(salt))

  # Window shape of an update row: the remaining axes, the first one halved.
  defp window(shape, axes, reduce?) do
    rest = for a <- 0..(tuple_size(shape) - 1), a not in axes, do: elem(shape, a)

    case {reduce?, rest} do
      {true, [first | tail]} -> [max(div(first, 2), 1) | tail]
      _ -> rest
    end
  end

  defp indexed_args(shape, axes, n, type, opts \\ []) do
    salt = {shape, axes, n, type, opts}
    window = window(shape, axes, Keyword.get(opts, :reduce, false))

    [
      values(shape, type, {:target, salt}),
      indices({n}, axis_dims(shape, axes), Keyword.get(opts, :index, {:s, 32}), {:idx, salt}),
      values(List.to_tuple([n | window]), Keyword.get(opts, :updates, type), {:upd, salt})
    ]
  end

  # -- gather --------------------------------------------------------------

  defp gather_rows(type, index_type) do
    for shape <- @shapes, axes <- subsets(tuple_size(shape)), batch <- [{4}, {2, 3}] do
      t = values(shape, type, {:gather, shape, axes})
      idx = indices(batch, axis_dims(shape, axes), index_type, {:gather_idx, shape, axes, batch})
      assert_same(&Nx.gather(&1, &2, axes: axes), [t, idx])
    end

    # Default axes: the leading ones.
    t = values({5, 4}, type, :default_axes)
    idx = indices({6}, [5, 4], index_type, :default_axes_idx)
    assert_same(&Nx.gather/2, [t, idx])
  end

  defp take_rows(type, index_type) do
    for shape <- @shapes, axis <- 0..(tuple_size(shape) - 1), idx_shape <- [{5}, {2, 3}] do
      t = values(shape, type, {:take, shape, axis})

      idx =
        indices(idx_shape, [elem(shape, axis)], index_type, {:take_idx, shape, axis})
        |> Nx.squeeze(axes: [-1])

      assert_same(&Nx.take(&1, &2, axis: axis), [t, idx])
    end
  end

  defp take_along_axis_rows(type, index_type) do
    for shape <- @shapes, axis <- 0..(tuple_size(shape) - 1) do
      out_shape = put_elem(shape, axis, 3)
      t = values(shape, type, {:taa, shape, axis})

      idx =
        indices(out_shape, [elem(shape, axis)], index_type, {:taa_idx, shape, axis})
        |> Nx.squeeze(axes: [-1])

      assert_same(
        &Nx.take_along_axis(&1, &2, axis: axis),
        [t, idx]
      )
    end
  end

  describe "gather, take and take_along_axis on every value type" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        gather_rows(type, {:s, 32})
        take_rows(type, {:s, 32})
        take_along_axis_rows(type, {:s, 32})
      end
    end
  end

  describe "gather, take and take_along_axis on other index types" do
    for index_type <- [{:s, 64}, {:u, 8}, {:u, 64}, {:s, 8}, {:u, 32}] do
      test "with #{Nx.Type.to_string(index_type)} indices" do
        index_type = unquote(Macro.escape(index_type))
        gather_rows({:f, 32}, index_type)
        take_rows({:s, 32}, index_type)
        take_along_axis_rows({:f, 32}, index_type)
      end
    end

    test "a u64 index above 2^63 is out of bounds, not negative" do
      t = Nx.iota({4}, type: :f32)
      idx = Nx.tensor([[1 <<< 63]], type: :u64)
      jit = Nx.Defn.jit(&Nx.gather/2, compiler: NxCL.Compiler)

      assert_raise NxCL.CompilerError, ~r/index out of bounds in gather/, fn -> jit.(t, idx) end
    end
  end

  # -- indexed_add and indexed_put -----------------------------------------

  defp put_rows(type, index_type) do
    for shape <- @shapes,
        axes <- subsets(tuple_size(shape)),
        n <- [1, 17, 40],
        reduce <- [false, true] do
      args = indexed_args(shape, axes, n, type, index: index_type, reduce: reduce)
      assert_same(&Nx.indexed_put(&1, &2, &3, axes: axes), args)
    end
  end

  defp add_rows(type, index_type) do
    for shape <- @shapes,
        axes <- subsets(tuple_size(shape)),
        n <- [1, 17, 40],
        reduce <- [false, true] do
      args = indexed_args(shape, axes, n, type, index: index_type, reduce: reduce)
      assert_same(&Nx.indexed_add(&1, &2, &3, axes: axes), args)
    end
  end

  defp assert_no_atomics_error(type) do
    shape = {5, 4}
    args = indexed_args(shape, [0], 9, type)
    jit = Nx.Defn.jit(&Nx.indexed_add(&1, &2, &3), compiler: NxCL.Compiler)

    assert_raise NxCL.CompilerError, ~r/cl_khr_int64_base_atomics/, fn ->
      apply(jit, args)
    end
  end

  describe "indexed_put on every value type" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        put_rows(unquote(Macro.escape(type)), {:s, 32})
      end
    end

    for index_type <- [{:s, 64}, {:u, 8}, {:u, 64}] do
      test "with #{Nx.Type.to_string(index_type)} indices" do
        put_rows({:f, 32}, unquote(Macro.escape(index_type)))
      end
    end

    test "a scalar update at one full index" do
      t = Nx.iota({3, 4}, type: :f32)
      assert_same(&Nx.indexed_put(&1, &2, &3), [t, Nx.tensor([1, 2]), 9.5])
    end
  end

  describe "indexed_add on every value type" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        if type in @wide_add and not capability?(:int64_atomics) do
          assert_no_atomics_error(type)
        else
          add_rows(type, {:s, 32})
        end
      end
    end

    for index_type <- [{:s, 64}, {:u, 8}, {:u, 64}] do
      test "with #{Nx.Type.to_string(index_type)} indices" do
        add_rows({:f, 32}, unquote(Macro.escape(index_type)))
      end
    end

    test "a scalar update at one full index" do
      t = Nx.iota({3, 4}, type: :f32)
      assert_same(&Nx.indexed_add(&1, &2, &3), [t, Nx.tensor([1, 2]), 9.5])
    end

    test "target and updates of different types merge to the wider type" do
      for {target, updates} <- [{{:s, 32}, {:f, 32}}, {{:f, 32}, {:s, 32}}, {{:u, 8}, {:s, 16}}] do
        args = indexed_args({5, 4}, [0], 17, target, updates: updates)
        assert_same(&Nx.indexed_add/3, args)
        assert_same(&Nx.indexed_put/3, args)
      end
    end
  end
end
