defmodule NxCL.SortTest do
  @moduledoc """
  Differential rows for `sort` and `argsort` through `NxCL.TestSupport.Compare` against
  `Nx.BinaryBackend`, on every dtype Nx accepts for them (Nx raises on `c64`
  and `c128`). Lines of up to 9000 elements cover the single-group sort and
  the multi-pass sort, whose local capacity depends on the device. A second
  group sorts 1M elements and states the answer, since the reference is too
  slow there.

  `f64` rows carry `requires: :fp64`; `f16` rows carry `requires: :fp16`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 600_000

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
    {:f, 64}
  ]

  @needs %{{:f, 64} => :fp64, {:f, 16} => :fp16}

  defp with_edges(type, n) do
    edges = Gen.edge_values(type)
    Nx.concatenate([edges, Gen.random({max(n - Nx.size(edges), 1)}, type)])
  end

  describe "sort and argsort" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for shape <- [{1}, {63}, {65}, {1000}, {4097}, {3, 5}, {2, 3, 4}, {2, 1, 3, 1, 2}],
            axis <- Enum.to_list(0..(tuple_size(shape) - 1)),
            dir <- [:asc, :desc] do
          t = Gen.random(shape, type)
          assert_same(&Nx.sort(&1, axis: axis, direction: dir), [t])
          assert_same(&Nx.argsort(&1, axis: axis, direction: dir), [t])
        end

        # Lines past the local capacity: block sorts, then global and local merge passes.
        for shape <- [{9000}, {3, 9000}, {9000, 2}], axis <- [0, 1], dir <- [:asc, :desc] do
          if axis < tuple_size(shape) do
            t = Gen.random(shape, type)
            assert_same(&Nx.sort(&1, axis: axis, direction: dir), [t])
            assert_same(&Nx.argsort(&1, axis: axis, direction: dir), [t])
          end
        end

        # NaN, infinities, signed zero and the integer extremes.
        for n <- [7, 70, 4097, 9000], dir <- [:asc, :desc] do
          t = with_edges(type, n)
          assert_same(&Nx.sort(&1, direction: dir), [t])
          assert_same(&Nx.argsort(&1, direction: dir), [t])
        end

        # Every element equal: `argsort` must keep the input order in both directions.
        for n <- [65, 4097, 9000], dir <- [:asc, :desc] do
          flat = Nx.broadcast(Nx.tensor(3, type: type), {n})
          assert_same(&Nx.argsort(&1, direction: dir), [flat])
        end

        assert_same(&Nx.argsort(&1, axis: 1, type: :u16), [Gen.random({3, 70}, type)])
      end
    end
  end

  describe "stability" do
    test "argsort keeps the input order among equal keys, in both directions" do
      # 4097 keys from 0..3, so every key repeats about a thousand times.
      keys = Nx.remainder(Nx.iota({4097}, type: :s32), 4)

      for dir <- [:asc, :desc] do
        out = Nx.Defn.jit(&Nx.argsort(&1, direction: dir), compiler: NxCL.Compiler).(keys)
        idx = Nx.to_flat_list(out)
        sorted = Enum.map(idx, &rem(&1, 4))
        expected = if dir == :asc, do: Enum.sort(sorted), else: Enum.sort(sorted, :desc)
        assert sorted == expected

        # Within one key the indices rise.
        for {_key, group} <- Enum.group_by(idx, &rem(&1, 4)) do
          assert group == Enum.sort(group)
        end
      end
    end
  end

  describe "1M elements with stated answers" do
    @n 1_048_576

    # `i * 7919 mod 2^20` is a permutation of 0..2^20-1: 7919 is odd.
    defp permutation do
      Nx.iota({@n}, type: :s64)
      |> Nx.multiply(7919)
      |> Nx.remainder(@n)
      |> Nx.as_type(:s32)
      |> Nx.backend_copy(Nx.BinaryBackend)
    end

    test "sort gives 0..n-1 and argsort gives the inverse permutation" do
      x = permutation()
      inverse = Enum.find(1..@n, fn m -> rem(m * 7919, @n) == 1 end)

      asc = Nx.Defn.jit(&Nx.sort/1, compiler: NxCL.Compiler).(x)
      assert Nx.to_binary(asc) == Nx.to_binary(Nx.iota({@n}, type: :s32))

      desc = Nx.Defn.jit(&Nx.sort(&1, direction: :desc), compiler: NxCL.Compiler).(x)
      assert Nx.to_binary(desc) == Nx.to_binary(Nx.iota({@n}, type: :s32) |> Nx.reverse())

      # The index of value v is v * inverse mod n.
      expected =
        Nx.iota({@n}, type: :s64)
        |> Nx.multiply(inverse)
        |> Nx.remainder(@n)
        |> Nx.as_type(:s32)

      arg = Nx.Defn.jit(&Nx.argsort/1, compiler: NxCL.Compiler).(x)
      assert Nx.to_binary(arg) == Nx.to_binary(expected)
    end
  end
end
