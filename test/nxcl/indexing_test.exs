defmodule NxCL.IndexingTest do
  @moduledoc """
  What the differential rows in `compiler/indexing_parity_test.exs` cannot show:
  the instructions lowering emits, out-of-bounds indices (the reference raises,
  so the device must), the `indexed_put` duplicate rule with stated values, the
  100 000-update `indexed_add` against an `f64` sum, the embedding lookup, the
  error for 64-bit atomics, and zero-size inputs through the native wire. The
  reference rules are listed in the moduledoc of that test.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare
  import NxCL.TestSupport.Device, only: [capability?: 1]

  alias NxCL.Lower
  alias NxCL.Native
  alias NxCL.Program.Instruction

  @moduletag :opencl
  @moduletag timeout: 600_000

  @f32 {:f, 32}

  defp lower(fun, args) do
    flat = args |> Nx.Defn.Composite.flatten_list() |> length()
    fun |> Nx.Defn.debug_expr_apply(args) |> Lower.lower(flat)
  end

  defp ops(program), do: Enum.map(program.instructions, & &1.op)

  defp run(fun, args), do: apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

  describe "lowering" do
    test "gather keeps its axes as attrs" do
      t = Nx.iota({3, 4, 5}, type: :f32)
      idx = Nx.tensor([[0, 1], [2, 3]])

      assert [%Instruction{op: :gather, attrs: [0, 2], out_shape: {2, 4}}] =
               lower(&Nx.gather(&1, &2, axes: [0, 2]), [t, idx]).instructions
    end

    test "take and take_along_axis lower through their default expression" do
      t = Nx.iota({3, 4}, type: :f32)

      take = lower(&Nx.take(&1, &2, axis: 1), [t, Nx.tensor([2, 0])])
      assert :gather in ops(take)
      refute :block in ops(take)
      assert take.output_template.shape == {3, 2}

      along = lower(&Nx.take_along_axis(&1, &2, axis: 1), [t, Nx.tensor([[0], [1], [2]])])
      assert :gather in ops(along)
      assert :concatenate in ops(along)
    end

    test "indexed ops cast target and updates to the merged type" do
      t = Nx.iota({4, 2}, type: :s32)
      idx = Nx.tensor([[0], [1]])
      upd = Nx.tensor([[1.5, 2.5], [3.5, 4.5]])

      program = lower(&Nx.indexed_add/3, [t, idx, upd])

      assert [
               %Instruction{op: :as_type, out_type: @f32},
               %Instruction{op: :indexed_add, out_type: @f32, attrs: [0]}
             ] = program.instructions
    end
  end

  describe "out-of-bounds indices" do
    # The reference raises ArgumentError for each of these. The device sets a flag,
    # and the run raises after the queue drains.
    @bad [{:s, 32, 4}, {:s, 32, -1}, {:s, 64, 100}, {:s, 64, -5}, {:u, 8, 4}, {:u, 8, 255}]

    defp bad_index(type, value), do: Nx.tensor([[value]], type: type)

    test "gather raises on the reference and on the device" do
      t = Nx.iota({4, 3}, type: :f32)

      for {kind, bits, value} <- @bad do
        idx = bad_index({kind, bits}, value)

        assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.gather(t, idx) end

        assert_raise NxCL.CompilerError, ~r/index out of bounds in gather/, fn ->
          run(&Nx.gather/2, [t, idx])
        end
      end
    end

    test "gather checks every axis of an index tuple" do
      t = Nx.iota({4, 3}, type: :f32)

      for idx <- [Nx.tensor([[1, 3]]), Nx.tensor([[4, 0]]), Nx.tensor([[0, -1]])] do
        assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.gather(t, idx) end

        assert_raise NxCL.CompilerError, ~r/index out of bounds/, fn ->
          run(&Nx.gather/2, [t, idx])
        end
      end
    end

    test "take and take_along_axis raise through their gather" do
      t = Nx.iota({4, 3}, type: :f32)

      assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.take(t, Nx.tensor([4])) end

      assert_raise NxCL.CompilerError, ~r/index out of bounds in gather/, fn ->
        run(&Nx.take/2, [t, Nx.tensor([4])])
      end

      idx = Nx.tensor([[3], [0], [0], [0]])
      assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.take_along_axis(t, idx, axis: 1) end

      assert_raise NxCL.CompilerError, ~r/index out of bounds in gather/, fn ->
        run(&Nx.take_along_axis(&1, &2, axis: 1), [t, idx])
      end
    end

    test "indexed_add and indexed_put raise on the reference and on the device" do
      t = Nx.iota({4, 3}, type: :f32)
      upd = Nx.tensor([[1.0, 2.0, 3.0]])

      for {kind, bits, value} <- @bad do
        idx = bad_index({kind, bits}, value)

        assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.indexed_add(t, idx, upd) end
        assert_raise ArgumentError, ~r/out of bounds/, fn -> Nx.indexed_put(t, idx, upd) end

        assert_raise NxCL.CompilerError, ~r/index out of bounds in indexed_add/, fn ->
          run(&Nx.indexed_add/3, [t, idx, upd])
        end

        assert_raise NxCL.CompilerError, ~r/index out of bounds in indexed_put/, fn ->
          run(&Nx.indexed_put/3, [t, idx, upd])
        end
      end
    end

    test "an index past the end of one of many rows still fails the run" do
      t = Nx.iota({10}, type: :f32)
      idx = Nx.concatenate([Nx.iota({99, 1}, type: :s32) |> Nx.remainder(10), Nx.tensor([[10]])])
      upd = Nx.broadcast(1.0, {100})

      assert_raise NxCL.CompilerError, ~r/index out of bounds in indexed_add/, fn ->
        run(&Nx.indexed_add/3, [t, idx, upd])
      end
    end

    test "a failed run leaves the next run clean" do
      t = Nx.iota({4}, type: :f32)
      jit = Nx.Defn.jit(&Nx.gather/2, compiler: NxCL.Compiler)

      assert_raise NxCL.CompilerError, fn -> jit.(t, Nx.tensor([[9]])) end
      assert Nx.to_flat_list(jit.(t, Nx.tensor([[2], [3]]))) == [2.0, 3.0]
    end
  end

  describe "indexed_put duplicates" do
    test "the last update wins, as in the reference" do
      t = Nx.broadcast(0.0, {4})
      idx = Nx.tensor([[1], [2], [1], [1], [2]])
      upd = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

      assert Nx.to_flat_list(Nx.indexed_put(t, idx, upd)) == [0.0, 40.0, 50.0, 0.0]
      assert Nx.to_flat_list(run(&Nx.indexed_put/3, [t, idx, upd])) == [0.0, 40.0, 50.0, 0.0]
    end

    test "a thousand rows on a few targets give the same answer in every run" do
      t = Nx.broadcast(0, {5})
      idx = Nx.remainder(Nx.iota({1000, 1}, type: :s32) |> Nx.multiply(7), 3)
      upd = Nx.iota({1000}, type: :s32)
      expected = Nx.to_binary(Nx.indexed_put(t, idx, upd))
      jit = Nx.Defn.jit(&Nx.indexed_put/3, compiler: NxCL.Compiler)

      for _ <- 1..20, do: assert(Nx.to_binary(jit.(t, idx, upd)) == expected)
    end

    test "rows that share an index keep the last row's window" do
      # Rows 0 and 1 both write columns 0..2 of row 0; row 2 writes row 1.
      t = Nx.broadcast(0, {2, 4})
      idx = Nx.tensor([[0], [0], [1]])
      upd = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

      assert Nx.to_flat_list(run(&Nx.indexed_put/3, [t, idx, upd])) ==
               [4, 5, 6, 0, 7, 8, 9, 0]
    end

    test "a window that is not contiguous in the target, with stated values" do
      # `BinaryBackend` writes the update as one chunk, so it cannot state this.
      # The window is 2 x 2 at the start of each {3, 4} slab; the rest stays.
      t = Nx.iota({2, 3, 4}, type: :s32)
      idx = Nx.tensor([[1]])
      upd = Nx.tensor([[[100, 101], [102, 103]]])

      put = Enum.to_list(0..23) |> List.replace_at(12, 100) |> List.replace_at(13, 101)
      put = put |> List.replace_at(16, 102) |> List.replace_at(17, 103)
      assert Nx.to_flat_list(run(&Nx.indexed_put/3, [t, idx, upd])) == put

      add = Enum.to_list(0..23) |> List.replace_at(12, 112) |> List.replace_at(13, 114)
      add = add |> List.replace_at(16, 118) |> List.replace_at(17, 120)
      assert Nx.to_flat_list(run(&Nx.indexed_add/3, [t, idx, upd])) == add
    end
  end

  describe "indexed_add sums" do
    test "small integer types wrap like the reference" do
      t = Nx.tensor([120, 0], type: :s8)
      idx = Nx.tensor([[0], [0], [1]])
      upd = Nx.tensor([5, 5, -3], type: :s8)

      # 120 + 5 + 5 = 130 wraps to -126.
      assert Nx.to_flat_list(run(&Nx.indexed_add/3, [t, idx, upd])) == [-126, -3]

      t = Nx.tensor([250], type: :u8)

      assert Nx.to_flat_list(
               run(&Nx.indexed_add/3, [t, Nx.tensor([[0], [0]]), Nx.tensor([4, 4], type: :u8)])
             ) == [2]
    end

    test "100 000 updates onto 10 targets match an f64 sum within the summation bound" do
      n = 100_000
      key = Nx.Random.key(NxCL.TestSupport.Compare.seed())
      {idx, key} = Nx.Random.randint(key, 0, 10, shape: {n, 1}, type: :s32)
      {upd, _key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {n}, type: :f32)
      target = Nx.broadcast(0.0, {10})

      actual = run(&Nx.indexed_add/3, [target, idx, upd])

      # The reference adds in f64, so its sum is exact enough to judge f32 by.
      exact =
        Nx.indexed_add(Nx.as_type(target, :f64), idx, Nx.as_type(upd, :f64))
        |> Nx.to_flat_list()

      counts =
        for t <- 0..9, do: idx |> Nx.equal(t) |> Nx.sum() |> Nx.to_number()

      # All terms are positive. Adding m positive f32 terms in any order has a
      # relative error of at most m * 2^-24, so that is the tolerance per target.
      for {{a, e}, m} <- Enum.zip(Enum.zip(Nx.to_flat_list(actual), exact), counts) do
        assert abs(a - e) <= m * :math.pow(2, -24) * e,
               "target sum #{a}, exact #{e}, #{m} terms: error #{abs(a - e) / e}"
      end
    end

    test "100 000 integer updates are exact" do
      n = 100_000
      idx = Nx.remainder(Nx.iota({n, 1}, type: :s32) |> Nx.multiply(31), 10)
      upd = Nx.remainder(Nx.iota({n}, type: :s32), 1000)
      target = Nx.broadcast(0, {10})

      assert_same(&Nx.indexed_add/3, [target, idx, upd])
    end
  end

  describe "embedding lookup" do
    test "take of a {1000, 64} table with {32, 128} ids matches the reference" do
      key = Nx.Random.key(NxCL.TestSupport.Compare.seed())
      {table, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {1000, 64}, type: :f32)
      {ids, _key} = Nx.Random.randint(key, 0, 1000, shape: {32, 128}, type: :s32)

      assert_same(&Nx.take(&1, &2, axis: 0), [table, ids])

      assert_same(&Nx.take(&1, &2, axis: 0), [table, Nx.as_type(ids, :u16)])

      out = run(&Nx.take(&1, &2, axis: 0), [table, ids])
      assert out.shape == {32, 128, 64}
    end
  end

  describe "64-bit atomics" do
    test "a device without cl_khr_int64_base_atomics raises the named error" do
      for type <- [{:s, 64}, {:u, 64}, {:f, 64}, {:c, 128}] do
        t = Nx.iota({4}, type: type)
        idx = Nx.tensor([[0], [1]])
        upd = Nx.tensor([1, 2], type: type)

        if capability?(:int64_atomics) do
          # The device has them: the add runs (the differential rows check the sums).
          if type in [{:f, 64}, {:c, 128}] and not capability?(:fp64) do
            assert_raise NxCL.CompilerError, ~r/cl_khr_fp64/, fn ->
              run(&Nx.indexed_add/3, [t, idx, upd])
            end
          else
            out = run(&Nx.indexed_add/3, [t, idx, upd])
            assert out.type == type
            assert Nx.to_binary(out) == Nx.to_binary(Nx.indexed_add(t, idx, upd))
          end
        else
          assert_raise NxCL.CompilerError, ~r/cl_khr_int64_base_atomics/, fn ->
            run(&Nx.indexed_add/3, [t, idx, upd])
          end
        end
      end
    end

    test "gather and indexed_put on 64-bit types need no atomics" do
      t = Nx.iota({4, 2}, type: :s64)
      idx = Nx.tensor([[1], [3]])
      assert_same(&Nx.gather/2, [t, idx])

      assert_same(&Nx.indexed_put/3, [
        t,
        idx,
        Nx.tensor([[7, 8], [9, 10]], type: :s64)
      ])
    end
  end

  describe "zero-size inputs, through the wire" do
    # Nx 1.0 has no zero-size tensors, so each program is written as a wire term.
    setup do
      {:ok, ctx: Native.device_ctx_create(0)}
    end

    defp exec(ctx, wire, buffers) do
      {:ok, program} = Native.compile_program(ctx, wire)
      {:ok, [out]} = Native.eval_program(program, buffers)
      out
    end

    test "a gather with no indices reads back as empty", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)
      table = Native.buffer_write(ctx, NxCL.TestHelpers.floats_to_binary([1.0, 2.0, 3.0]))
      s32 = {:s, 32}

      wire =
        {[{@f32, {3}}, {s32, {0, 1}}], [], [],
         [{:gather, [{:input, 0}, {:input, 1}], @f32, {0}, [0]}], [{:result, 0}]}

      out = exec(ctx, wire, [table, empty])
      assert Native.buffer_read(out, 0) == <<>>
    end

    test "indexed_add and indexed_put with no updates return the target", %{ctx: ctx} do
      empty = Native.buffer_write(ctx, <<>>)
      values = [1.0, 2.0, 3.0]
      s32 = {:s, 32}

      for op <- [:indexed_add, :indexed_put] do
        target = Native.buffer_write(ctx, NxCL.TestHelpers.floats_to_binary(values))

        wire =
          {[{@f32, {3}}, {s32, {0, 1}}, {@f32, {0}}], [], [],
           [{op, [{:input, 0}, {:input, 1}, {:input, 2}], @f32, {3}, [0]}], [{:result, 0}]}

        out = exec(ctx, wire, [target, empty, empty])
        assert out |> Native.buffer_read(12) |> NxCL.TestHelpers.binary_to_floats() == values
      end
    end
  end

  test "the rank cap holds for gather" do
    ctx = Native.device_ctx_create(0)
    shape = List.to_tuple(List.duplicate(1, 9))
    s32 = {:s, 32}

    wire =
      {[{@f32, shape}, {s32, {1, 1}}], [], [],
       [{:gather, [{:input, 0}, {:input, 1}], @f32, List.to_tuple(List.duplicate(1, 9)), [0]}],
       [{:result, 0}]}

    assert {:error, reason} = Native.compile_program(ctx, wire)
    assert reason =~ "rank 9"
  end
end
