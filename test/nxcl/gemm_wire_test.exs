defmodule NxCL.GemmWireTest do
  @moduledoc """
  `batched_gemm` and `conv` through the native wire: sizes `Nx.BinaryBackend` cannot
  reach in reasonable time (answers stated), `K == 0` and zero-size products (Nx
  1.0 builds no zero-size tensors), and the `NXCL_GEMM_TILE` override.
  """
  use ExUnit.Case, async: false

  alias NxCL.Native

  @moduletag :opencl
  @moduletag timeout: 600_000

  @f32 {:f, 32}

  defp upload(ctx, tensor), do: Native.buffer_write(ctx, Nx.to_binary(tensor))

  defp gemm_wire(b, m, n, k, type \\ @f32) do
    {[{type, {b, m, k}}, {type, {b, k, n}}], [], [],
     [{:batched_gemm, [{:input, 0}, {:input, 1}], type, {b, m, n}, [b, m, n, k]}], [{:result, 0}]}
  end

  defp run_gemm(ctx, wire, a, b, type \\ @f32) do
    {:ok, program} = Native.compile_program(ctx, wire)
    {:ok, [out]} = Native.eval_program(program, [upload(ctx, a), upload(ctx, b)])
    {_, _, _, [{_, _, _, shape, _}], _} = wire
    size = shape |> Tuple.product() |> Kernel.*(div(elem(type, 1), 8))
    out |> Native.buffer_read(size) |> Nx.from_binary(type) |> Nx.reshape(shape)
  end

  # The override is read from the process environment, and the BEAM's `System.put_env`
  # does not reach native code, so each override runs in a child VM that starts with it.
  @child """
  alias NxCL.Native
  ctx = Native.device_ctx_create(0)
  wire = {[{{:f, 32}, {1, 37, 29}}, {{:f, 32}, {1, 29, 41}}], [], [],
          [{:batched_gemm, [{:input, 0}, {:input, 1}], {:f, 32}, {1, 37, 41}, [1, 37, 41, 29]}],
          [{:result, 0}]}
  up = fn t -> Native.buffer_write(ctx, Nx.to_binary(t)) end
  a = Nx.iota({1, 37, 29}, type: :f32) |> Nx.divide(7)
  b = Nx.iota({1, 29, 41}, type: :f32) |> Nx.divide(5)
  case Native.compile_program(ctx, wire) do
    {:ok, p} ->
      {:ok, [out]} = Native.eval_program(p, [up.(a), up.(b)])
      bin = Native.buffer_read(out, 37 * 41 * 4)
      IO.puts("RESULT " <> Base.encode64(bin))

    {:error, reason} ->
      IO.puts("ERROR " <> Base.encode64(reason))
  end
  """

  defp child_result(tile) do
    env = [{"MIX_ENV", "test"}, {"NXCL_GEMM_TILE", tile}]

    args = [
      "run",
      "--no-compile",
      "--no-start",
      "-e",
      "Application.ensure_all_started(:nx); " <> @child
    ]

    {out, 0} = System.cmd("mix", args, env: env, stderr_to_stdout: true)

    case Regex.run(~r/RESULT (\S+)/, out) do
      [_, b64] ->
        b64 |> Base.decode64!() |> Nx.from_binary(:f32)

      nil ->
        [_, b64] = Regex.run(~r/ERROR (\S+)/, out)
        {:error, Base.decode64!(b64)}
    end
  end

  describe "large products with stated answers" do
    test "ones times ones is K, on sizes that are not multiples of any tile" do
      ctx = Native.device_ctx_create(0)

      for {m, n, k} <- [{257, 259, 261}, {1, 1031, 1500}, {1031, 1, 1500}, {1, 1, 4097}] do
        out =
          run_gemm(
            ctx,
            gemm_wire(1, m, n, k),
            Nx.broadcast(1.0, {1, m, k}),
            Nx.broadcast(1.0, {1, k, n})
          )

        assert Nx.to_flat_list(out) == List.duplicate(k * 1.0, m * n), inspect({m, n, k})
      end
    end

    test "a column index pattern: C[i][j] = K * j" do
      ctx = Native.device_ctx_create(0)
      {m, n, k} = {300, 130, 70}
      a = Nx.broadcast(1.0, {2, m, k})
      b = Nx.iota({2, k, n}, axis: 2, type: :f32)
      out = run_gemm(ctx, gemm_wire(2, m, n, k), a, b)
      expected = Nx.iota({2, m, n}, axis: 2, type: :f32) |> Nx.multiply(k)
      assert Nx.to_flat_list(out) == Nx.to_flat_list(expected)
    end

    test "a row index pattern with integers: C[i][j] = K * i, wrapping nowhere" do
      ctx = Native.device_ctx_create(0)
      {m, n, k} = {70, 33, 64}
      a = Nx.iota({1, m, k}, axis: 1, type: :s32)
      b = Nx.broadcast(Nx.tensor(1, type: :s32), {1, k, n})
      out = run_gemm(ctx, gemm_wire(1, m, n, k, {:s, 32}), a, b, {:s, 32})

      assert Nx.to_flat_list(out) ==
               Nx.to_flat_list(Nx.iota({1, m, n}, axis: 1, type: :s32) |> Nx.multiply(k))
    end

    test "s32 sums above 2^24 stay exact (they do not pass through f32)" do
      ctx = Native.device_ctx_create(0)
      k = 2048
      a = Nx.broadcast(Nx.tensor(100_001, type: :s32), {1, 5, k})
      b = Nx.broadcast(Nx.tensor(3, type: :s32), {1, k, 7})
      out = run_gemm(ctx, gemm_wire(1, 5, 7, k, {:s, 32}), a, b, {:s, 32})
      assert Enum.uniq(Nx.to_flat_list(out)) == [100_001 * 3 * k]
    end
  end

  describe "K == 0 and zero-size products" do
    test "K == 0 gives zeros" do
      ctx = Native.device_ctx_create(0)
      {:ok, program} = Native.compile_program(ctx, gemm_wire(2, 3, 4, 0))

      {:ok, [out]} =
        Native.eval_program(program, [Native.buffer_create(ctx, 0), Native.buffer_create(ctx, 0)])

      assert out |> Native.buffer_read(2 * 3 * 4 * 4) |> Nx.from_binary(@f32) |> Nx.to_flat_list() ==
               List.duplicate(0.0, 24)
    end

    test "K == 0 gives zeros for every dtype" do
      ctx = Native.device_ctx_create(0)

      for type <- [{:s, 8}, {:u, 16}, {:s, 32}, {:u, 64}, {:bf, 16}, {:f, 32}, {:c, 64}] do
        {:ok, program} = Native.compile_program(ctx, gemm_wire(1, 2, 3, 0, type))
        empty = Native.buffer_create(ctx, 0)
        {:ok, [out]} = Native.eval_program(program, [empty, empty])
        bytes = 6 * div(elem(type, 1), 8)

        assert out |> Native.buffer_read(bytes) |> :binary.bin_to_list() |> Enum.uniq() == [0],
               inspect(type)
      end
    end

    test "an empty output evaluates without error" do
      ctx = Native.device_ctx_create(0)

      for {b, m, n, k} <- [{1, 0, 4, 3}, {1, 4, 0, 3}, {0, 4, 4, 3}] do
        {:ok, program} = Native.compile_program(ctx, gemm_wire(b, m, n, k))
        empty = Native.buffer_create(ctx, 0)
        {:ok, [_out]} = Native.eval_program(program, [empty, empty])
      end
    end

    test "a size that disagrees with the shapes is a compile error" do
      ctx = Native.device_ctx_create(0)
      {ins, c, k, [{op, refs, type, _shape, attrs}], outs} = gemm_wire(1, 2, 3, 4)
      wire = {ins, c, k, [{op, refs, type, {1, 2, 4}, attrs}], outs}
      assert {:error, reason} = Native.compile_program(ctx, wire)
      assert reason =~ "do not match"
    end
  end

  describe "NXCL_GEMM_TILE" do
    test "a valid override gives the same answer as the table" do
      a = Nx.iota({1, 37, 29}, type: :f32) |> Nx.divide(7)
      b = Nx.iota({1, 29, 41}, type: :f32) |> Nx.divide(5)
      ctx = Native.device_ctx_create(0)
      default = run_gemm(ctx, gemm_wire(1, 37, 41, 29), a, b) |> Nx.reshape({:auto})

      for tile <- ["16,16,8,2,2", "8,8,4,1,1"] do
        assert_all_close(child_result(tile), default)
      end
    end

    test "an override that is not valid is an error" do
      assert {:error, message} = child_result("3,3,3,2,2")
      assert message =~ "NXCL_GEMM_TILE"
      assert {:error, message} = child_result("not a tile")
      assert message =~ "NXCL_GEMM_TILE"
    end
  end

  describe "conv with stated answers" do
    test "all ones with same padding count the cells under the kernel" do
      {c, h, w, o} = {16, 40, 36, 8}
      x = Nx.broadcast(1.0, {2, c, h, w})
      k = Nx.broadcast(1.0, {o, c, 3, 3})
      out = Nx.Defn.jit(&Nx.conv(&1, &2, padding: :same), compiler: NxCL.Compiler).(x, k)
      assert out.shape == {2, o, h, w}

      # Corner windows hold 4 cells, edge windows 6, interior windows 9; each adds `c` ones.
      assert Nx.to_number(out[[0, 0, 0, 0]]) == 4.0 * c
      assert Nx.to_number(out[[1, o - 1, 0, 5]]) == 6.0 * c
      assert Nx.to_number(out[[0, 3, 20, 18]]) == 9.0 * c
      assert Nx.to_number(out[[1, 2, h - 1, w - 1]]) == 4.0 * c
      assert Nx.to_number(Nx.sum(out)) == 2 * o * c * (3 * h - 2) * (3 * w - 2)
    end

    test "a depthwise conv with the kernel picks each channel on its own" do
      x =
        Nx.concatenate([Nx.broadcast(1.0, {1, 1, 6, 6}), Nx.broadcast(10.0, {1, 1, 6, 6})],
          axis: 1
        )

      k =
        Nx.concatenate([Nx.broadcast(2.0, {1, 1, 3, 3}), Nx.broadcast(3.0, {1, 1, 3, 3})],
          axis: 0
        )

      out = Nx.Defn.jit(&Nx.conv(&1, &2, feature_group_size: 2), compiler: NxCL.Compiler).(x, k)
      assert out.shape == {1, 2, 4, 4}
      assert Enum.uniq(Nx.to_flat_list(out[[0, 0]])) == [18.0]
      assert Enum.uniq(Nx.to_flat_list(out[[0, 1]])) == [270.0]
    end
  end

  defp assert_all_close(a, b) do
    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(a, b)))) < 1.0e-3
  end
end
