defmodule NxCL.E2E.MLPTest do
  @moduledoc """
  A simple MLP on the NxCL backend against the CPU backend.
  """
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  defp run_mlp(params, input) do
    h = Nx.dot(input, params.w1)
    h = Nx.add(h, params.b1)
    h = Nx.max(h, Nx.tensor(0.0))

    out = Nx.dot(h, params.w2)
    Nx.add(out, params.b2)
  end

  defp build_params(key) do
    {w1, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {4, 8}, type: :f32)
    {b1, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {1, 8}, type: :f32)
    {w2, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {8, 3}, type: :f32)
    {b2, _key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {1, 3}, type: :f32)
    %{w1: w1, b1: b1, w2: w2, b2: b2}
  end

  defp transfer_params_to_gpu(params) do
    Map.new(params, fn {k, v} -> {k, Nx.backend_transfer(v, NxCL.Backend)} end)
  end

  test "2-layer MLP matches CPU backend" do
    key = Nx.Random.key(42)
    params = build_params(key)
    {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

    cpu_result = run_mlp(params, input)

    gpu_params = transfer_params_to_gpu(params)
    gpu_input = Nx.backend_transfer(input, NxCL.Backend)
    gpu_result = run_mlp(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

    assert_tensors_close(gpu_result, cpu_result, 1.0e-4)
  end

  test "MLP with batch of inputs" do
    key = Nx.Random.key(99)
    params = build_params(key)
    {input, _key} = Nx.Random.uniform(key, shape: {4, 4}, type: :f32)

    cpu_result = run_mlp(params, input)

    gpu_params = transfer_params_to_gpu(params)
    gpu_input = Nx.backend_transfer(input, NxCL.Backend)
    gpu_result = run_mlp(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

    assert_tensors_close(gpu_result, cpu_result, 1.0e-4)
  end

  test "MLP is deterministic (same result every time)" do
    key = Nx.Random.key(42)
    params = build_params(key)
    {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

    gpu_params = transfer_params_to_gpu(params)
    gpu_input = Nx.backend_transfer(input, NxCL.Backend)

    results =
      for _ <- 1..10 do
        run_mlp(gpu_params, gpu_input)
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.to_binary()
      end

    first = hd(results)

    for r <- tl(results) do
      assert r == first, "Non-deterministic results detected"
    end
  end
end
