defmodule NxCL.E2E.MLPTest do
  @moduledoc """
  A simple MLP run repeatedly on the NxCL backend. The comparison with the CPU
  backend is in `NxCL.E2E.ModelsTest`.
  """
  use ExUnit.Case, async: false

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

  test "MLP is deterministic (same result every time)" do
    {params_key, input_key} = {Nx.Random.key(42), Nx.Random.key(43)}
    params = build_params(params_key)
    {input, _key} = Nx.Random.uniform(input_key, shape: {1, 4}, type: :f32)

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
