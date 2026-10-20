defmodule NxCL.E2E.AxonTest do
  @moduledoc """
  Axon models through the backend: dense, ReLU, sigmoid and softmax layers.
  """
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  defp transfer_params(%Axon.ModelState{} = model_state, backend) do
    new_data =
      model_state.data
      |> Enum.map(fn {layer_name, layer_params} ->
        transferred =
          Enum.map(layer_params, fn {param_name, tensor} ->
            {param_name, Nx.backend_transfer(tensor, backend)}
          end)
          |> Map.new()

        {layer_name, transferred}
      end)
      |> Map.new()

    %{model_state | data: new_data}
  end

  describe "MLP with Dense + ReLU + Softmax" do
    setup do
      model =
        Axon.input("input", shape: {nil, 4})
        |> Axon.dense(8, activation: :relu)
        |> Axon.dense(3, activation: :softmax)

      {init_fn, predict_fn} = Axon.build(model)

      template = Nx.template({1, 4}, :f32)
      params = init_fn.(template, %{})

      %{model: model, predict_fn: predict_fn, params: params}
    end

    test "GPU prediction matches CPU prediction", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

      cpu_result = predict_fn.(params, input)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      gpu_result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-3)
    end

    test "softmax output sums to ~1.0", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      sum = result |> Nx.sum() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-3
    end

    test "all probabilities are non-negative", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      min_val = result |> Nx.reduce_min() |> Nx.to_number()
      assert min_val >= 0.0
    end

    test "batch inference matches CPU", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(99)
      {input, _key} = Nx.Random.uniform(key, shape: {8, 4}, type: :f32)

      cpu_result = predict_fn.(params, input)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      gpu_result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-3)
    end

    test "deterministic across runs", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)

      results =
        for _ <- 1..5 do
          predict_fn.(gpu_params, gpu_input)
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Nx.to_binary()
        end

      first = hd(results)
      for r <- tl(results), do: assert(r == first, "Non-deterministic")
    end
  end

  describe "MLP with Dense + Sigmoid (binary classification)" do
    setup do
      model =
        Axon.input("input", shape: {nil, 8})
        |> Axon.dense(16, activation: :sigmoid)
        |> Axon.dense(8, activation: :sigmoid)
        |> Axon.dense(1, activation: :sigmoid)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({1, 8}, :f32)
      params = init_fn.(template, %{})

      %{predict_fn: predict_fn, params: params}
    end

    test "GPU matches CPU for 3-layer sigmoid network", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 8}, type: :f32)

      cpu_result = predict_fn.(params, input)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      gpu_result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-3)
    end

    test "output is in [0, 1] range", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {4, 8}, type: :f32)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      min_val = result |> Nx.reduce_min() |> Nx.to_number()
      max_val = result |> Nx.reduce_max() |> Nx.to_number()
      assert min_val >= 0.0
      assert max_val <= 1.0
    end
  end

  describe "MLP with Dense + Tanh (regression)" do
    setup do
      model =
        Axon.input("input", shape: {nil, 4})
        |> Axon.dense(16, activation: :tanh)
        |> Axon.dense(8, activation: :tanh)
        |> Axon.dense(1)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({1, 4}, :f32)
      params = init_fn.(template, %{})

      %{predict_fn: predict_fn, params: params}
    end

    test "GPU matches CPU for tanh regression network", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 4}, type: :f32)

      cpu_result = predict_fn.(params, input)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      gpu_result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-3)
    end
  end

  describe "deeper network (stress test)" do
    setup do
      model =
        Axon.input("input", shape: {nil, 16})
        |> Axon.dense(32, activation: :relu)
        |> Axon.dense(32, activation: :relu)
        |> Axon.dense(16, activation: :relu)
        |> Axon.dense(8, activation: :relu)
        |> Axon.dense(4, activation: :softmax)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({1, 16}, :f32)
      params = init_fn.(template, %{})

      %{predict_fn: predict_fn, params: params}
    end

    test "5-layer network GPU matches CPU", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 16}, type: :f32)

      cpu_result = predict_fn.(params, input)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      gpu_result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-2)
    end

    test "softmax sums to 1.0 on 5-layer network", %{predict_fn: predict_fn, params: params} do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {1, 16}, type: :f32)

      gpu_params = transfer_params(params, NxCL.Backend)
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)
      result = predict_fn.(gpu_params, gpu_input) |> Nx.backend_transfer(Nx.BinaryBackend)

      sum = result |> Nx.sum() |> Nx.to_number()
      assert_in_delta sum, 1.0, 1.0e-2
    end
  end
end
