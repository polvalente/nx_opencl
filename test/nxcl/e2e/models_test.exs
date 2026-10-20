defmodule NxCL.E2E.ModelsTest do
  @moduledoc """
  Three Axon models through `NxCL.Compiler`, each compared with the evaluator on
  `Nx.BinaryBackend` (`NxCL.TestSupport.Compare` also runs the eager backend):

    * an MLP: forward pass and one training step (loss, gradient, SGD update);
    * a small CNN: forward pass and one training step;
    * a transformer encoder block (embedding lookup, attention, layer norm, GELU,
      residuals): forward pass and one training step.

  `f32` tolerances: `1.0e-4` for a forward pass and `1.0e-3` for a step, which adds
  a gradient and an update. `NXCL_MODELS_REPORT=1` prints the largest absolute error
  of each row. Axon is a test-only dependency: the last test checks `mix.exs` and
  the library source.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  @moduletag :opencl
  @moduletag timeout: 600_000

  @forward [rtol: 1.0e-4, atol: 1.0e-4]
  @step [rtol: 1.0e-3, atol: 1.0e-3]

  defp mlp do
    Axon.input("x", shape: {nil, 6})
    |> Axon.dense(8, activation: :relu)
    |> Axon.dense(3)
  end

  defp cnn do
    Axon.input("x", shape: {nil, 8, 8, 1})
    |> Axon.conv(4, kernel_size: 3, activation: :relu)
    |> Axon.max_pool(kernel_size: 2)
    |> Axon.flatten()
    |> Axon.dense(3)
  end

  @vocab 16
  @width 8
  @tokens 5

  defp transformer do
    ids = Axon.input("x", shape: {nil, @tokens})
    x = Axon.embedding(ids, @vocab, @width)

    q = Axon.dense(x, @width, name: "q")
    k = Axon.dense(x, @width, name: "k")
    v = Axon.dense(x, @width, name: "v")

    attended =
      Axon.layer(
        fn q, k, v, _opts ->
          scores = Nx.dot(q, [2], [0], k, [2], [0]) |> Nx.divide(Nx.sqrt(@width))
          weights = Axon.Activations.softmax(scores, axis: -1)
          Nx.dot(weights, [2], [0], v, [1], [0])
        end,
        [q, k, v],
        name: "attention"
      )
      |> Axon.dense(@width, name: "out")

    h = x |> Axon.add(attended) |> Axon.layer_norm(name: "norm1")

    ffn = h |> Axon.dense(16, name: "ffn1") |> Axon.gelu() |> Axon.dense(@width, name: "ffn2")

    h
    |> Axon.add(ffn)
    |> Axon.layer_norm(name: "norm2")
    |> Axon.nx(&Nx.mean(&1, axes: [1]))
    |> Axon.dense(3, name: "head")
  end

  defp random(shape, seed, opts \\ []) do
    key = Nx.Random.key(seed)

    {t, _} =
      Nx.Random.uniform(key, opts[:min] || -1.0, opts[:max] || 1.0, shape: shape, type: :f32)

    Nx.backend_copy(t, Nx.BinaryBackend)
  end

  defp tokens(batch, seed) do
    key = Nx.Random.key(seed)
    {t, _} = Nx.Random.randint(key, 0, @vocab, shape: {batch, @tokens}, type: :s32)
    Nx.backend_copy(t, Nx.BinaryBackend)
  end

  defp labels(batch) do
    Nx.tensor(
      for(i <- 0..(batch - 1), do: Enum.map(0..2, &if(&1 == rem(i, 3), do: 1.0, else: 0.0))),
      backend: Nx.BinaryBackend
    )
  end

  defp params(model, input_shape, type \\ :f32) do
    {init_fn, predict_fn} = Axon.build(model)

    state =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        init_fn.(%{"x" => Nx.template(input_shape, type)}, Axon.ModelState.empty())
      end)

    {state, predict_fn}
  end

  defp cross_entropy(labels, logits) do
    shifted = Nx.subtract(logits, Nx.reduce_max(logits, axes: [1], keep_axes: true))
    log_probs = Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [1], keep_axes: true)))
    Nx.negate(Nx.mean(Nx.sum(Nx.multiply(labels, log_probs), axes: [1])))
  end

  defp step_fun(predict_fn, lr) do
    fn state, x, labels ->
      {loss, grads} =
        Nx.Defn.value_and_grad(state, fn st ->
          cross_entropy(labels, predict_fn.(st, %{"x" => x}))
        end)

      {updated, []} =
        Nx.Defn.Composite.traverse(state, Nx.Defn.Composite.flatten_list([grads]), fn p,
                                                                                      [g | rest] ->
          {Nx.subtract(p, Nx.multiply(lr, g)), rest}
        end)

      {loss, updated.data}
    end
  end

  defp forward_fun(predict_fn), do: fn state, x -> predict_fn.(state, %{"x" => x}) end

  defp assert_model(name, fun, args, opts) do
    assert_same(fun, args, opts)

    if System.get_env("NXCL_MODELS_REPORT") == "1" do
      expected = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)
      actual = apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), args)

      error =
        Enum.zip(
          Nx.Defn.Composite.flatten_list([expected]),
          Nx.Defn.Composite.flatten_list([actual])
        )
        |> Enum.map(fn {e, a} ->
          e
          |> Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend))
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()
        end)
        |> Enum.max()

      IO.puts("#{name}: max abs error #{error}")
    end
  end

  describe "MLP" do
    test "forward pass" do
      {state, predict_fn} = params(mlp(), {1, 6})

      assert_model(
        :model_mlp_forward,
        forward_fun(predict_fn),
        [state, random({4, 6}, 1)],
        @forward
      )
    end

    test "training step" do
      {state, predict_fn} = params(mlp(), {1, 6})
      args = [state, random({4, 6}, 2), labels(4)]
      assert_model(:model_mlp_step, step_fun(predict_fn, 0.1), args, @step)
    end
  end

  describe "CNN" do
    test "forward pass" do
      {state, predict_fn} = params(cnn(), {1, 8, 8, 1})
      args = [state, random({2, 8, 8, 1}, 3)]
      assert_model(:model_cnn_forward, forward_fun(predict_fn), args, @forward)
    end

    test "training step" do
      {state, predict_fn} = params(cnn(), {1, 8, 8, 1})
      args = [state, random({3, 8, 8, 1}, 4), labels(3)]
      assert_model(:model_cnn_step, step_fun(predict_fn, 0.05), args, @step)
    end
  end

  describe "transformer encoder block" do
    test "forward pass" do
      {state, predict_fn} = params(transformer(), {1, @tokens}, :s32)
      args = [state, tokens(3, 5)]
      assert_model(:model_transformer_forward, forward_fun(predict_fn), args, @forward)
    end

    test "training step" do
      {state, predict_fn} = params(transformer(), {1, @tokens}, :s32)
      args = [state, tokens(3, 6), labels(3)]
      assert_model(:model_transformer_step, step_fun(predict_fn, 0.05), args, @step)
    end
  end

  describe "README example" do
    test "an Axon model runs through Nx.Defn.jit with NxCL.Compiler" do
      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(128, activation: :relu)
        |> Axon.dense(10, activation: :softmax)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 784}, :f32), Axon.ModelState.empty())
      input = random({2, 784}, 7)

      predict = Nx.Defn.jit(predict_fn, compiler: NxCL.Compiler)
      output = predict.(params, input)
      expected = predict_fn.(params, input)

      assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(output, expected)))) < 1.0e-5
      assert_in_delta Nx.to_number(Nx.sum(output)), 2.0, 1.0e-4
    end
  end

  describe "dependencies" do
    test "Axon is a test-only dependency" do
      deps = Mix.Project.config()[:deps]
      assert {:axon, _version, opts} = List.keyfind(deps, :axon, 0)
      assert opts[:only] == [:test]

      users =
        for path <- Path.wildcard("lib/**/*.ex"), File.read!(path) =~ ~r/\bAxon\./, do: path

      assert users == []
    end
  end
end
