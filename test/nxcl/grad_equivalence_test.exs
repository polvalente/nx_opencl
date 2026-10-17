defmodule NxCL.GradEquivalenceTest do
  @moduledoc """
  `Nx.Defn.grad` and `value_and_grad` under `NxCL.Compiler` and under
  `NxCL.Backend` match the evaluator on `Nx.BinaryBackend`, after
  `emlx/test/emlx/grad_equivalence_test.exs`. Each function runs through
  `NxCL.Parity` (so the no-fallback tripwire watches it too), in `f32`.

  The functions: polynomials, an elementwise chain, softmax cross-entropy, layer
  norm, a matrix product, a convolution, a pooling window, a reduction and sort,
  a `cond` and a `while`.
  """
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxCL.Parity

  @moduletag :opencl
  @moduletag timeout: 600_000

  # Unique, zero-free values: no ties for argmax or sort, no zero in a quotient.
  defp data(shape, scale \\ 0.37) do
    size = Tuple.product(shape)

    for(i <- 0..(size - 1), do: (i - size / 2) * scale + 0.6)
    |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end

  defp assert_grad(name, fun, args, opts \\ []) do
    opts = Keyword.merge([record: false, rtol: 1.0e-4, atol: 1.0e-4], opts)
    assert_parity(name, fun, args, opts)
  end

  # -- Smooth elementwise --------------------------------------------------------

  defn(polynomial(x), do: Nx.sum(x * x * x + 2 * x * x - 5 * x + 3))
  defn(polynomial_grad(x), do: grad(x, &polynomial/1))

  defn(chain(x), do: Nx.sum(Nx.sin(x) * Nx.cos(x) + Nx.exp(x * 0.1) - Nx.tanh(x)))
  defn(chain_grad(x), do: grad(x, &chain/1))

  defn(division(x, y), do: Nx.sum(x / (y * y + 1.0) + Nx.sqrt(x * x + 1.0)))
  defn(division_grad(x, y), do: grad({x, y}, fn {x, y} -> division(x, y) end))

  # -- Softmax cross-entropy and layer norm ---------------------------------------

  defn softmax_xent(logits, labels) do
    shifted = logits - Nx.reduce_max(logits, axes: [1], keep_axes: true)
    log_probs = shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [1], keep_axes: true))
    -Nx.mean(Nx.sum(labels * log_probs, axes: [1]))
  end

  defn(softmax_xent_grad(logits, labels),
    do: grad(logits, fn logits -> softmax_xent(logits, labels) end)
  )

  defn layer_norm(x, gamma, beta) do
    mean = Nx.mean(x, axes: [1], keep_axes: true)
    centered = x - mean
    var = Nx.mean(centered * centered, axes: [1], keep_axes: true)
    Nx.sum(centered * Nx.rsqrt(var + 1.0e-5) * gamma + beta)
  end

  defn(layer_norm_grad(x, gamma, beta),
    do: grad({x, gamma, beta}, fn {x, gamma, beta} -> layer_norm(x, gamma, beta) end)
  )

  # -- Dot, conv, windows, reductions ---------------------------------------------

  defn(dot_loss(a, b), do: Nx.sum(Nx.tanh(Nx.dot(a, b))))
  defn(dot_grad(a, b), do: grad({a, b}, fn {a, b} -> dot_loss(a, b) end))

  defn(conv_loss(x, k), do: Nx.sum(Nx.sin(Nx.conv(x, k, padding: :same))))
  defn(conv_grad(x, k), do: grad({x, k}, fn {x, k} -> conv_loss(x, k) end))

  defn(window_loss(x), do: Nx.sum(Nx.window_max(x, {1, 2, 2}, strides: [1, 2, 2]) ** 2))
  defn(window_grad(x), do: grad(x, &window_loss/1))

  defn(window_sum_loss(x), do: Nx.sum(Nx.window_sum(x, {1, 2, 2}, padding: :same) ** 2))
  defn(window_sum_grad(x), do: grad(x, &window_sum_loss/1))

  defn(reduce_loss(x),
    do: Nx.reduce_max(x, axes: [1]) |> Nx.multiply(Nx.sum(x, axes: [1])) |> Nx.sum()
  )

  defn(reduce_grad(x), do: grad(x, &reduce_loss/1))

  defn(sort_loss(x), do: Nx.sum(Nx.sort(x, axis: 1) * Nx.iota({1, 5}, type: :f32)))
  defn(sort_grad(x), do: grad(x, &sort_loss/1))

  defn(cumsum_loss(x), do: Nx.sum(Nx.cumulative_sum(x, axis: 1) ** 2))
  defn(cumsum_grad(x), do: grad(x, &cumsum_loss/1))

  # -- Control flow ---------------------------------------------------------------

  defn cond_loss(x) do
    s = Nx.sum(x)

    if s > 0 do
      Nx.sum(x * x)
    else
      Nx.sum(Nx.exp(x))
    end
  end

  defn(cond_grad(x), do: grad(x, &cond_loss/1))

  defn while_loss(x) do
    {_i, acc} =
      while {i = 0, acc = x}, i < 3 do
        {i + 1, Nx.sin(acc) + acc * 0.5}
      end

    Nx.sum(acc)
  end

  defn(while_grad(x), do: grad(x, &while_loss/1))

  # -- Rows -----------------------------------------------------------------------

  test "polynomial" do
    for shape <- [{}, {4}, {2, 3}],
        do: assert_grad(:grad_polynomial, &polynomial_grad/1, [data(shape)])
  end

  test "elementwise chain" do
    for shape <- [{4}, {2, 3}, {2, 2, 3}],
        do: assert_grad(:grad_chain, &chain_grad/1, [data(shape)])
  end

  test "division and sqrt" do
    assert_grad(:grad_division, &division_grad/2, [data({2, 3}), data({2, 3}, 0.21)])
  end

  test "softmax cross-entropy" do
    labels =
      Nx.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], backend: Nx.BinaryBackend)

    assert_grad(:grad_softmax_xent, &softmax_xent_grad/2, [data({3, 3}), labels])
    assert_grad(:grad_softmax_xent, &softmax_xent_grad/2, [data({3, 3}, 1.7), labels])
  end

  test "layer norm" do
    assert_grad(:grad_layer_norm, &layer_norm_grad/3, [
      data({3, 5}),
      data({5}, 0.2),
      data({5}, 0.1)
    ])
  end

  test "dot" do
    assert_grad(:grad_dot, &dot_grad/2, [data({3, 4}, 0.1), data({4, 2}, 0.2)])
  end

  test "conv" do
    assert_grad(:grad_conv, &conv_grad/2, [data({1, 2, 5, 5}, 0.05), data({3, 2, 3, 3}, 0.1)])
  end

  test "window_max and window_sum" do
    assert_grad(:grad_window_max, &window_grad/1, [data({1, 4, 4})])
    assert_grad(:grad_window_sum, &window_sum_grad/1, [data({1, 4, 4}, 0.1)])
  end

  test "reduce_max, sort and cumulative_sum" do
    assert_grad(:grad_reduce, &reduce_grad/1, [data({3, 5})])

    assert_grad(:grad_sort, &sort_grad/1, [
      Nx.tensor([[3.0, 1.0, 4.0, 1.5, 9.0]], backend: Nx.BinaryBackend)
    ])

    assert_grad(:grad_cumsum, &cumsum_grad/1, [data({2, 5}, 0.1)])
  end

  test "cond takes either branch" do
    assert_grad(:grad_cond, &cond_grad/1, [data({4}, 0.5)])
    assert_grad(:grad_cond, &cond_grad/1, [Nx.negate(data({4}, 0.5))])
  end

  test "while" do
    assert_grad(:grad_while, &while_grad/1, [data({4})])
  end
end
