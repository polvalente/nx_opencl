# Model functions shared by the benchmark scripts and the profiler.

defmodule Models do
  @moduledoc false

  def relu(x), do: Nx.max(x, 0)
  def dense(x, w, b), do: Nx.add(Nx.dot(x, w), b)

  def softmax(z) do
    m = Nx.reduce_max(z, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(z, m))
    Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
  end

  def mlp_forward(x, w1, b1, w2, b2, w3, b3) do
    x |> dense(w1, b1) |> relu() |> dense(w2, b2) |> relu() |> dense(w3, b3)
  end

  def mlp_train(x, y, w1, b1, w2, b2, w3, b3) do
    loss = fn {w1, b1, w2, b2, w3, b3} ->
      logits = mlp_forward(x, w1, b1, w2, b2, w3, b3)
      Nx.mean(Nx.pow(Nx.subtract(logits, y), 2))
    end

    {l, {gw1, gb1, gw2, gb2, gw3, gb3}} = Nx.Defn.value_and_grad({w1, b1, w2, b2, w3, b3}, loss)
    lr = 0.01

    {l, Nx.subtract(w1, Nx.multiply(gw1, lr)), Nx.subtract(b1, Nx.multiply(gb1, lr)),
     Nx.subtract(w2, Nx.multiply(gw2, lr)), Nx.subtract(b2, Nx.multiply(gb2, lr)),
     Nx.subtract(w3, Nx.multiply(gw3, lr)), Nx.subtract(b3, Nx.multiply(gb3, lr))}
  end

  def cnn(x, k1, b1, k2, b2, w, b) do
    h = x |> Nx.conv(k1, padding: :same) |> then(&Nx.add(&1, Nx.reshape(b1, {1, :auto, 1, 1})))
    h = relu(h)
    h = Nx.window_sum(h, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) |> Nx.divide(4)
    h = h |> Nx.conv(k2, padding: :same) |> then(&Nx.add(&1, Nx.reshape(b2, {1, :auto, 1, 1})))
    h = relu(h)
    h = Nx.window_sum(h, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) |> Nx.divide(4)
    {n, _, _, _} = Nx.shape(h)
    h |> Nx.reshape({n, :auto}) |> Nx.dot(w) |> Nx.add(b) |> softmax()
  end

  def layer_norm(x, g, b) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    c = Nx.subtract(x, mean)
    var = Nx.mean(Nx.multiply(c, c), axes: [-1], keep_axes: true)
    c |> Nx.multiply(Nx.rsqrt(Nx.add(var, 1.0e-5))) |> Nx.multiply(g) |> Nx.add(b)
  end

  def gelu(x) do
    inner = Nx.multiply(0.7978845608, Nx.add(x, Nx.multiply(0.044715, Nx.pow(x, 3))))
    Nx.multiply(Nx.multiply(0.5, x), Nx.add(1.0, Nx.tanh(inner)))
  end

  def transformer(x, g1, b1, wq, wk, wv, wo, g2, b2, w1, c1, w2, c2) do
    {_b, _t, d} = Nx.shape(x)
    h = layer_norm(x, g1, b1)
    q = Nx.dot(h, [2], wq, [0])
    k = Nx.dot(h, [2], wk, [0])
    v = Nx.dot(h, [2], wv, [0])
    scores = Nx.dot(q, [2], [0], k, [2], [0]) |> Nx.divide(:math.sqrt(d))
    att = Nx.dot(softmax(scores), [2], [0], v, [1], [0])
    x = Nx.add(x, Nx.dot(att, [2], wo, [0]))
    h = layer_norm(x, g2, b2)
    f = h |> Nx.dot([2], w1, [0]) |> Nx.add(c1) |> gelu() |> Nx.dot([2], w2, [0]) |> Nx.add(c2)
    Nx.add(x, f)
  end

  # {label, fun, args} for the four models. Fixed sizes, so runs are comparable.
  def cases do
    r = &BenchHelper.rand/2
    {n, d} = {128, 256}

    mlp_params = [
      r.({d, d}, 0.1),
      r.({d}, 0.1),
      r.({d, d}, 0.1),
      r.({d}, 0.1),
      r.({d, 10}, 0.1),
      r.({10}, 0.1)
    ]

    {bt, tt, td} = {4, 64, 128}

    [
      {"MLP forward (128x256)", &mlp_forward/7, [r.({n, d}, 1.0) | mlp_params]},
      {"MLP training step", &mlp_train/8, [r.({n, d}, 1.0), r.({n, 10}, 1.0) | mlp_params]},
      {"CNN forward (16x3x32x32)", &cnn/7,
       [
         r.({16, 3, 32, 32}, 1.0),
         r.({8, 3, 3, 3}, 0.1),
         r.({8}, 0.1),
         r.({16, 8, 3, 3}, 0.1),
         r.({16}, 0.1),
         r.({16 * 8 * 8, 10}, 0.1),
         r.({10}, 0.1)
       ]},
      {"Transformer block (4x64x128)", &transformer/13,
       [
         r.({bt, tt, td}, 1.0),
         Nx.broadcast(1.0, {td}),
         r.({td}, 0.1),
         r.({td, td}, 0.1),
         r.({td, td}, 0.1),
         r.({td, td}, 0.1),
         r.({td, td}, 0.1),
         Nx.broadcast(1.0, {td}),
         r.({td}, 0.1),
         r.({td, 4 * td}, 0.1),
         r.({4 * td}, 0.1),
         r.({4 * td, td}, 0.1),
         r.({td}, 0.1)
       ]}
    ]
  end
end
