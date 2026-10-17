# MLP inference: one dense layer, three layers, and batch sizes of a two-layer net.
#
# Usage: mix run benchmarks/mlp_inference.exs

Code.require_file("bench_helper.exs", __DIR__)
Code.require_file("models.exs", __DIR__)

r = &BenchHelper.rand/2

mlp2 = fn x, w1, b1, w2, b2 ->
  x |> Models.dense(w1, b1) |> Models.relu() |> Models.dense(w2, b2)
end

mlp3 = fn x, w1, b1, w2, b2, w3, b3 ->
  x
  |> Models.dense(w1, b1)
  |> Models.relu()
  |> Models.dense(w2, b2)
  |> Models.relu()
  |> Models.dense(w3, b3)
end

two = fn bs ->
  [
    r.({bs, 784}, 1.0),
    r.({784, 256}, 0.1),
    r.({1, 256}, 0.1),
    r.({256, 10}, 0.1),
    r.({1, 10}, 0.1)
  ]
end

BenchHelper.run_cases(
  "MLP inference",
  [
    {"dense+relu batch=1 (784->256)", fn x, w, b -> x |> Models.dense(w, b) |> Models.relu() end,
     [r.({1, 784}, 1.0), r.({784, 256}, 0.1), r.({1, 256}, 0.1)]},
    {"3-layer batch=16 (784->256->128->10)", mlp3,
     [
       r.({16, 784}, 1.0),
       r.({784, 256}, 0.1),
       r.({1, 256}, 0.1),
       r.({256, 128}, 0.1),
       r.({1, 128}, 0.1),
       r.({128, 10}, 0.1),
       r.({1, 10}, 0.1)
     ]}
  ] ++ for(bs <- [1, 4, 16, 64], do: {"2-layer batch=#{bs} (784->256->10)", mlp2, two.(bs)})
)
