# Reductions and softmax.
#
# Usage: mix run benchmarks/reduction.exs

Code.require_file("bench_helper.exs", __DIR__)
Code.require_file("models.exs", __DIR__)

r = &BenchHelper.rand/2

BenchHelper.run_cases(
  "Reductions",
  for n <- [1_000, 10_000, 100_000, 1_000_000] do
    {"sum n=#{n}", &Nx.sum/1, [r.({n}, 1.0)]}
  end ++
    for n <- [1_000, 100_000, 1_000_000] do
      {"reduce_max n=#{n}", &Nx.reduce_max/1, [r.({n}, 1.0)]}
    end ++
    for {m, n} <- [{100, 100}, {256, 256}, {1000, 1000}] do
      {"sum #{m}x#{n}", &Nx.sum/1, [r.({m, n}, 1.0)]}
    end ++
    for {m, n} <- [{128, 128}, {256, 1024}, {1024, 1024}] do
      {"softmax #{m}x#{n}", &Models.softmax/1, [r.({m, n}, 1.0)]}
    end
)
