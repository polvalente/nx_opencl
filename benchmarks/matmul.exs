# Matrix multiply sweep: square sizes, then three non-square shapes.
#
# Usage: mix run benchmarks/matmul.exs

Code.require_file("bench_helper.exs", __DIR__)

r = &BenchHelper.rand/2

BenchHelper.run_cases(
  "Matmul",
  for n <- [32, 64, 128, 256, 512] do
    {"#{n}x#{n} (#{div(2 * n * n * n, 1_000_000)}M FLOPs)", &Nx.dot/2,
     [r.({n, n}, 1.0), r.({n, n}, 1.0)]}
  end ++
    for {m, k, n} <- [{32, 128, 64}, {128, 256, 64}, {256, 512, 128}] do
      {"#{m}x#{k} * #{k}x#{n}", &Nx.dot/2, [r.({m, k}, 1.0), r.({k, n}, 1.0)]}
    end
)
