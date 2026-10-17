# Elementwise benchmarks: single ops at several sizes, and a three-op chain.
#
# Usage: mix run benchmarks/elementwise.exs

Code.require_file("bench_helper.exs", __DIR__)

r = &BenchHelper.rand/2
chain = fn x -> x |> Nx.exp() |> Nx.negate() |> Nx.sigmoid() end

BenchHelper.run_cases(
  "Elementwise",
  for n <- [1_000, 100_000, 1_000_000] do
    {"exp n=#{n}", &Nx.exp/1, [r.({n}, 1.0)]}
  end ++
    for n <- [1_000, 100_000, 1_000_000] do
      {"add n=#{n}", &Nx.add/2, [r.({n}, 1.0), r.({n}, 1.0)]}
    end ++
    for n <- [1_000, 100_000, 1_000_000] do
      {"exp-negate-sigmoid chain n=#{n}", chain, [r.({n}, 1.0)]}
    end
)
