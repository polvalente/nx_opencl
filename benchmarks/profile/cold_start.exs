# First-call time of each model in a fresh process, and how much of it is clBuildProgram.
# Run with benchmarks/profile/run.sh cold_start.exs (same environment as profile.exs).

Code.require_file("../bench_helper.exs", __DIR__)
Code.require_file("../models.exs", __DIR__)
Code.require_file("profile_stats.exs", __DIR__)

file = System.fetch_env!("NXCL_PROF_FILE")

for {label, fun, args} <- Models.cases() do
  args = Enum.map(args, &Nx.backend_copy(&1, NxCL.Backend))
  jitted = Nx.Defn.jit(fn a -> apply(fun, Tuple.to_list(a)) end, compiler: NxCL.Compiler)
  before = ProfStats.read(file)

  {us, _} =
    :timer.tc(fn ->
      jitted.(List.to_tuple(args))
      |> then(&Nx.Defn.Composite.flatten_list([&1]))
      |> Enum.each(&Nx.to_binary/1)
    end)

  after_ = ProfStats.read(file)
  {b0, n0} = before.build
  {b1, n1} = after_.build

  IO.puts(
    "#{label}: first call #{div(us, 1000)} ms, #{n1 - n0} program builds took #{div(b1 - b0, 1_000_000)} ms"
  )
end
