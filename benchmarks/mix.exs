defmodule NxOpenclBenchmarks.MixProject do
  use Mix.Project

  # Benchmark project. It is not part of the library or its test suite.
  # Run scripts from this directory with `mix run <script>.exs`.
  # EXLA comes in only with BENCH_EXLA=1, so the project works without a network
  # or an XLA binary.

  def project do
    [
      app: :nx_opencl_benchmarks,
      version: "0.0.0",
      elixir: "~> 1.19",
      deps: deps()
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [{:nx_opencl, path: ".."}] ++
      if System.get_env("BENCH_EXLA") == "1", do: [{:exla, "~> 1.0"}], else: []
  end
end
