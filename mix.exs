defmodule NxOpencl.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_opencl,
      version: "0.1.0",
      elixir: "~> 1.19",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxOpenCL",
      description: "Nx backend for OpenCL GPUs",
      docs: docs(),
      package: package(),
      aliases: aliases(),
      dialyzer: dialyzer()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {NxCL.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  def docs do
    [
      main: "readme",
      extras: ["README.md"]
    ]
  end

  def package do
    [
      name: :nx_opencl,
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/TODO/nx_opencl"}
    ]
  end

  def aliases do
    [
      check: [
        "hex.audit",
        "compile --warnings-as-errors --force",
        "format --check-formatted",
        "credo",
        "deps.unlock --check-unused",
        "spellweaver.check",
        "dialyzer"
      ]
    ]
  end

  def dialyzer do
    [
      plt_add_apps: [:mix, :ex_unit],
      ignore_warnings: ".dialyzer_ignore.exs"
    ]
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:axon, "~> 0.9", only: [:test]},
      {:rustler, "~> 0.36"},
      {:nstandard, "~> 0.1"},
      {:igniter, "~> 0.6", only: [:dev, :test]},
      {:ex_doc, "~> 0.31", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.0", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:spellweaver, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end
end
