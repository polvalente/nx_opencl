# Common setup for all benchmarks. See README.md for how to run them, and for
# BENCH_BUDGET_MS and BENCH_BACKENDS.

defmodule BenchHelper do
  @moduledoc false

  def exla?, do: Code.ensure_loaded?(EXLA)

  def device_name do
    info = NxCL.Devices.fetch!().ctx |> NxCL.Native.device_info() |> Map.new()
    "#{info["name"]} (#{info["device_type"]})"
  end

  def report do
    IO.puts("device: #{device_name()}")
    IO.puts("EXLA:   #{if exla?(), do: "loaded", else: "not run: not installed in this project"}")
    IO.puts("")
  end

  # name, compiler for Nx.Defn.jit, backend that holds the inputs.
  def backends do
    base = [
      {"Evaluator/Binary", Nx.Defn.Evaluator, Nx.BinaryBackend},
      {"NxCL.Backend", Nx.Defn.Evaluator, NxCL.Backend},
      {"NxCL.Compiler", NxCL.Compiler, NxCL.Backend}
    ]

    all = if exla?(), do: base ++ [{"EXLA", EXLA, EXLA.Backend}], else: base

    case System.get_env("BENCH_BACKENDS") do
      nil -> all
      list -> Enum.filter(all, fn {name, _, _} -> name in String.split(list, ",") end)
    end
  end

  def rand(shape, scale \\ 1.0) do
    key = Nx.Random.key(:erlang.phash2(shape))
    {t, _} = Nx.Random.normal(key, 0.0, scale, shape: shape, type: :f32)
    Nx.backend_copy(t, Nx.BinaryBackend)
  end

  defp force(result) do
    Nx.Defn.Composite.flatten_list([result]) |> Enum.each(&Nx.to_binary/1)
  end

  @doc """
  Times `fun.(args...)` on every backend. Inputs sit on the backend before timing.
  Each call reads its outputs back to the host. Returns `{first_us, median_us, reps}`
  per backend, or an error string.
  """
  def measure(fun, args, opts \\ []) do
    budget = String.to_integer(System.get_env("BENCH_BUDGET_MS") || "1500") * 1000
    skip = Keyword.get(opts, :skip, [])

    for {name, compiler, backend} <- backends(), into: %{} do
      if name in skip do
        {name, "skipped"}
      else
        try do
          {name, measure_one(fun, args, compiler, backend, budget)}
        rescue
          e -> {name, "error: " <> String.slice(Exception.message(e), 0, 60)}
        end
      end
    end
  end

  defp measure_one(fun, args, compiler, backend, budget) do
    args = Enum.map(args, &Nx.backend_copy(&1, backend))

    # A fresh closure per backend, so columns share no compile cache.
    jitted = Nx.Defn.jit(fn a -> apply(fun, Tuple.to_list(a)) end, compiler: compiler)
    call = fn -> jitted.(List.to_tuple(args)) |> force() end

    {first, _} = :timer.tc(call)
    call.()
    call.()

    start = System.monotonic_time(:microsecond)
    times = collect(call, start, budget, [])
    sorted = Enum.sort(times)
    {first, Enum.at(sorted, div(length(sorted), 2)), length(sorted)}
  end

  defp collect(call, start, budget, acc) do
    {us, _} = :timer.tc(call)
    acc = [us | acc]
    elapsed = System.monotonic_time(:microsecond) - start

    if (elapsed >= budget and length(acc) >= 3) or length(acc) >= 500 do
      acc
    else
      collect(call, start, budget, acc)
    end
  end

  @doc "Runs a list of `{label, fun, args}` cases and prints one table: first / per call in us."
  def run_cases(title, cases, opts \\ []) do
    names = Enum.map(backends(), &elem(&1, 0))
    IO.puts("### #{title}\n")
    IO.puts("Cell: first call us / median per call us. Device: #{device_name()}\n")
    IO.puts("| case | " <> Enum.join(names, " | ") <> " |")
    IO.puts("|---|" <> String.duplicate("---|", length(names)))

    for {label, fun, args} <- cases do
      result = measure(fun, args, opts)

      cells =
        for n <- names do
          case result[n] do
            {first, med, _reps} -> "#{first} / #{med}"
            msg -> msg
          end
        end

      IO.puts("| #{label} | " <> Enum.join(cells, " | ") <> " |")
    end

    unless exla?(), do: IO.puts("\nEXLA column: not run: not installed in this project.")
    IO.puts("")
  end
end

BenchHelper.report()
