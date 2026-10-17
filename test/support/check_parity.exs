# Checks the parity matrix against recorded results. Run from the repo root:
#
#     elixir test/support/check_parity.exs [--cpu-only] [results.exs ...]
#
# Reads parity_matrix.exs (next to this file; .work/gen_parity_matrix.exs writes it)
# and the results files (`NXCL_PARITY_REPORT=1 mix test` writes one per device; with no
# argument: NXCL_PARITY_RESULTS, else .work/parity_results.exs, else tmp/parity_results.exs).
# Results of several files add up. `--cpu-only` skips the two GPU checks below, for CI,
# where only PoCL runs; it still needs a passing result for every cell.
#
# It fails (exit 1) when:
#   * a cell that is :in or {:host, reason} has no passing result;
#   * a {:out, reason} cell has a reason that is not a row of the README "Out of scope" table;
#   * a {:host, reason} cell is not one of qr, eigh, svd, lu (gate G5);
#   * a host call ran for anything but qr, eigh, svd, lu or a user callback (the results
#     record the count per kind);
#   * no result comes from a GPU device (device type, not the name; not with --cpu-only);
#   * a GPU device missed a cell its own extensions allow;
#   * a cell that needs an extension passed only on a device without it.

dir = __DIR__

{matrix, _} = Code.eval_file(Path.join(dir, "parity_matrix.exs"))

{flags, args} = Enum.split_with(System.argv(), &String.starts_with?(&1, "--"))
cpu_only? = "--cpu-only" in flags

paths =
  case args do
    [] ->
      cond do
        path = System.get_env("NXCL_PARITY_RESULTS") -> [path]
        File.exists?(".work/parity_results.exs") -> [".work/parity_results.exs"]
        true -> ["tmp/parity_results.exs"]
      end

    given ->
      given
  end

results =
  Enum.flat_map(paths, fn path ->
    case File.read(path) do
      {:ok, text} ->
        text |> Code.eval_string() |> elem(0)

      {:error, _} ->
        IO.puts("no #{path}; counting every cell as open")
        []
    end
  end)

# Extensions a cell needs, by dtype (f64, c128) or op and dtype (64-bit atomics; the
# `all_close` of two integer tensors widens to f64 first).
fp64 = "cl_khr_fp64"
int64 = "cl_khr_int64_base_atomics"

needs = fn %{op: op, dtype: dtype} ->
  cond do
    dtype in [{:f, 64}, {:c, 128}] -> [fp64]
    op == :indexed_add and dtype in [{:s, 64}, {:u, 64}] -> [int64]
    op == :block_all_close and elem(dtype, 0) in [:s, :u] -> [fp64]
    true -> []
  end
end

cells = Enum.filter(matrix, &(&1.kind == :cell))
counted = Enum.filter(cells, &(&1.scope == :in or match?({:host, _}, &1.scope)))

passes = Enum.filter(results, &(&1.status == :pass))
passing = MapSet.new(passes, &{&1.op, &1.dtype, &1.path})
open = Enum.reject(counted, &MapSet.member?(passing, {&1.op, &1.dtype, &1.path}))

devices = for %{status: :device} = d <- results, do: d
device_names = devices |> Enum.map(& &1.device) |> Enum.uniq() |> Enum.sort()

IO.puts(
  "results from: #{if device_names == [], do: "(none)", else: Enum.join(device_names, ", ")}"
)

for d <- devices do
  IO.puts(
    "  #{d.device} | #{d.device_type} | #{d.vendor} | driver #{d.driver_version} | #{d.version}"
  )
end

open
|> Enum.group_by(& &1.op)
|> Enum.sort_by(fn {op, _} -> op end)
|> Enum.each(fn {op, op_cells} ->
  by_path = Enum.frequencies_by(op_cells, & &1.path)

  IO.puts(
    "#{op}: #{length(op_cells)} open (compiler #{Map.get(by_path, :compiler, 0)}, backend #{Map.get(by_path, :backend, 0)})"
  )
end)

# {:out, reason} must match a README row.
cut_reasons = for %{kind: :cut, reason: reason} <- matrix, into: MapSet.new(), do: reason

bad_reasons =
  for %{scope: {:out, reason}} = c <- cells,
      not MapSet.member?(cut_reasons, reason),
      uniq: true do
    {c.op, reason}
  end

for {op, reason} <- bad_reasons,
    do: IO.puts("out of scope without a README row: #{op}: #{reason}")

# {:host, reason} only for the G5 ops.
g5 = [:block_lin_alg_qr, :block_lin_alg_eigh, :block_lin_alg_svd, :block_lin_alg_lu]

bad_host =
  for %{scope: {:host, _}} = c <- cells, c.op not in g5, uniq: true, do: c.op

for op <- bad_host, do: IO.puts("host scope outside qr, eigh, svd, lu: #{op}")

# Host calls at run time, counted by kind: only the G5 blocks, and the user's own callbacks.
host_allowed = g5 ++ [:callback]

host_counts =
  for %{status: :host_calls, ops: ops} <- results, {kind, n} <- ops, reduce: %{} do
    acc -> Map.update(acc, kind, n, &(&1 + n))
  end

bad_host_calls = for {kind, n} <- host_counts, kind not in host_allowed, do: {kind, n}

for {kind, n} <- bad_host_calls do
  IO.puts("host call of a kind that may not run on the host: #{kind} (#{n})")
end

IO.puts(
  "host calls by row: " <>
    (host_counts |> Enum.sort() |> Enum.map_join(", ", fn {kind, n} -> "#{kind} #{n}" end))
)

# Device checks.
has = fn device, ext -> ext in String.split(device.extensions, " ", trim: true) end

by_device =
  Enum.group_by(passes, & &1.device, &{&1.op, &1.dtype, &1.path})
  |> Map.new(fn {name, keys} -> {name, MapSet.new(keys)} end)

gpus = if cpu_only?, do: [], else: Enum.filter(devices, &(&1.device_type == "GPU"))

problems = []

problems =
  if gpus == [] and not cpu_only? do
    IO.puts("no result comes from a GPU device")
    [:no_gpu | problems]
  else
    problems
  end

# A GPU device must pass every cell its extensions allow.
problems =
  Enum.reduce(gpus, problems, fn gpu, problems ->
    mine = Map.get(by_device, gpu.device, MapSet.new())
    allowed = Enum.filter(counted, fn c -> Enum.all?(needs.(c), &has.(gpu, &1)) end)
    missed = Enum.reject(allowed, &MapSet.member?(mine, {&1.op, &1.dtype, &1.path}))

    IO.puts(
      "#{gpu.device}: passes #{length(allowed) - length(missed)} of #{length(allowed)} cells its extensions allow"
    )

    for c <- Enum.take(missed, 20), do: IO.puts("  missed: #{c.op} #{inspect(c.dtype)} #{c.path}")
    if missed == [], do: problems, else: [{:gpu_missed, gpu.device, length(missed)} | problems]
  end)

# A cell that needs an extension must pass on a device that has it.
misplaced =
  for c <- counted,
      req = needs.(c),
      req != [],
      MapSet.member?(passing, {c.op, c.dtype, c.path}),
      not Enum.any?(devices, fn d ->
        Enum.all?(req, &has.(d, &1)) and
          MapSet.member?(Map.get(by_device, d.device, MapSet.new()), {c.op, c.dtype, c.path})
      end) do
    c
  end

for c <- Enum.take(misplaced, 20) do
  IO.puts("passed only on a device without the extension: #{c.op} #{inspect(c.dtype)} #{c.path}")
end

out = Enum.count(cells, &match?({:out, _}, &1.scope))
host = Enum.count(cells, &match?({:host, _}, &1.scope))

IO.puts(
  "cells: #{length(cells)} total, #{out} out of scope, #{host} host, #{length(counted)} counted, " <>
    "#{length(counted) - length(open)} passing, #{length(open)} open"
)

failed? =
  open != [] or bad_reasons != [] or bad_host != [] or bad_host_calls != [] or problems != [] or
    misplaced != []

if failed?, do: System.halt(1), else: IO.puts("parity check: ok")
