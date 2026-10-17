# Splits the time of one model call into transfer, kernel and host time.
#
# The split comes from a shim that stands in for libOpenCL (shim.c). It reads
# CL_QUEUE_PROFILING_ENABLE events for kernel time and times buffer uploads and reads.
# Nothing in lib/ or native/ changes.
#
#   Apple:  benchmarks/profile/run.sh
#   PoCL:   OPENCL_DYLIB_PATH=/opt/homebrew/opt/opencl-icd-loader/lib/libOpenCL.dylib \
#           OCL_ICD_VENDORS=/opt/homebrew/etc/OpenCL/vendors benchmarks/profile/run.sh
#
# Per call: wall = whole call with output read back; transfer = input uploads and output
# reads; kernel = device time of launches; host = wall - transfer - kernel (Elixir tracing,
# lowering, argument setup, enqueue, waiting gaps). Kernel time is also split by entry name.

Code.require_file("../bench_helper.exs", __DIR__)
Code.require_file("../models.exs", __DIR__)
Code.require_file("profile_stats.exs", __DIR__)

defmodule Prof do
  def class(name) do
    cond do
      String.starts_with?(name, "gemm") -> "dot"
      String.starts_with?(name, "conv") or String.starts_with?(name, "im2col") -> "conv"
      String.starts_with?(name, "fz_") -> "fused elementwise"
      String.starts_with?(name, "ew_") -> "elementwise"
      String.starts_with?(name, "mv_") -> "shape"
      String.starts_with?(name, "rd_") -> "reduce"
      true -> "other (" <> name <> ")"
    end
  end

  def diff(a, b, calls) do
    d = fn {x, n}, {y, m} -> {(y - x) / calls, (m - n) / calls} end

    kernels =
      for {n, {c2, ns2}} <- b.kernels, reduce: %{} do
        acc ->
          {c1, ns1} = Map.get(a.kernels, n, {0, 0})

          Map.update(acc, class(n), {(c2 - c1) / calls, (ns2 - ns1) / calls}, fn {c, ns} ->
            {c + (c2 - c1) / calls, ns + (ns2 - ns1) / calls}
          end)
      end

    %{
      up: d.(a.up, b.up),
      read: d.(a.read, b.read),
      alloc: d.(a.alloc, b.alloc),
      enq: d.(a.enq, b.enq),
      fin: d.(a.fin, b.fin),
      kernels: kernels
    }
  end

  def run(label, fun, args, compiler, backend, file, calls) do
    args = Enum.map(args, &Nx.backend_copy(&1, backend))
    jitted = Nx.Defn.jit(fn a -> apply(fun, Tuple.to_list(a)) end, compiler: compiler)

    call = fn ->
      Nx.Defn.Composite.flatten_list([jitted.(List.to_tuple(args))]) |> Enum.each(&Nx.to_binary/1)
    end

    call.()
    call.()
    before = ProfStats.read(file)
    t0 = System.monotonic_time(:microsecond)
    for _ <- 1..calls, do: call.()
    wall = (System.monotonic_time(:microsecond) - t0) / calls
    d = diff(before, ProfStats.read(file), calls)

    transfer = (elem(d.up, 0) + elem(d.read, 0)) / 1000
    kernel = d.kernels |> Map.values() |> Enum.map(&elem(&1, 1)) |> Enum.sum() |> Kernel./(1000)
    host = wall - transfer - kernel
    alloc = elem(d.alloc, 0) / 1000

    IO.puts(
      "#{label} [#{inspect(compiler)}, inputs on #{inspect(backend)}]: wall #{round(wall)} us, transfer #{round(transfer)} us " <>
        "(upload #{round(elem(d.up, 0) / 1000)} in #{elem(d.up, 1)}, read #{round(elem(d.read, 0) / 1000)} in #{elem(d.read, 1)}), " <>
        "kernel #{round(kernel)} us, host #{round(host)} us (buffer allocation #{round(alloc)} us in #{round(elem(d.alloc, 1))}; " <>
        "enqueue #{round(elem(d.enq, 0) / 1000)} us in #{round(elem(d.enq, 1))}; finish wait #{round(elem(d.fin, 0) / 1000)} us in #{round(elem(d.fin, 1))})"
    )

    for {c, {n, ns}} <- Enum.sort_by(d.kernels, fn {_, {_, ns}} -> -ns end) do
      IO.puts(
        "    #{String.pad_trailing(c, 22)} #{String.pad_leading(to_string(Float.round(n, 1)), 6)} launches  #{round(ns / 1000)} us"
      )
    end
  end
end

file = System.fetch_env!("NXCL_PROF_FILE")
calls = String.to_integer(System.get_env("CALLS") || "50")
only = System.get_env("ONLY")

for {label, fun, args} <- Models.cases(), is_nil(only) or String.contains?(label, only) do
  for {compiler, backend} <- [
        {NxCL.Compiler, Nx.BinaryBackend},
        {NxCL.Compiler, NxCL.Backend},
        {Nx.Defn.Evaluator, NxCL.Backend}
      ] do
    Prof.run(label, fun, args, compiler, backend, file, calls)
  end
end
