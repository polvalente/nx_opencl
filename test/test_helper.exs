# `NXCL_FUSION=off mix test` runs every row with elementwise fusion off.
if System.get_env("NXCL_FUSION") in ["0", "off", "false"],
  do: Application.put_env(:nx_opencl, :fusion, false)

# Tests tagged `requires: :fp64 | :fp16 | :int64_atomics` are excluded when device 0 lacks the extension.
skipped = for {:requires, cap} = filter <- NxCL.Parity.unsupported_filters(), do: {filter, cap}

for {_filter, cap} <- skipped, do: IO.puts(NxCL.Parity.skip_reason(cap))

NxCL.Parity.Recorder.start()

ExUnit.start(exclude: [:skip, :known_gap] ++ Enum.map(skipped, &elem(&1, 0)))

ExUnit.after_suite(fn _ ->
  NxCL.Parity.Recorder.write_if_enabled(
    System.get_env("NXCL_PARITY_REPORT"),
    ".work/parity_results.exs",
    Enum.map(skipped, &elem(&1, 1))
  )
end)
