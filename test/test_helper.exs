# Tests tagged `requires: :fp64 | :fp16 | :int64_atomics` are excluded when device 0 lacks the extension.
skipped = for {:requires, cap} = filter <- NxCL.Parity.unsupported_filters(), do: {filter, cap}

for {_filter, cap} <- skipped, do: IO.puts(NxCL.Parity.skip_reason(cap))

NxCL.Parity.Recorder.start()
NxCL.Parity.Tripwire.start()

ExUnit.start(exclude: [:skip, :known_gap] ++ Enum.map(skipped, &elem(&1, 0)))

ExUnit.after_suite(fn _ ->
  NxCL.Parity.Recorder.write_if_enabled(
    System.get_env("NXCL_PARITY_REPORT"),
    NxCL.Parity.Recorder.default_path(),
    Enum.map(skipped, &elem(&1, 1))
  )
end)
