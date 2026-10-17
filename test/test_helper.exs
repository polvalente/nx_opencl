# Tests tagged `requires: :fp64 | :fp16 | :int64_atomics` are excluded when device 0 lacks the extension.
skipped =
  for {:requires, cap} = filter <- NxCL.TestSupport.Device.unsupported_filters(),
      do: {filter, cap}

for {_filter, cap} <- skipped, do: IO.puts(NxCL.TestSupport.Device.skip_reason(cap))

ExUnit.start(exclude: [:skip, :known_gap] ++ Enum.map(skipped, &elem(&1, 0)))
