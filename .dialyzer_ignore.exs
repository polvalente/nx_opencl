# Every entry is a warning in lib/nxcl/backend.ex from the typespecs of the
# Nx.Backend behaviour (deps/nx/lib/nx/backend.ex), not from our logic. Each
# entry names one callback, so a new warning in the file still shows.
#
# Our clauses match only `%NxCL.Backend{}` data, while the behaviour types the
# data as any backend struct, so dialyzer reports a narrower argument type.
[
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to to_binary/2."},
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to inspect/2."},
  # Nx types the return as a tensor; it expects iodata.
  {"lib/nxcl/backend.ex", "Type mismatch for @callback inspect."},
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_deallocate/1."},
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_copy/3."},
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_transfer/3."},
  # Narrower argument type: only `%NxCL.Backend{}` data.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to to_batched/3."},
  # Nx types the return as a list; it reads any enumerable, and a stream keeps batches lazy.
  {"lib/nxcl/backend.ex", "Type mismatch for @callback to_batched."},
  # Nx types it (out, tensor, tensor, list) but calls it with (out, tensor, starts, slice):
  # a list of starts and a tensor. Two warnings, one per argument; this entry matches both.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to put_slice/4."},
  # Nx types the second argument as a tensor but passes a list of tensors.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to concatenate/3."},
  # Nx types the second argument as a tensor but passes a list of tensors.
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to stack/3."}
]
