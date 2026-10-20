# All 12 warnings are in lib/nxcl/backend.ex and come from the typespecs of the
# Nx.Backend behaviour (deps/nx/lib/nx/backend.ex), not from our logic. The 11
# entries below cover them: put_slice/4 raises two warnings, one per wrong
# argument, and one entry matches both. Each entry names one callback, so a new
# warning in the file still shows.
#
# Our clauses match only `%NxCL.Backend{}` data. The behaviour types the data as any
# backend struct, so dialyzer reports a narrower argument type. Six entries:
# to_binary/2, inspect/2, backend_deallocate/1, backend_copy/3, backend_transfer/3
# and to_batched/3.
#
# Five callbacks have a wrong type in Nx itself:
#   * inspect/2 is typed to return a tensor; Nx expects iodata. Two entries (the
#     argument above, and the return type).
#   * to_batched/3 is typed to return a list; Nx reads any enumerable, and a
#     stream keeps the batches lazy. Two entries (argument and return type).
#   * put_slice/4 is typed (out, tensor, tensor, list); Nx calls it with
#     (out, tensor, starts, slice), a list of starts and a tensor.
#   * concatenate/3 and stack/3 are typed with a tensor as the second argument;
#     Nx passes a list of tensors.
[
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to to_binary/2."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to inspect/2."},
  {"lib/nxcl/backend.ex", "Type mismatch for @callback inspect."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_deallocate/1."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_copy/3."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to backend_transfer/3."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to to_batched/3."},
  {"lib/nxcl/backend.ex", "Type mismatch for @callback to_batched."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to put_slice/4."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to concatenate/3."},
  {"lib/nxcl/backend.ex", "Type mismatch with behaviour callback to stack/3."}
]
