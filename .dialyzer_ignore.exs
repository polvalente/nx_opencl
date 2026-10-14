# Nx.Backend callbacks are typed with a generic tensor and a wrong return type
# for inspect/2 (deps/nx/lib/nx/backend.ex). Our clauses match only NxCL.Backend
# data, which dialyzer reads as a narrower argument type.
[
  {"lib/nxcl/backend.ex", :callback_arg_type_mismatch},
  {"lib/nxcl/backend.ex", :callback_type_mismatch}
]
