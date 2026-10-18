# Changelog

## v0.1.0

`NxCL.Compiler` is a full `Nx.Defn.Compiler` for OpenCL, and `NxCL.Backend`
runs on the same programs and kernels.

- `NxCL.Compiler` lowers a `defn` expression to a typed program (`NxCL.Compiler.Program`),
  runs it through a native executor, and caches programs and kernels. Rust
  generates the OpenCL C.
- Every Nx dtype runs except `s2`, `s4`, `u2`, `u4`, `f8`, `f8_e4m3fn` and `f16`
  (no test device has `cl_khr_fp16`). `f64` and `c128` need `cl_khr_fp64`.
  `c64` and `c128` have their own math preamble, and FFT runs on them.
- Ops: elementwise, shape, reductions, scans, windows, sort, general `dot` and
  `conv`, gather, take, indexed add and put, `cond`, `while`, `runtime_call`,
  `io_call`, hooks, Cholesky, triangular solve, FFT and the `Nx.Block` ops.
- `qr`, `eigh`, `svd` and `lu` lower through their default implementation and run
  on the device, with no host call. `solve` runs fully on the device.
  `runtime_call` and `io_call` are the only ops that call back into the host.
- `NxCL.Backend` no longer copies an op to the CPU. An op with no lowering
  raises `ArgumentError`.
- `NxCL.Backend.to_binary/2` returns every element of a vectorized tensor.
- Compiler options `:device`, `:command_queue` and `:max_concurrency`;
  `NxCL.Devices` holds one context per device.
- `NxCL.Native.device_info/1` reports `driver_version` and `device_type`.
- A compiled function takes tensors that are already on the device without a
  read back and a new upload. A run on the default queue returns before the
  device finishes, and a program with several outputs reads them with one wait.
- Requires `{:nx, "~> 1.0"}`. Axon (`~> 0.9`) is a test dependency only.
- Tests: Nx doctests on the backend and through the compiler, each op and dtype
  against `Nx.BinaryBackend`, gradient equivalence, three Axon models, and a
  check that ops run on the device and not on the CPU. A workflow runs the
  suite on PoCL.
