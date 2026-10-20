# Changelog

## v0.1.0

`NxCL.Compiler` is an `Nx.Defn.Compiler` for OpenCL. `NxCL.Backend` runs on the
same programs and kernels.

- `NxCL.Compiler` lowers a `defn` expression to a typed program, runs it with a
  native executor, and caches programs and kernels. Rust generates the OpenCL C.
- Every Nx dtype runs except `s2`, `s4`, `u2`, `u4`, `f8`, `f8_e4m3fn` and `f16`
  (no test device has `cl_khr_fp16`). `f64` and `c128` need `cl_khr_fp64`.
- Ops: elementwise, shape, reductions, scans, windows, sort, `dot`, `conv`,
  gather, take, indexed add and put, `cond`, `while`, `runtime_call`, `io_call`,
  Cholesky, triangular solve, FFT (also on `c64` and `c128`) and the `Nx.Block` ops.
- `qr`, `eigh`, `svd` and `lu` lower through their default implementation and run
  on the device. They are slow (about 2 s for a 64 x 64 `svd`). A block whose
  default cannot lower raises. Only `runtime_call` and `io_call` call back into
  the host.
- `NxCL.Backend` runs no op on the CPU. An op with no lowering raises
  `ArgumentError`.
- `Nx.to_binary/2` on a vectorized `NxCL.Backend` tensor returns every element.
- Compiler options: `:device`, `:command_queue`, `:max_concurrency`. `:hooks` must
  be empty. Any other key raises `ArgumentError`. `NxCL.Devices` holds one context
  per device.
- Device info reports `driver_version` and `device_type`.
- A compiled function reuses tensors already on the device, with no read back and
  no new upload. A run on the default queue returns before the device finishes,
  and a program with several outputs reads them with one wait.
- Tensors a function captures and program constants upload once and stay with the
  cached program.
- `NxCL.Compiler.CompileCache` holds at most 1024 entries by default (`max_items`).
- A call with the wrong number of arguments raises `NxCL.CompilerError` with the
  expected and given counts. Operands on two contexts raise `ArgumentError`.
- A NIF that can fail returns `{:error, message}`, and Elixir raises
  `ArgumentError`. Reading a freed buffer raises `ArgumentError`, not `ErlangError`.
- FFT axis length is at most 2^26 for a power of two and 2^22 for other lengths
  (Bluestein). A bad `NXCL_GEMM_TILE` value is an error.
- Complex `log1p` and `atanh` are accurate near `z = -1`. Before, `log1p(-0.99988, 0)`
  gave `-inf` instead of `-9.01`.
- The `SIGCHLD` reset at load time used signal number 20, which is `SIGTSTP` on
  Linux, so it did nothing there. It now uses the right number.
- Removed: operator fusion (`NxCL.Fusion`, the `:fusion` option), the `:census`
  option, `NxCL.CacheStats`, the NIFs `kernel_cache_stats` and `live_buffer_count`,
  `SubProgram`, the parity matrix with its checker, recorder and generated README table.
- Lowering, programs, the executor, the program cache and the eager evaluator
  moved under `NxCL.Compiler`. An instruction holds nested instructions.
- Requires `{:nx, "~> 1.0"}`. Axon (`~> 0.9`) is a test dependency only.
- Tests: Nx doctests on the backend and through the compiler, each op and dtype
  against `Nx.BinaryBackend`, gradient equivalence, three Axon models, and a check
  that ops run on the device. A workflow runs the suite on PoCL.
