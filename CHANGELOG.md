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
  `io_call`, Cholesky, triangular solve, FFT and the `Nx.Block` ops.
- `qr`, `eigh`, `svd` and `lu` lower through their default implementation and run
  on the device, with no host call. `solve` runs fully on the device.
  `runtime_call` and `io_call` are the only ops that call back into the host.
- `NxCL.Backend` no longer copies an op to the CPU. An op with no lowering
  raises `ArgumentError`.
- `Nx.to_binary/2` on a vectorized `NxCL.Backend` tensor returns every element.
- Compiler options `:device`, `:command_queue` and `:max_concurrency`;
  `NxCL.Devices` holds one context per device. `:hooks` is accepted only when
  empty; any other key raises `ArgumentError`.
- Device info reports `driver_version` and `device_type`.
- A compiled function takes tensors that are already on the device without a
  read back and a new upload. A run on the default queue returns before the
  device finishes, and a program with several outputs reads them with one wait.
- Removed: operator fusion (`NxCL.Fusion` and the `:fusion` option), the `:census`
  option, `NxCL.CacheStats`, and the NIFs `kernel_cache_stats` and
  `live_buffer_count`. An unknown option raises `ArgumentError`.
- Removed: the parity matrix, its checker and recorder, and the generated op table in
  the README. The tests are plain tests; the README lists what is not supported.
- Changed: `qr`, `eigh`, `svd` and `lu` no longer call back into the host. They
  lower through Nx's default implementation and run on the device, and they are
  slow (about 2 s for a 64 x 64 `svd`). A block whose default cannot lower raises.
- Changed: lowering, programs, the executor, the program cache and the eager
  evaluator moved under `NxCL.Compiler`. `SubProgram` is gone; an instruction holds
  nested instructions.
- Changed: `NxCL.Compiler.CompileCache` holds at most 1024 entries by default
  (`max_items`).
- Changed: a call with the wrong number of arguments raises `NxCL.CompilerError`,
  naming the expected and given counts. Operands on two contexts raise
  `ArgumentError`. Tensors captured by a function upload once and stay with the
  cached program, and program constants upload once.
- Changed: a NIF that can fail returns `{:error, message}` and Elixir raises
  `ArgumentError`. Reading a freed buffer raises `ArgumentError`, not
  `ErlangError`.
- Changed: FFT axis length is limited to 2^26 for a power of two and 2^22 for
  other lengths (Bluestein). A bad `NXCL_GEMM_TILE` value is an error.
- Fixed: complex `log1p` and `atanh` are accurate near `z = -1`. The old kernel
  lost every digit there (`log1p(-0.99988, 0)` gave `-inf` instead of `-9.01`).
- Fixed: the `SIGCHLD` reset at load time used signal number 20, which is
  `SIGTSTP` on Linux, so it did nothing there.
- Requires `{:nx, "~> 1.0"}`. Axon (`~> 0.9`) is a test dependency only.
- Tests: Nx doctests on the backend and through the compiler, each op and dtype
  against `Nx.BinaryBackend`, gradient equivalence, three Axon models, and a
  check that ops run on the device and not on the CPU. A workflow runs the
  suite on PoCL.
