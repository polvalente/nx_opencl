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
- `inspect/2` on a tensor reads at most the inspect limit. A freed tensor prints a
  short note.
- Compiler options: `:device`, `:command_queue`, `:max_concurrency`. `:hooks` must
  be empty. Any other key raises `ArgumentError`, and so does a `max_concurrency`
  below 1. Backend options take only `:device`; an unknown key raises. A bad
  `device:` raises `ArgumentError`. `NxCL.Devices` holds one context per device.
- Device info reports `driver_version` and `device_type`.
- A compiled function reuses tensors that are already on the device. A call returns
  before the device finishes; a program with several outputs reads them with one
  wait.
- Tensors a function captures and program constants upload once and stay with the
  cached program.
- `NxCL.Compiler.CompileCache` holds at most 1024 entries by default (`max_items`).
  Each cached closure keeps the tensors it captured on the device, so a full cache
  can hold a lot of device memory.
- A call with the wrong number of arguments raises `NxCL.CompilerError` with the
  expected and given counts. Operands on two contexts raise `ArgumentError`.
- A NIF that can fail returns `{:error, message}`, and Elixir raises
  `ArgumentError`. Reading a freed buffer raises `ArgumentError`, not `ErlangError`.
- FFT axis length is at most 2^26 for a power of two and 2^22 for other lengths
  (Bluestein).
- Requires `{:nx, "~> 1.0"}`. Axon (`~> 0.9`) is a test dependency only.
