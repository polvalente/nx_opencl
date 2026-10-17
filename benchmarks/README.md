# NxCL Benchmarks

Scripts that time `NxCL.Compiler` and `NxCL.Backend` against the Nx Evaluator on
`Nx.BinaryBackend` and, when installed, EXLA.

## Run

From the repo root, without EXLA:

```bash
mix run benchmarks/elementwise.exs
mix run benchmarks/matmul.exs
mix run benchmarks/reduction.exs
mix run benchmarks/mlp_inference.exs
mix run benchmarks/compiler_vs_eager.exs
```

From `benchmarks/`, with EXLA. This is a separate Mix project that depends on the
library by path. EXLA is not a dependency of the library.

```bash
cd benchmarks
BENCH_EXLA=1 mix deps.get
BENCH_EXLA=1 mix run elementwise.exs
```

EXLA 1.0 downloads an XLA binary on first build. On the Mac it runs on the CPU only.
Do not combine EXLA with PoCL in one process: the two C++ runtimes clash and the VM
aborts. Run PoCL without EXLA.

Set `BENCH_BUDGET_MS` to change the time spent per cell (default 1500) and
`BENCH_BACKENDS` (for example `NxCL.Compiler,EXLA`) to run some columns only.
To use PoCL, set `OPENCL_DYLIB_PATH` and `OCL_ICD_VENDORS` as the test suite does.

## What each script measures

| Script | What it measures |
|---|---|
| `elementwise.exs` | `exp`, `add` and a three-op chain at 1e3, 1e5 and 1e6 elements |
| `matmul.exs` | Square matmul from 32 to 512, and three non-square shapes |
| `reduction.exs` | `sum`, `reduce_max`, 2D sums and softmax |
| `mlp_inference.exs` | Dense layer, two and three layer MLPs, batch sizes |
| `compiler_vs_eager.exs` | MLP forward and training step, CNN forward, transformer block |

Each table cell reads `first call us / median per call us`. The first call includes
tracing, lowering and kernel builds. The per-call time comes after warmup. Inputs sit
on the backend before timing, and every call reads its outputs back to the host.
The kernel cache is shared inside one process, so the first call of a later row can
be cheaper than a cold start.

## Columns

- **Evaluator/Binary**: `Nx.Defn.Evaluator` on `Nx.BinaryBackend`.
- **NxCL.Backend**: `Nx.Defn.Evaluator` with tensors on `NxCL.Backend`, one op at a time.
- **NxCL.Compiler**: `Nx.Defn.jit` with `compiler: NxCL.Compiler`.
- **EXLA**: `compiler: EXLA` with tensors on `EXLA.Backend`, when loaded.

## Profile split

`benchmarks/profile/` splits a model call into transfer, kernel and host time. It
holds a small C shim that stands in for `libOpenCL`, so nothing in `lib/` or `native/`
changes. The shim reads the `CL_QUEUE_PROFILING_ENABLE` events for kernel time and times
buffer uploads and reads.

```bash
benchmarks/profile/run.sh                  # Apple
CALLS=100 ONLY=Transformer benchmarks/profile/run.sh
benchmarks/profile/run.sh cold_start.exs   # first call and program build time
```

Set `NXCL_PROF_FINISH=1` to make every queue flush wait for the device, which shows
the split of a run that ends in `clFinish`.
