# NxCL

Run Nx code on OpenCL devices: an Nx backend and a `defn` compiler. A Rust NIF
drives the device.

## Install

You need Elixir 1.19 or newer, Rust (see `mise.toml`) to build the NIF, and an
OpenCL runtime. On Ubuntu or Debian, PoCL gives you a CPU device for development:

```bash
sudo apt install pocl-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
clinfo --list
```

```elixir
def deps do
  [
    {:nx_opencl, "~> 0.1.0"}
  ]
end
```

NVIDIA, AMD and Intel GPU drivers usually include OpenCL. The library loads
`libOpenCL` when it first opens a device. Set `OPENCL_DYLIB_PATH` to load another
library.

## Quick start

### Backend

Each op runs on the device as a one-op program. An op with no lowering raises
`ArgumentError`; nothing runs on the CPU.

```elixir
Nx.default_backend(NxCL.Backend)

a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
Nx.dot(a, b)

# Move tensors between backends
device_tensor = Nx.backend_transfer(a, NxCL.Backend)
host_tensor = Nx.backend_transfer(device_tensor, Nx.BinaryBackend)
```

### Compiler

`NxCL.Compiler` compiles a `defn` function into one device program. It reuses the
program on later calls.

```elixir
defmodule MyModel do
  import Nx.Defn

  defn predict(x, w1, b1, w2, b2) do
    x
    |> Nx.dot(w1) |> Nx.add(b1)
    |> Nx.dot(w2) |> Nx.add(b2) |> Nx.sigmoid()
  end
end

jitted = Nx.Defn.jit(&MyModel.predict/5, compiler: NxCL.Compiler)
result = jitted.(input, w1, b1, w2, b2)
```

An Axon model is plain Nx, so its predict function compiles the same way:

```elixir
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 4}, :f32), %{})
predict = Nx.Defn.jit(predict_fn, compiler: NxCL.Compiler)
predict.(params, input)
```

`test/nxcl/e2e/axon_test.exs` has working models. Axon is a test dependency only.

### Compiler options

Pass these to `Nx.Defn.jit/2`. An unknown key raises `ArgumentError`.

- `device:` an index over all devices (default `0`), or `{platform, index}` within
  one platform. `clinfo --list` lists the devices.
- `command_queue:` a queue from `NxCL.Devices.new_queue/1`. Default: the device's
  own queue.
- `max_concurrency:` how many partitions Nx may run at once, each on its own queue.
  It must be at least 1.
- `hooks:` must be empty.

## Prerequisites

Some dtypes and ops need a device extension (`clinfo` lists them). A program that
needs a missing one raises `NxCL.CompilerError`.

| Extension | Needed for |
|---|---|
| `cl_khr_fp64` | `f64` and `c128` |
| `cl_khr_int64_base_atomics` | `indexed_add` on `s64`, `u64`, `f64` and `c128` |
| `cl_khr_fp16` | `f16` (untested on any device) |

## What works

Both paths run Nx ops through one set of generated kernels.

- **Dtypes:** `u8 u16 u32 u64 s8 s16 s32 s64 bf16 f32 f64 c64 c128`. `bf16` is
  stored in 16 bits and computed in `f32`.
- **Elementwise:** unary and binary math, comparison, logic and bitwise ops,
  `select`, `clip` and casts.
- **Shape:** reshape, transpose, broadcast, slice, `put_slice`, pad, concatenate,
  reverse and the like.
- **Reductions:** sums, products, min and max, `all`, `any`, `argmax`, `argmin`,
  `reduce`, scans (`cumulative_*`), windows (`window_*`), `sort` and `argsort`.
- **Contractions:** `dot` and `conv`, with strides, padding, dilation, groups and
  batch axes.
- **Indexing:** `gather`, `take`, `take_along_axis`, `indexed_add`, `indexed_put`.
- **Control flow:** nested `cond` and `while`.
- **Linear algebra and FFT:** Cholesky, triangular solve, determinant, `solve`,
  `qr`, `eigh`, `svd`, `lu`, `fft` and `ifft`. `qr`, `eigh`, `svd` and `lu` lower
  through Nx's default implementation. They run on the device but are slow (about
  2 s for a 64 x 64 `svd`). Signs and order can differ from `Nx.BinaryBackend`.
- **Host callbacks:** `Nx.runtime_call/4` and `Nx.io_call/2` read back the tensors
  the callback takes, run it on the host and upload the result.

Neither path copies an op to the CPU. A test runs every op family under a backend
that raises on host compute.

## What is not supported

- **Small dtypes:** `s2`, `s4`, `u2`, `u4`, `f8` and `f8_e4m3fn`. OpenCL C has no
  type for them; a program that uses one raises `NxCL.CompilerError`.
- **`f16`:** needs `cl_khr_fp16`. No tested device has it, so `f16` is untested
  except for Rust source tests.
- **`f64` and `c128`:** need `cl_khr_fp64`. Without it the function raises
  `NxCL.CompilerError` when it runs.
- **64-bit `indexed_add`** (`s64`, `u64`, `f64`, `c128`): needs
  `cl_khr_int64_base_atomics`. Without it the op raises a named error.
- **Ops Nx rejects on a dtype**, such as `bitwise_and` on floats, or `max`, `argmax`
  and `sort` on complex. Nx raises, and so does the compiler.
- **`quantized_matmul`** and fused attention kernels.
- **Multi-device sharding.** One device runs a partition.
- **`from_pointer` and `to_pointer`:** OpenCL has no portable handle to export.
- **The `:hooks` option** of `Nx.Defn.jit/2`. An `io_call` runs its own callback.
- **`cond`, `while` and other expression-only forms on `NxCL.Backend`.** Use
  `NxCL.Compiler`.

## Nerves

`libOpenCL.so` loads at run time, so the build needs no OpenCL library. Nerves sets
the target and the linker. To choose a target by hand, set `RUSTLER_TARGET`. The
target needs an OpenCL runtime, which standard Nerves systems lack. See
[Nerves and Mesa](docs/nerves.md) for the kernel, Buildroot and PoCL setup.

## Running tests

```bash
mix test                    # needs an OpenCL runtime
mix test --exclude opencl   # skip the OpenCL tests
```

The suite runs on device 0. Tests that need `cl_khr_fp64`, `cl_khr_fp16` or
`cl_khr_int64_base_atomics` are skipped when the device lacks the extension, and
the run says which. Run one `mix test` at a time: two runs share the device and
can fail each other.

On a Mac, the suite runs on the Apple GPU with no setup. That GPU has neither
`cl_khr_fp64` nor `cl_khr_int64_base_atomics`, so those tests are skipped. The
full suite takes about 15 minutes.

On Linux with `pocl-opencl-icd`, `mix test` finds PoCL with no overrides. PoCL has
`cl_khr_fp64` and 64-bit atomics, so it runs every test except those that need
`cl_khr_fp16`. The full suite takes about 30 minutes there.

For PoCL on macOS, install it with Homebrew (`brew install pocl opencl-icd-loader`)
and run through the loader:

```bash
OPENCL_DYLIB_PATH="$(brew --prefix)/opt/opencl-icd-loader/lib/libOpenCL.dylib" \
OCL_ICD_VENDORS="$(brew --prefix)/etc/OpenCL/vendors" \
mix test
```

## License

Apache-2.0. See `LICENSE.md`.
