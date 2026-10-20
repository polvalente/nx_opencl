# NxCL

An Nx backend and `defn` compiler for OpenCL 1.2+ devices, built on a Rust NIF
that uses the `opencl3` crate.

## Prerequisites

You need an OpenCL runtime, a Rust toolchain to build the NIF, Elixir 1.19 or
newer, and `{:nx, "~> 1.0"}`.

```bash
# Ubuntu/Debian: PoCL, a CPU OpenCL runtime for development and tests
sudo apt install pocl-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev
clinfo --list
```

NVIDIA, AMD and Intel GPU drivers usually include OpenCL. The library loads
`libOpenCL` when it first opens a device; set `OPENCL_DYLIB_PATH` to load another
library.

Some dtypes and ops need a device extension (`clinfo` lists them). A program that
needs a missing one raises `NxCL.CompilerError`.

| Extension | Needed for |
|---|---|
| `cl_khr_fp64` | `f64` and `c128` |
| `cl_khr_int64_base_atomics` | `indexed_add` on `s64`, `u64`, `f64` and `c128` |
| `cl_khr_fp16` | `f16` (not tested on any device) |

## Installation

```elixir
def deps do
  [
    {:nx_opencl, "~> 0.1.0"}
  ]
end
```

## Usage

### Eager backend

Each op runs on the device as a one-op program, with the kernels the compiler
uses. An op with no lowering raises `ArgumentError`; nothing runs on the CPU.

```elixir
Nx.default_backend(NxCL.Backend)

a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
Nx.dot(a, b)

# Move tensors between backends
gpu_tensor = Nx.backend_transfer(a, NxCL.Backend)
cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)
```

### Defn compiler

`NxCL.Compiler` runs a whole `defn` graph as one program and caches programs and
kernels.

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

An Axon model is plain Nx, so `Nx.Defn.jit(predict_fn, compiler: NxCL.Compiler)`
compiles its predict function, and a training step works the same way. Axon is a
test dependency only.

### Compiler options

Pass these to `Nx.Defn.jit/2`. An unknown key raises `ArgumentError`.

- `device:` an index over all devices (default `0`), or `{platform, index}` within
  one platform. `clinfo --list` lists the devices.
- `command_queue:` a queue from `NxCL.Devices.new_queue/1`. Default: the device's
  own queue.
- `max_concurrency:` how many partitions Nx may run at once, each on its own queue.
- `hooks:` must be empty.

## What works

Both paths run Nx ops through one set of generated kernels.

- **Dtypes:** `u8 u16 u32 u64 s8 s16 s32 s64 bf16 f32 f64 c64 c128`. `bf16` is
  stored in 16 bits and computed in `f32`.
- **Elementwise:** unary and binary math, comparison, logic and bitwise ops,
  `select`, `clip` and casts, on every dtype Nx accepts for the op.
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
  through Nx's default implementation and run on the device as `while` loops. They
  are slow (about 2 s for a 64 x 64 `svd`). Their factors may differ from
  `Nx.BinaryBackend`'s in sign and order, so the tests check the defining
  identities (for `qr`, `Q R == A` and `Q` orthonormal), not the values.
- **Host callbacks:** `Nx.runtime_call/4` and `Nx.io_call/2` read back the tensors
  the callback takes, run it on the host and upload the result.

Neither path copies an op to the CPU. An op with no lowering raises
`ArgumentError` ("does not yet lower op ..."). A test covers about 120 representative
ops, through both paths, and Nx's own doctests, under a backend that raises on any
compute call. It does not cover every op.

## What is not supported

- **Small dtypes:** `s2`, `s4`, `u2`, `u4`, `f8` and `f8_e4m3fn`. OpenCL C has no
  type for them; a program that uses one raises `NxCL.CompilerError`.
- **`f16`:** needs `cl_khr_fp16`. No tested device has it, so the `f16` kernels run
  only in Rust source tests.
- **`f64` and `c128`:** need `cl_khr_fp64`. Without it the function raises
  `NxCL.CompilerError` when it runs.
- **64-bit `indexed_add`** (`s64`, `u64`, `f64`, `c128`): needs
  `cl_khr_int64_base_atomics`. Without it the op raises a named error.
- **Ops Nx rejects on a dtype**, such as `bitwise_and` on floats, or `max`, `argmax`
  and `sort` on complex. Nx raises, and so does the compiler.
- **`quantized_matmul`** and quantization.
- **Fused attention, rope and norm kernels** such as `EMLX.Fast.*`.
- **Multi-device sharding.** One device runs a partition.
- **`from_pointer` and `to_pointer`:** OpenCL has no portable handle to export.
- **The `:hooks` option** of `Nx.Defn.jit/2`. An `io_call` runs its own callback.
- **`cond`, `while` and other expression-only forms on `NxCL.Backend`.** Use
  `NxCL.Compiler`.

## Nerves

`libOpenCL.so` loads at run time, so the build needs no OpenCL library. The Rust
target and linker come from the Nerves variables `TARGET_ARCH`, `TARGET_OS`,
`TARGET_ABI` and `CC`. `RUSTLER_TARGET` overrides the target.

Before you build:

- Add the Rust target of your board, for example
  `rustup target add aarch64-unknown-linux-gnu`.
- Nerves must put a cross C compiler in `CC`. The build passes it to Cargo as the
  linker.
- Outside Nerves, set `RUSTLER_TARGET` and install the linker that
  `native/nxcl_nif/.cargo/config.toml` names for it (`aarch64-linux-gnu-gcc` for
  `aarch64-unknown-linux-gnu`).

```bash
export MIX_TARGET=my_custom_target
mix deps.get
mix firmware
```

The target needs an OpenCL runtime. Standard Nerves systems (rpi4, bbb) have no
GPU compute support, so you need a custom system for a SoC with an OpenCL GPU.

### GPU on the target

Enable the DRM driver in the kernel defconfig:

```
# Core
CONFIG_DRM=y
CONFIG_IOMMU_SUPPORT=y
CONFIG_PM=y

CONFIG_DRM_V3D=y        # Raspberry Pi 4/5 (VideoCore VI/VII)
CONFIG_DRM_MSM=y        # Qualcomm Adreno
CONFIG_DRM_PANFROST=y   # ARM Mali Bifrost/Valhall (for example RK3588 Mali-G610)
CONFIG_DRM_PANTHOR=y    # ARM Mali Valhall CSF (kernel 6.9+)
```

Enable Mesa with Rusticl (OpenCL on Mesa's Gallium drivers) in the Buildroot
config:

```
BR2_PACKAGE_MESA3D=y
BR2_PACKAGE_MESA3D_OPENCL=y
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_V3D=y        # RPi4/5
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_FREEDRENO=y  # Adreno
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_PANFROST=y   # Mali
BR2_PACKAGE_OPENCL_ICD_LOADER=y
```

Rusticl needs LLVM and a Rust toolchain at build time; Buildroot pulls them in.
On the device, set `RUSTICL_ENABLE=freedreno` (or `panfrost`) to turn the driver
on. You can also ship a vendor OpenCL library in the rootfs.

| GPU | Kernel driver | Mesa driver | OpenCL status |
|---|---|---|---|
| RPi4 VideoCore VI | `v3d` | v3d | Rusticl, merged Mesa 24.2 |
| RPi5 VideoCore VII | `v3d` | v3d | Rusticl, merged Mesa 24.2 |
| Adreno 6xx/7xx | `drm_msm` | freedreno | Rusticl, merged Mesa 24.3 |
| Mali Bifrost/Valhall | `drm_panfrost` | panfrost | Rusticl, merged Mesa 25.1 |
| Mali Valhall CSF | `drm_panthor` | panfrost | Rusticl, merged Mesa 25.1 |

### CPU OpenCL with PoCL

[PoCL](https://portablecl.org/) runs OpenCL on the CPU through LLVM, on any ARM
device. It is slower than a GPU but works for development and tests. On plain
Linux, `sudo apt install pocl-opencl-icd`, then `clinfo --list` should show
"Portable Computing Language".

Buildroot has no PoCL package. Add one to a `br2-external` tree. PoCL builds with
CMake and needs LLVM and Clang, which add build time and image size.

```
config BR2_PACKAGE_POCL
    bool "pocl"
    depends on BR2_PACKAGE_LLVM
    depends on BR2_INSTALL_LIBSTDCPP
    help
      Portable Computing Language - OpenCL implementation for CPUs.
      https://portablecl.org/
```

## Running Tests

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

For PoCL on macOS, install it with Homebrew (`brew install pocl opencl-icd-loader`)
and run through the loader:

```bash
OPENCL_DYLIB_PATH="$(brew --prefix)/opt/opencl-icd-loader/lib/libOpenCL.dylib" \
OCL_ICD_VENDORS="$(brew --prefix)/etc/OpenCL/vendors" \
mix test
```

On Linux with `pocl-opencl-icd`, `mix test` finds PoCL with no overrides. PoCL has
`cl_khr_fp64` and 64-bit atomics, so it runs every test except those that need
`cl_khr_fp16`. The full suite takes about 30 minutes there.

`.github/workflows/ci.yml` runs the suite on PoCL. It has not run on GitHub yet.

## License

Apache-2.0. See `LICENSE.md`.
