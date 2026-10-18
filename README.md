# NxCL

Nx backend and `defn` compiler for OpenCL GPUs. Targets OpenCL 1.2+ devices
including mobile GPUs (Adreno, Mali, PowerVR) and desktop GPUs via a Rust
NIF using the `opencl3` crate.

## Prerequisites

An OpenCL runtime must be installed:

```bash
# Ubuntu/Debian - PoCL (CPU OpenCL, good for development/testing)
sudo apt install pocl-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev

# Verify
clinfo --list
```

NVIDIA, AMD, and Intel GPU drivers typically include OpenCL support.
Rust toolchain is required for building the NIF.

## Installation

Add `nx_opencl` to your dependencies:

```elixir
def deps do
  [
    {:nx_opencl, "~> 0.1.0"}
  ]
end
```

## Usage

NxCL provides two ways to run Nx computations on the GPU: the **eager
backend** for ad-hoc tensor operations, and the **defn compiler** for
whole-graph execution of `defn` functions.

### Eager backend

Set `NxCL.Backend` as the default and use Nx as normal. Each op runs on
the device as a one-op program, with the same kernels the compiler uses.
An op with no lowering raises `ArgumentError`; nothing runs on the CPU
backend behind your back.

```elixir
Nx.default_backend(NxCL.Backend)

a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
Nx.dot(a, b)
```

### Defn compiler

Use `NxCL.Compiler` with `Nx.Defn.jit/2` to trace a `defn` function
into a program and run the whole graph on the GPU. The compiler caches
programs and kernels.

```elixir
defmodule MyModel do
  import Nx.Defn

  defn predict(x, w1, b1, w2, b2) do
    x
    |> Nx.dot(w1) |> Nx.add(b1) |> Nx.negate() |> Nx.negate()
    |> Nx.dot(w2) |> Nx.add(b2) |> Nx.sigmoid()
  end
end

jitted = Nx.Defn.jit(&MyModel.predict/5, compiler: NxCL.Compiler)
result = jitted.(input, w1, b1, w2, b2)
```

### Transfer tensors to/from GPU

```elixir
gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)
cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)
```

### With Axon models

Axon is a test dependency only; add it to your own project to use it. An
Axon model is plain Nx, so `Nx.Defn.jit/2` compiles its predict function:

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dense(10, activation: :softmax)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 784}, :f32), Axon.ModelState.empty())

predict = Nx.Defn.jit(predict_fn, compiler: NxCL.Compiler)
output = predict.(params, input)
```

Training steps work the same way: wrap the loss, `Nx.Defn.value_and_grad/2`
and the parameter update in one function and jit it.

### Query device info

```elixir
ctx = NxCL.Native.device_ctx_create(0)
NxCL.Native.device_info(ctx)
# [{"name", "NVIDIA GeForce RTX 3090 Ti"}, {"vendor", "NVIDIA"}, {"device_type", "GPU"}, ...]
```

## What works

`NxCL.Backend` and `NxCL.Compiler` run Nx ops on the device through one set of
generated kernels.

- **Dtypes:** `u8 u16 u32 u64 s8 s16 s32 s64 bf16 f32 f64 c64 c128`. `bf16` is
  stored as 16 bits and computed in `f32`. `f64` and `c128` need `cl_khr_fp64`.
- **Elementwise:** unary and binary math, comparison, logic and bitwise ops, `select`,
  `clip`, casts, on every dtype Nx accepts for the op.
- **Shape:** reshape, transpose, broadcast, slice, `put_slice`, pad, concatenate,
  reverse, as_type and the like.
- **Reductions:** sums, products, min and max, `all`, `any`, `argmax`, `argmin`,
  `reduce`, scans (`cumulative_*`), windows (`window_*`) and `sort`/`argsort`.
- **Contractions:** general `dot` and `conv`, with strides, padding, dilation, groups
  and batch axes.
- **Indexing:** `gather`, `take`, `take_along_axis`, `indexed_add`, `indexed_put`.
- **Control flow:** `cond` and `while`, nested.
- **Linear algebra and FFT:** Cholesky, triangular solve, determinant, `solve`,
  `qr`, `eigh`, `svd`, `lu`, `fft` and `ifft`. `qr`, `eigh`, `svd` and `lu` lower
  through Nx's own default implementation, so they run on the device as `while`
  loops over device kernels. They give the same factors as `Nx.BinaryBackend` up
  to sign and order, and they are slower per matrix than a native kernel (about 2 s
  for a 64 x 64 `svd`).
- **Host callbacks:** `Nx.runtime_call/4` and `Nx.io_call/2` read back the tensors the
  callback takes, run it on the host and upload the result.
- **Compiler options:** `:device`, `:command_queue`, `:max_concurrency`.

Neither path copies an op to the CPU silently. An op with no lowering raises
`ArgumentError` ("does not yet lower op ..."). `Nx.backend_deallocate/1` frees
the device buffer at once.

## What is not supported

- **Small dtypes:** `s2`, `s4`, `u2`, `u4`, `f8` and `f8_e4m3fn`. OpenCL C has no
  type for them.
- **`f16`:** needs `cl_khr_fp16`. No device we tested has it, so the `f16` kernels
  run only in Rust source tests.
- **`f64` and `c128`:** need `cl_khr_fp64`. A device without it raises
  `NxCL.CompilerError` when the function runs.
- **64-bit `indexed_add`** (`s64`, `u64`, `f64`, `c128`): needs
  `cl_khr_int64_base_atomics`. Without it the op raises a named error.
- **Ops Nx rejects on a dtype**, for example `bitwise_and` on floats, or `max`,
  `argmax` and `sort` on complex. Nx raises for these, and so does the compiler.
- **`quantized_matmul`** and quantization.
- **Fused attention, rope and norm kernels** such as `EMLX.Fast.*`.
- **Multi-device sharding** and distributed execution. One device runs a partition.
- **`from_pointer` and `to_pointer`:** OpenCL has no portable handle to export.
- **The `:hooks` option** of `Nx.Defn.jit/2`: `NxCL.Compiler` raises
  `ArgumentError` when it is not empty. An `io_call` runs its own callback.
- **`cond`, `while` and the other expression-only forms on `NxCL.Backend`.** Use
  `NxCL.Compiler` for them.

## Cross-Compilation (Nerves)

NxCL uses dynamic loading (`dlopen`) for `libOpenCL.so` at runtime,
so **no OpenCL library is needed at compile time**. The Rust target
and cross-linker are derived automatically from the Nerves environment
variables (`TARGET_ARCH`, `TARGET_OS`, `TARGET_ABI`, `CC`).

### Build

No extra environment variables are needed. Just build your Nerves
firmware as usual:

```bash
export MIX_TARGET=my_custom_target
mix deps.get
mix firmware
```

NxCL reads the Nerves toolchain variables at compile time to:
- Derive the Rust target triple (e.g. `aarch64-unknown-linux-gnu`)
  from `TARGET_ARCH` + `TARGET_OS` + `TARGET_ABI`
- Set the Cargo linker from `CC`

You can override with `RUSTLER_TARGET` if needed.

### Nerves system requirements

NxCL needs an OpenCL runtime on the target device. Standard Nerves
targets (rpi4, bbb) do **not** include GPU compute support. You need
a custom Nerves system for a SoC with an OpenCL-capable GPU.

#### Linux kernel defconfig

Enable the DRM driver for your GPU:

```
# Core (required)
CONFIG_DRM=y
CONFIG_IOMMU_SUPPORT=y
CONFIG_PM=y

# Raspberry Pi 4/5 (VideoCore VI/VII)
CONFIG_DRM_V3D=y

# Qualcomm Adreno (Snapdragon SoCs)
CONFIG_DRM_MSM=y

# ARM Mali Bifrost/Valhall (e.g. RK3588 Mali-G610)
CONFIG_DRM_PANFROST=y

# ARM Mali Valhall CSF (newer Mali, requires kernel 6.9+)
CONFIG_DRM_PANTHOR=y
```

#### Buildroot / Nerves system config

Enable Mesa with Rusticl (open-source OpenCL via Mesa's Gallium drivers):

```
# Mesa with the appropriate GPU driver
BR2_PACKAGE_MESA3D=y
BR2_PACKAGE_MESA3D_OPENCL=y

# Pick your GPU driver:
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_V3D=y        # RPi4/5
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_FREEDRENO=y  # Adreno
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_PANFROST=y   # Mali

# ICD loader (dispatches to the Mesa driver)
BR2_PACKAGE_OPENCL_ICD_LOADER=y
```

Rusticl requires LLVM and a Rust toolchain at build time, which
Buildroot pulls in automatically. Set `RUSTICL_ENABLE=freedreno`
(or `panfrost`) on the target device to activate the driver.

Alternatively, include vendor-proprietary OpenCL blobs in your rootfs
if available for your SoC.

#### Supported GPU families

| GPU | Kernel driver | Mesa driver | OpenCL status |
|---|---|---|---|
| RPi4 VideoCore VI | `v3d` | v3d | Rusticl, merged Mesa 24.2 |
| RPi5 VideoCore VII | `v3d` | v3d | Rusticl, merged Mesa 24.2 |
| Adreno 6xx/7xx | `drm_msm` | freedreno | Rusticl, merged Mesa 24.3 |
| Mali Bifrost/Valhall | `drm_panfrost` | panfrost | Rusticl, merged Mesa 25.1 |
| Mali Valhall CSF | `drm_panthor` | panfrost | Rusticl, merged Mesa 25.1 |

#### CPU OpenCL (no GPU required)

[PoCL](https://portablecl.org/) (Portable Computing Language) provides
an OpenCL implementation that runs on the CPU via LLVM. This works on
any ARM device — no GPU needed. It's slower than a real GPU for
parallel workloads, but it gives you a working OpenCL runtime for
development, testing, or devices where the GPU isn't available.

**On the host (development):**

```bash
# Ubuntu/Debian
sudo apt install pocl-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev
```

**On-device (Nerves):** There is no upstream Buildroot package for
PoCL. You'll need a custom package in a `br2-external` tree. PoCL
builds with CMake and depends on LLVM/Clang:

```
# In your br2-external package Config.in
config BR2_PACKAGE_POCL
    bool "pocl"
    depends on BR2_PACKAGE_LLVM
    depends on BR2_INSTALL_LIBSTDCPP
    help
      Portable Computing Language - OpenCL implementation for CPUs.
      https://portablecl.org/
```

PoCL supports ARM (32-bit and aarch64) and is OpenCL 3.0 conformant
for CPU targets. Note that LLVM is a heavy dependency — it will
significantly increase build time and image size.

**On-device (plain Linux):**

```bash
# Debian/Ubuntu ARM
sudo apt install pocl-opencl-icd

# Verify
clinfo --list
# Should show: Portable Computing Language / pthread-...
```

## Running Tests

```bash
# All tests (requires an OpenCL runtime)
mix test

# Skip OpenCL tests
mix test --exclude opencl
```

The suite runs on device 0. Tests that need `cl_khr_fp64`, `cl_khr_fp16` or
`cl_khr_int64_base_atomics` are skipped when the device lacks the extension, and
the run prints which. Run one `mix test` at a time: two runs share the device
and can fail each other.

To pick a device, put it first in the OpenCL loader's list. On macOS with PoCL
from Homebrew, run the suite through the ICD loader:

```bash
OPENCL_DYLIB_PATH=/opt/homebrew/opt/opencl-icd-loader/lib/libOpenCL.dylib \
OCL_ICD_VENDORS=/opt/homebrew/etc/OpenCL/vendors \
mix test
```

On Linux with `pocl-opencl-icd` installed, `mix test` finds PoCL with no
overrides. PoCL has `cl_khr_fp64` and 64-bit atomics, so it runs every test except the
ones that need `cl_khr_fp16`. The full suite takes about 15 minutes on an Apple GPU and about 30 on PoCL.

CI runs the suite on PoCL (`.github/workflows/ci.yml`).
