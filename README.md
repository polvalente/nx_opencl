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

## Operations

`NxCL.Backend` and `NxCL.Compiler` run Nx ops on the GPU through one set of
generated kernels. The table lists every op the two paths know, and the
dtypes each runs on. "All" means every in-scope dtype: `u8 u16 u32 u64 s8
s16 s32 s64 bf16 f32 f64 c64 c128`. `f64` and `c128` need `cl_khr_fp64`.
`s64` and `u64` `indexed_add` need `cl_khr_int64_base_atomics`. `bf16` is
stored as 16 bits and computed in `f32`.

The table is generated from the parity matrix, which a test checks against
every device run (see "Running Tests").

<!-- parity:ops:start -->
| Op | `NxCL.Compiler` | `NxCL.Backend` | Dtypes |
| --- | --- | --- | --- |
| `abs` | device | device | all |
| `acos` | device | device | all |
| `acosh` | device | device | all |
| `add` | device | device | all |
| `all` | device | device | all |
| `any` | device | device | all |
| `argmax` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `argmin` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `argsort` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `as_type` | device | device | all |
| `asin` | device | device | all |
| `asinh` | device | device | all |
| `atan` | device | device | all |
| `atan2` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `atanh` | device | device | all |
| `bitcast` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `bitwise_and` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `bitwise_not` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `bitwise_or` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `bitwise_xor` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `block` | device | device | all |
| `block_all_close` | device | device | all |
| `block_cumulative_max` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `block_cumulative_min` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `block_cumulative_product` | device | device | all |
| `block_cumulative_sum` | device | device | all |
| `block_fft2` | device | device | all |
| `block_ifft2` | device | device | all |
| `block_irfft` | device | device | all |
| `block_lin_alg_cholesky` | device | device | all |
| `block_lin_alg_determinant` | device | device | all |
| `block_lin_alg_eigh` | host call | host call | all |
| `block_lin_alg_lu` | host call | host call | all |
| `block_lin_alg_qr` | host call | host call | all |
| `block_lin_alg_solve` | device | device | all |
| `block_lin_alg_svd` | host call | host call | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `block_logical_not` | device | device | all |
| `block_phase` | device | device | all |
| `block_rfft` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `block_take` | device | device | all |
| `block_take_along_axis` | device | device | all |
| `block_top_k` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `broadcast` | device | device | all |
| `cbrt` | device | device | all |
| `ceil` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `clip` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `concatenate` | device | device | all |
| `cond` | device | out of scope | all |
| `conjugate` | device | device | all |
| `constant` | device | device | all |
| `conv` | device | device | all |
| `cos` | device | device | all |
| `cosh` | device | device | all |
| `count_leading_zeros` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `divide` | device | device | all |
| `dot` | device | device | all |
| `elem` | device | out of scope | all |
| `equal` | device | device | all |
| `erf` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `erf_inv` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `erfc` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `exp` | device | device | all |
| `expm1` | device | device | all |
| `eye` | device | device | all |
| `fft` | device | device | all |
| `floor` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `fun` | device | out of scope | all |
| `gather` | device | device | all |
| `greater` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `greater_equal` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `ifft` | device | device | all |
| `imag` | device | device | all |
| `indexed_add` | device | device | all |
| `indexed_put` | device | device | all |
| `io_call` | device | out of scope | all |
| `iota` | device | device | all |
| `is_infinity` | device | device | all |
| `is_nan` | device | device | all |
| `left_shift` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `less` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `less_equal` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `log` | device | device | all |
| `log1p` | device | device | all |
| `logical_and` | device | device | all |
| `logical_or` | device | device | all |
| `logical_xor` | device | device | all |
| `max` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `metadata` | device | out of scope | all |
| `min` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `multiply` | device | device | all |
| `negate` | device | device | all |
| `not_equal` | device | device | all |
| `pad` | device | device | all |
| `parameter` | device | out of scope | all |
| `population_count` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `pow` | device | device | all |
| `product` | device | device | all |
| `put_slice` | device | device | all |
| `quotient` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `real` | device | device | all |
| `reduce` | device | device | all |
| `reduce_max` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `reduce_min` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `remainder` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `reshape` | device | device | all |
| `reverse` | device | device | all |
| `right_shift` | device | device | `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `round` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `rsqrt` | device | device | all |
| `runtime_call` | device | out of scope | all |
| `select` | device | device | all |
| `sigmoid` | device | device | all |
| `sign` | device | device | all |
| `sin` | device | device | all |
| `sinh` | device | device | all |
| `slice` | device | device | all |
| `sort` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `sqrt` | device | device | all |
| `squeeze` | device | device | all |
| `stack` | device | device | all |
| `subtract` | device | device | all |
| `sum` | device | device | all |
| `tan` | device | device | all |
| `tanh` | device | device | all |
| `tensor` | device | out of scope | all |
| `transpose` | device | device | all |
| `triangular_solve` | device | device | all |
| `while` | device | out of scope | all |
| `window_max` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `window_min` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `window_product` | device | device | all |
| `window_reduce` | device | device | all |
| `window_scatter_max` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `window_scatter_min` | device | device | `bf16` `f32` `f64` `s8` `s16` `s32` `s64` `u8` `u16` `u32` `u64` |
| `window_sum` | device | device | all |
<!-- parity:ops:end -->

Four linear algebra blocks have no device kernel: `qr`, `eigh`, `svd` and
`lu`. Both paths run them as an explicit host call: the operands are read
back, `Nx.BinaryBackend` runs the block, and the result goes back up. Ops
built on them, such as `solve`, use the same host call.

Neither path copies an op to the CPU silently. An op with no lowering raises
`ArgumentError` ("does not yet lower op ..."). `Nx.backend_deallocate/1`
frees the device buffer at once.

### Out of scope

<!-- parity:cuts:start -->
| Cut | Reason |
| --- | --- |
| Expr-only forms (`cond`, `while`, `fun` and the like) on `NxCL.Backend` | Expr-only form; the backend has no callback. |
| Multi-device sharding, distributed execution | One device per partition is enough for parity. |
| Ops Nx rejects on a dtype (for example `bitwise_and` on floats, `max` on complex) | Nx raises for that op and dtype, so there is nothing to match. The compiler raises at lowering too. Decided by the user 2026-10-14. |
| Sub-byte and 8-bit float dtypes (`s2`, `s4`, `u2`, `u4`, `f8`, `f8_e4m3fn`) | No OpenCL C type; packed storage and widening not needed yet. Decided by the user 2026-10-14. |
| `EMLX.Fast.*` fused attention, rope, norm kernels | Model-specific. |
| `f16` on devices without `cl_khr_fp16` | No device in the test set has the extension, so the cells cannot run. The f16 kernels stay in the code and are tested by Rust source tests. Decided by the user 2026-10-16. |
| `from_pointer` / `to_pointer` | No portable OpenCL handle to export. |
| `quantized_matmul`, quantization | Needs its own format and kernels; no demand yet. |
<!-- parity:cuts:end -->

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

### Parity check

Every `mix test` run records the cells it passed. With
`NXCL_PARITY_REPORT=1` it writes them, with the device's name, vendor,
driver, OpenCL version, type and extensions, to `.work/parity_results.exs`
(or `tmp/parity_results.exs` when there is no `.work` directory, or the file
in `NXCL_PARITY_RESULTS`). Each device adds its own results to the file.
`test/support/check_parity.exs` then lists every in-scope cell with no
passing result:

```bash
# On a real GPU
NXCL_PARITY_REPORT=1 mix test

# On PoCL through the ICD loader (covers f64, c128 and 64-bit atomics)
OPENCL_DYLIB_PATH=/opt/homebrew/opt/opencl-icd-loader/lib/libOpenCL.dylib \
OCL_ICD_VENDORS=/opt/homebrew/etc/OpenCL/vendors \
NXCL_PARITY_REPORT=1 mix test

# The check; exits 1 when a cell is open
elixir test/support/check_parity.exs
```

The check also fails when no result comes from a GPU device, when a GPU
missed a cell its extensions allow, or when a host call ran for anything but
`qr`, `eigh`, `svd`, `lu` or a callback you passed to `Nx.runtime_call/4` or
`Nx.io_call/2`.

CI runs the suite on PoCL (`.github/workflows/ci.yml`). Real-GPU runs stay
manual; use the commands above.
