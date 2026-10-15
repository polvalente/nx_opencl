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
into a flat op sequence and execute the whole graph on the GPU in one
pass.

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

Axon models are pure Nx under the hood, so they work with the eager
backend by transferring params and inputs to the GPU:

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dense(10, activation: :softmax)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 784}, :f32), %{})

# Transfer params to GPU (walk the ModelState struct)
gpu_params = %{params |
  data: Map.new(params.data, fn {layer, layer_params} ->
    {layer, Map.new(layer_params, fn {k, v} ->
      {k, Nx.backend_transfer(v, NxCL.Backend)}
    end)}
  end)
}

gpu_input = Nx.backend_transfer(input, NxCL.Backend)
output = predict_fn.(gpu_params, gpu_input)
```

### Query device info

```elixir
ctx = NxCL.Native.device_ctx_create(0)
NxCL.Native.device_info(ctx)
# [{"name", "NVIDIA GeForce RTX 3090 Ti"}, {"vendor", "NVIDIA"}, ...]
```

## Operations

`NxCL.Backend` and `NxCL.Compiler` run every Nx op on the GPU through one
set of generated kernels, for every Nx dtype except `s2`, `s4`, `u2`, `u4`,
`f8` and `f8_e4m3fn`. `f64` and `c128` need `cl_khr_fp64`; `f16` needs
`cl_khr_fp16`.

Four linear algebra blocks have no device kernel: `qr`, `eigh`, `svd` and
`lu`. Both paths run them as an explicit host call: the operands are read
back, `Nx.BinaryBackend` runs the block, and the result goes back up.

Neither path copies an op to the CPU silently. An op with no lowering raises
`ArgumentError` ("does not yet lower op ..."). `Nx.from_pointer/3` and
`Nx.to_pointer/2` raise on the backend: OpenCL has no portable handle to
export. `Nx.backend_deallocate/1` frees the device buffer at once.

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
# All tests (requires OpenCL runtime)
mix test

# Skip OpenCL tests
mix test --exclude opencl
```
