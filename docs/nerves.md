# Nerves and Mesa

`libOpenCL.so` loads at run time, so the build needs no OpenCL library. Nerves sets
the target and the linker: the Rust target comes from `TARGET_ARCH`, `TARGET_OS` and
`TARGET_ABI`, and the linker from `CC`. To choose a target by hand, set
`RUSTLER_TARGET`.

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

None of the setup below has been tested with this library.

## GPU on the target

Enable the DRM driver in the kernel defconfig:

```
# Core
CONFIG_DRM=y
CONFIG_IOMMU_SUPPORT=y
CONFIG_PM=y

CONFIG_DRM_MSM=y        # Qualcomm Adreno
CONFIG_DRM_PANFROST=y   # ARM Mali Bifrost/Valhall (for example RK3588 Mali-G610)
CONFIG_DRM_PANTHOR=y    # ARM Mali Valhall CSF (kernel 6.9+)
```

Enable Mesa with Rusticl (OpenCL on Mesa's Gallium drivers) in the Buildroot
config:

```
BR2_PACKAGE_MESA3D=y
BR2_PACKAGE_MESA3D_OPENCL=y
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_FREEDRENO=y  # Adreno
BR2_PACKAGE_MESA3D_GALLIUM_DRIVER_PANFROST=y   # Mali
BR2_PACKAGE_OPENCL_ICD_LOADER=y
```

Rusticl needs LLVM and a Rust toolchain at build time; Buildroot pulls them in.
On the device, set `RUSTICL_ENABLE=freedreno` (or `panfrost`) to turn the driver
on. You can also ship a vendor OpenCL library in the rootfs.

The table lists drivers reported to expose OpenCL through Rusticl. We have not
tested any row, and we checked none against the Mesa documentation. Check the
Mesa release notes for your version before you rely on one.

| GPU | Kernel driver | Mesa driver | Status |
|---|---|---|---|
| Adreno 6xx/7xx | `drm_msm` | freedreno | reported, untested |
| Mali Bifrost/Valhall | `drm_panfrost` | panfrost | reported, untested |
| Mali Valhall CSF | `drm_panthor` | panfrost | reported, untested |

## CPU OpenCL with PoCL

[PoCL](https://portablecl.org/) runs OpenCL on the CPU through LLVM, on ARM CPUs that PoCL
supports. It is slower than a GPU but works for development and tests. On plain
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
