#!/bin/sh
# Builds the shim for the device in use, then runs profile.exs from the repo root.
# macOS only (the shim is a .dylib built with `cc -dynamiclib`).
# On PoCL, set OPENCL_DYLIB_PATH and OCL_ICD_VENDORS first (see profile.exs).
# Environment: OPENCL_DYLIB_PATH (the real libOpenCL; default: Apple's framework),
# NXCL_PROF_FILE (counters file; default /tmp/nxcl_prof.bin), NXCL_MIX (how to run
# Mix; default "mise exec -- mix").
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
real="${OPENCL_DYLIB_PATH:-/System/Library/Frameworks/OpenCL.framework/OpenCL}"
"$root/benchmarks/profile/build.sh" "$real"
export NXCL_REAL_OPENCL="$real"
export NXCL_PROF_FILE="${NXCL_PROF_FILE:-/tmp/nxcl_prof.bin}"
: > "$NXCL_PROF_FILE"
export OPENCL_DYLIB_PATH="$root/benchmarks/profile/libnxcl_prof.dylib"
cd "$root"
${NXCL_MIX:-mise exec -- mix} run benchmarks/profile/${1:-profile.exs}
