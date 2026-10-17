#!/bin/sh
# Builds the shim for the device in use, then runs profile.exs from the repo root.
# On PoCL, set OPENCL_DYLIB_PATH and OCL_ICD_VENDORS first (see profile.exs).
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
real="${OPENCL_DYLIB_PATH:-/System/Library/Frameworks/OpenCL.framework/OpenCL}"
"$root/benchmarks/profile/build.sh" "$real"
export NXCL_REAL_OPENCL="$real"
export NXCL_PROF_FILE="${NXCL_PROF_FILE:-/tmp/nxcl_prof.bin}"
: > "$NXCL_PROF_FILE"
export OPENCL_DYLIB_PATH="$root/benchmarks/profile/libnxcl_prof.dylib"
cd "$root"
mise exec -- mix run benchmarks/profile/${1:-profile.exs}
