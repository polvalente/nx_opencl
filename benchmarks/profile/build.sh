#!/bin/sh
# Builds libnxcl_prof.dylib next to this script. Usage: build.sh <real libOpenCL path>
# Apple: /System/Library/Frameworks/OpenCL.framework/OpenCL
# PoCL:  /opt/homebrew/opt/opencl-icd-loader/lib/libOpenCL.dylib
set -e
cd "$(dirname "$0")"
real="${1:-/System/Library/Frameworks/OpenCL.framework/OpenCL}"
if [ "$real" = "/System/Library/Frameworks/OpenCL.framework/OpenCL" ]; then
  reexport="-Wl,-reexport_framework,OpenCL"
else
  reexport="-Wl,-reexport_library,$real"
fi
cc -O2 -dynamiclib -o libnxcl_prof.dylib shim.c $reexport -Wno-deprecated-declarations
