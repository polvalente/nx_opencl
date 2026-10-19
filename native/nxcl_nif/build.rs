fn main() {
    // opencl3 0.10+ uses dynamic loading (dlopen) by default.
    // No link-time libOpenCL.so is needed — the NIF loads it at runtime.
    //
    // Cross-compilation linker is set by NxCL.Native via Rustler's :env
    // option, using Nerves' CC environment variable.
    //
    // OPENCL_LIB_PATH adds a link search path for libOpenCL. Nothing here links
    // against it; it matters only if a dependency is built to link at build
    // time (unusual).
    if let Ok(opencl_lib) = std::env::var("OPENCL_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", opencl_lib);
    }

    println!("cargo:rerun-if-env-changed=OPENCL_LIB_PATH");
}
