use crate::device_ctx::{cl_error, lock, DeviceCtx};
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use std::sync::{Arc, Mutex};

/// Cache key: source text, entry name, build options. One cache per device context.
pub type KernelKey = (String, String, String);

pub struct CompiledKernel {
    pub kernel: Kernel,
    /// Held from `set_arg` to enqueue: two queues share this kernel object, and
    /// its arguments are one set per object.
    pub launch: Mutex<()>,
    /// Keeps the built program alive as long as the kernel object.
    _program: Program,
}

// SAFETY: `Kernel` holds a raw OpenCL handle, so the compiler will not mark it
// `Sync`. OpenCL (1.1 and later) makes every API call thread safe except
// `clSetKernelArg` on one kernel object, which races with another thread's
// `clSetKernelArg` or enqueue of that object. Every such use takes `launch` from
// the first `set_arg` to the enqueue (see `Program::eval`), so only one thread
// touches a kernel's arguments at a time.
unsafe impl Sync for CompiledKernel {}

impl CompiledKernel {
    pub fn compile(
        ctx: &DeviceCtx,
        source: &str,
        entry_point: &str,
        build_opts: &str,
    ) -> Result<Self, String> {
        let program = Program::create_and_build_from_source(&ctx.context, source, build_opts)
            .map_err(|e| cl_error("Failed to compile kernel", e))?;

        let kernel = Kernel::create(&program, entry_point)
            .map_err(|e| cl_error(&format!("Failed to create kernel '{}'", entry_point), e))?;

        Ok(Self {
            kernel,
            launch: Mutex::new(()),
            _program: program,
        })
    }

    /// Fails unless the kernel declares `n` parameters. `ExecuteKernel` panics
    /// on too many or too few arguments, so the count is checked before any run.
    pub fn expect_args(&self, n: usize) -> Result<(), String> {
        let declared = self
            .kernel
            .num_args()
            .map_err(|e| cl_error("Failed to read the kernel's argument count", e))?;
        if declared as usize == n {
            Ok(())
        } else {
            Err(format!(
                "the kernel declares {} arguments, the program passes {}",
                declared, n
            ))
        }
    }
}

/// Return the cached kernel for the key, building it on a miss.
pub fn cached(
    ctx: &DeviceCtx,
    source: &str,
    entry_point: &str,
    build_opts: &str,
) -> Result<Arc<CompiledKernel>, String> {
    let key = (
        source.to_string(),
        entry_point.to_string(),
        build_opts.to_string(),
    );
    // The lock is held through a build, so two threads that want the same new
    // kernel build it once. A build of a different kernel waits; compiles are
    // rare next to launches.
    let mut cache = lock(&ctx.kernels);

    if let Some(kernel) = cache.get(&key) {
        return Ok(kernel.clone());
    }

    let kernel = Arc::new(CompiledKernel::compile(
        ctx,
        source,
        entry_point,
        build_opts,
    )?);
    cache.insert(key, kernel.clone());
    Ok(kernel)
}
