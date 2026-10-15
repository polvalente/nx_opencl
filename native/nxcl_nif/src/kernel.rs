use crate::device_ctx::DeviceCtx;
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Cache key: source text, entry name, build options. One cache per device context.
pub type KernelKey = (String, String, String);

pub struct CompiledKernel {
    pub kernel: Kernel,
    pub _program: Program,
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl CompiledKernel {
    pub fn compile(
        ctx: &DeviceCtx,
        source: &str,
        entry_point: &str,
        build_opts: &str,
    ) -> Result<Self, String> {
        let program = Program::create_and_build_from_source(&ctx.context, source, build_opts)
            .map_err(|e| format!("Failed to compile kernel: {:?}", e))?;

        let kernel = Kernel::create(&program, entry_point)
            .map_err(|e| format!("Failed to create kernel '{}': {:?}", entry_point, e))?;

        Ok(Self {
            kernel,
            _program: program,
        })
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
    let mut cache = ctx
        .kernels
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    if let Some(kernel) = cache.get(&key) {
        ctx.stats.hits.fetch_add(1, Ordering::Relaxed);
        return Ok(kernel.clone());
    }

    let kernel = Arc::new(CompiledKernel::compile(
        ctx,
        source,
        entry_point,
        build_opts,
    )?);
    ctx.stats.builds.fetch_add(1, Ordering::Relaxed);
    cache.insert(key, kernel.clone());
    Ok(kernel)
}
