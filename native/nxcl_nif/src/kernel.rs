use crate::device_ctx::DeviceCtx;
use opencl3::kernel::Kernel;
use opencl3::program::Program;

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

// Embedded kernel sources
pub const ELEMENTWISE_SOURCE: &str = include_str!("kernel_source/elementwise.cl");
pub const MATMUL_SOURCE: &str = include_str!("kernel_source/matmul.cl");
pub const REDUCE_SOURCE: &str = include_str!("kernel_source/reduce.cl");
pub const SOFTMAX_SOURCE: &str = include_str!("kernel_source/softmax.cl");
