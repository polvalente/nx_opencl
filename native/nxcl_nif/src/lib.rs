mod buffer;
mod device_ctx;
mod kernel;
mod ops;
mod program;

use buffer::DeviceBuffer;
use device_ctx::DeviceCtx;
use rustler::{NifResult, OwnedBinary, ResourceArc, Term};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::Arc;

struct DeviceCtxInner {
    ctx: DeviceCtx,
}

pub struct NifDeviceCtx {
    inner: Arc<DeviceCtxInner>,
}

pub struct NifDeviceBuffer {
    buffer: Arc<DeviceBuffer>,
    ctx: Arc<DeviceCtxInner>,
}

unsafe impl Send for NifDeviceCtx {}
unsafe impl Sync for NifDeviceCtx {}
unsafe impl Send for NifDeviceBuffer {}
unsafe impl Sync for NifDeviceBuffer {}

#[rustler::resource_impl]
impl rustler::Resource for NifDeviceCtx {}

pub struct NifProgram {
    program: program::Program,
    ctx: Arc<DeviceCtxInner>,
}

unsafe impl Send for NifProgram {}
unsafe impl Sync for NifProgram {}

#[rustler::resource_impl]
impl rustler::Resource for NifDeviceBuffer {}

#[rustler::resource_impl]
impl rustler::Resource for NifProgram {}

// ── Device Context ──────────────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn device_ctx_create(device_id: usize) -> NifResult<ResourceArc<NifDeviceCtx>> {
    match DeviceCtx::new(device_id) {
        Ok(ctx) => {
            let inner = Arc::new(DeviceCtxInner { ctx });
            Ok(ResourceArc::new(NifDeviceCtx { inner }))
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn device_info(ctx: ResourceArc<NifDeviceCtx>) -> NifResult<Vec<(String, String)>> {
    match ctx.inner.ctx.device_info() {
        Ok(info) => Ok(vec![
            ("name".to_string(), info.name),
            ("vendor".to_string(), info.vendor),
            ("version".to_string(), info.version),
            (
                "max_workgroup_size".to_string(),
                info.max_workgroup_size.to_string(),
            ),
            (
                "max_compute_units".to_string(),
                info.max_compute_units.to_string(),
            ),
            (
                "local_mem_size".to_string(),
                info.local_mem_size.to_string(),
            ),
            (
                "global_mem_size".to_string(),
                info.global_mem_size.to_string(),
            ),
            (
                "unified_memory".to_string(),
                info.unified_memory.to_string(),
            ),
            ("extensions".to_string(), info.extensions),
        ]),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Buffer Operations ───────────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_create(
    ctx: ResourceArc<NifDeviceCtx>,
    size: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match DeviceBuffer::create(&ctx.inner.ctx, size) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_write(
    ctx: ResourceArc<NifDeviceCtx>,
    data: rustler::Binary,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match DeviceBuffer::create_with_data(&ctx.inner.ctx, data.as_slice()) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_read(buf: ResourceArc<NifDeviceBuffer>, size: usize) -> NifResult<OwnedBinary> {
    match buf.buffer.read(&buf.ctx.ctx, size) {
        Ok(data) => {
            let mut binary = OwnedBinary::new(data.len()).ok_or(rustler::Error::Term(Box::new(
                "Failed to allocate binary".to_string(),
            )))?;
            binary.as_mut_slice().copy_from_slice(&data);
            Ok(binary)
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Elementwise Operations ──────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn elementwise_binary_op(
    ctx: ResourceArc<NifDeviceCtx>,
    op: String,
    a: ResourceArc<NifDeviceBuffer>,
    b: ResourceArc<NifDeviceBuffer>,
    n: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match ops::elementwise_binary(&ctx.inner.ctx, &op, &a.buffer, &b.buffer, n) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn elementwise_unary_op(
    ctx: ResourceArc<NifDeviceCtx>,
    op: String,
    a: ResourceArc<NifDeviceBuffer>,
    n: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match ops::elementwise_unary(&ctx.inner.ctx, &op, &a.buffer, n) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Matrix Multiply ─────────────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn matmul_op(
    ctx: ResourceArc<NifDeviceCtx>,
    a: ResourceArc<NifDeviceBuffer>,
    b: ResourceArc<NifDeviceBuffer>,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match ops::matmul(&ctx.inner.ctx, &a.buffer, &b.buffer, m, n, k) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Reduce Operations ───────────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn reduce_op(
    ctx: ResourceArc<NifDeviceCtx>,
    op: String,
    input: ResourceArc<NifDeviceBuffer>,
    n: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match ops::reduce(&ctx.inner.ctx, &op, &input.buffer, n) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Softmax ─────────────────────────────────────────────

#[rustler::nif(schedule = "DirtyIo")]
fn softmax_op(
    ctx: ResourceArc<NifDeviceCtx>,
    input: ResourceArc<NifDeviceBuffer>,
    rows: usize,
    cols: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match ops::softmax(&ctx.inner.ctx, &input.buffer, rows, cols) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer {
            buffer: Arc::new(buf),
            ctx: ctx.inner.clone(),
        })),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

// ── Program Execution ───────────────────────────────────

/// Run `f`, turning a panic into an error term so the VM stays up.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("internal error: NIF panicked".to_string()))
}

#[rustler::nif(schedule = "DirtyIo")]
fn compile_program(
    ctx: ResourceArc<NifDeviceCtx>,
    wire: Term,
) -> Result<ResourceArc<NifProgram>, String> {
    guarded(|| {
        let program = program::Program::decode(&ctx.inner.ctx, wire)?;
        Ok(ResourceArc::new(NifProgram {
            program,
            ctx: ctx.inner.clone(),
        }))
    })
}

#[rustler::nif(schedule = "DirtyIo")]
fn eval_program(
    prog: ResourceArc<NifProgram>,
    buffers: Vec<Option<ResourceArc<NifDeviceBuffer>>>,
) -> Result<Vec<ResourceArc<NifDeviceBuffer>>, String> {
    guarded(|| {
        let externals = buffers
            .iter()
            .map(|b| b.as_ref().map(|b| b.buffer.clone()))
            .collect();
        let outputs = prog.program.eval(&prog.ctx.ctx, externals)?;
        Ok(outputs
            .into_iter()
            .map(|buffer| {
                ResourceArc::new(NifDeviceBuffer {
                    buffer,
                    ctx: prog.ctx.clone(),
                })
            })
            .collect())
    })
}

// ── Counters ────────────────────────────────────────────

#[rustler::nif]
fn kernel_cache_stats(ctx: ResourceArc<NifDeviceCtx>) -> (u64, u64, u64) {
    let stats = &ctx.inner.ctx.stats;
    (
        stats.builds.load(Ordering::Relaxed),
        stats.hits.load(Ordering::Relaxed),
        stats.launches.load(Ordering::Relaxed),
    )
}

#[rustler::nif]
fn live_buffer_count(ctx: ResourceArc<NifDeviceCtx>) -> usize {
    ctx.inner.ctx.live_buffers.load(Ordering::Relaxed)
}

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

const SIGCHLD: i32 = 20; // same value on macOS and Linux (x86, arm)
const SIG_DFL: usize = 0;
const SIG_IGN: usize = 1;

fn on_load(_env: rustler::Env, _info: rustler::Term) -> bool {
    // The BEAM sets SIGCHLD to SIG_IGN, so the kernel reaps children at once and
    // wait4 never sees their exit. PoCL runs clang/ld as a child process and waits
    // on it forever. Restore the default only when the VM left it ignored. The BEAM
    // starts its own ports through erl_child_setup and does not need the ignore.
    unsafe {
        let old = signal(SIGCHLD, SIG_DFL);
        if old != SIG_IGN {
            signal(SIGCHLD, old);
        }
    }
    true
}

rustler::init!("Elixir.NxCL.Native", load = on_load);
