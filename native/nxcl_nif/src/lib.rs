mod buffer;
mod codegen;
mod device_ctx;
mod kernel;
mod program;

use buffer::DeviceBuffer;
use device_ctx::DeviceCtx;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use rustler::{NifResult, OwnedBinary, ResourceArc, Term};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

struct DeviceCtxInner {
    ctx: DeviceCtx,
}

pub struct NifDeviceCtx {
    inner: Arc<DeviceCtxInner>,
}

/// A device buffer handle. `buffer_free` empties the handle; the device memory
/// goes when the last handle (an aliasing result shares it) lets go.
pub struct NifDeviceBuffer {
    buffer: Mutex<Option<Arc<DeviceBuffer>>>,
    ctx: Arc<DeviceCtxInner>,
}

impl NifDeviceBuffer {
    fn new(buffer: Arc<DeviceBuffer>, ctx: Arc<DeviceCtxInner>) -> Self {
        Self {
            buffer: Mutex::new(Some(buffer)),
            ctx,
        }
    }

    fn get(&self) -> Result<Arc<DeviceBuffer>, String> {
        self.buffer
            .lock()
            .map_err(|e| format!("Lock error: {:?}", e))?
            .clone()
            .ok_or_else(|| "buffer was deallocated".to_string())
    }
}

unsafe impl Send for NifDeviceCtx {}
unsafe impl Sync for NifDeviceCtx {}
unsafe impl Send for NifDeviceBuffer {}
unsafe impl Sync for NifDeviceBuffer {}

#[rustler::resource_impl]
impl rustler::Resource for NifDeviceCtx {}

/// An extra command queue on a device context. Each one runs a program at a
/// time, and different queues run at the same time.
pub struct NifQueue {
    queue: Mutex<CommandQueue>,
    ctx: Arc<DeviceCtxInner>,
}

unsafe impl Send for NifQueue {}
unsafe impl Sync for NifQueue {}

#[rustler::resource_impl]
impl rustler::Resource for NifQueue {}

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
    open_ctx(None, device_id)
}

/// Like `device_ctx_create`, with the index counted inside one platform.
#[rustler::nif(schedule = "DirtyIo")]
fn device_ctx_create_on(platform: usize, device_id: usize) -> NifResult<ResourceArc<NifDeviceCtx>> {
    open_ctx(Some(platform), device_id)
}

fn open_ctx(platform: Option<usize>, device_id: usize) -> NifResult<ResourceArc<NifDeviceCtx>> {
    match DeviceCtx::new(platform, device_id) {
        Ok(ctx) => {
            let inner = Arc::new(DeviceCtxInner { ctx });
            Ok(ResourceArc::new(NifDeviceCtx { inner }))
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

/// A new command queue on the context, for `eval_program_on`.
#[rustler::nif(schedule = "DirtyIo")]
fn queue_create(ctx: ResourceArc<NifDeviceCtx>) -> NifResult<ResourceArc<NifQueue>> {
    // OpenCL 1.2 entry point, as in `DeviceCtx::new`.
    #[allow(deprecated)]
    let queue = CommandQueue::create_default(&ctx.inner.ctx.context, CL_QUEUE_PROFILING_ENABLE)
        .map_err(|e| {
            rustler::Error::Term(Box::new(format!("Failed to create command queue: {:?}", e)))
        })?;
    Ok(ResourceArc::new(NifQueue {
        queue: Mutex::new(queue),
        ctx: ctx.inner.clone(),
    }))
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
            (
                "max_kernel_args".to_string(),
                info.max_kernel_args.to_string(),
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
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer::new(
            Arc::new(buf),
            ctx.inner.clone(),
        ))),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_write(
    ctx: ResourceArc<NifDeviceCtx>,
    data: rustler::Binary,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    match DeviceBuffer::create_with_data(&ctx.inner.ctx, data.as_slice()) {
        Ok(buf) => Ok(ResourceArc::new(NifDeviceBuffer::new(
            Arc::new(buf),
            ctx.inner.clone(),
        ))),
        Err(e) => Err(rustler::Error::Term(Box::new(e))),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_read(buf: ResourceArc<NifDeviceBuffer>, size: usize) -> NifResult<OwnedBinary> {
    let device = buf
        .get()
        .map_err(|e| rustler::Error::RaiseTerm(Box::new(e)))?;
    match device.read(&buf.ctx.ctx, size) {
        Ok(data) => {
            let mut binary = OwnedBinary::new(data.len()).ok_or(rustler::Error::RaiseTerm(
                Box::new("Failed to allocate binary".to_string()),
            ))?;
            binary.as_mut_slice().copy_from_slice(&data);
            Ok(binary)
        }
        Err(e) => Err(rustler::Error::RaiseTerm(Box::new(e))),
    }
}

/// Drops the handle's hold on the device memory. Reading the handle afterwards is
/// an error. Returns false when the handle was already empty.
#[rustler::nif]
fn buffer_free(buf: ResourceArc<NifDeviceBuffer>) -> NifResult<bool> {
    let freed = buf
        .buffer
        .lock()
        .map_err(|e| rustler::Error::Term(Box::new(format!("Lock error: {:?}", e))))?
        .take();
    Ok(freed.is_some())
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
            .map(|b| b.as_ref().map(|b| b.get()).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = prog
            .program
            .eval(&prog.ctx.ctx, &prog.ctx.ctx.queue, externals)?;
        Ok(wrap_outputs(&prog.ctx, outputs))
    })
}

/// `eval_program` on an extra queue of the program's context.
#[rustler::nif(schedule = "DirtyIo")]
fn eval_program_on(
    prog: ResourceArc<NifProgram>,
    buffers: Vec<Option<ResourceArc<NifDeviceBuffer>>>,
    queue: ResourceArc<NifQueue>,
) -> Result<Vec<ResourceArc<NifDeviceBuffer>>, String> {
    guarded(|| {
        if !Arc::ptr_eq(&prog.ctx, &queue.ctx) {
            return Err("the queue belongs to another device context".to_string());
        }
        let externals = buffers
            .iter()
            .map(|b| b.as_ref().map(|b| b.get()).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = prog.program.eval(&prog.ctx.ctx, &queue.queue, externals)?;
        Ok(wrap_outputs(&prog.ctx, outputs))
    })
}

fn wrap_outputs(
    ctx: &Arc<DeviceCtxInner>,
    outputs: Vec<Arc<DeviceBuffer>>,
) -> Vec<ResourceArc<NifDeviceBuffer>> {
    outputs
        .into_iter()
        .map(|buffer| ResourceArc::new(NifDeviceBuffer::new(buffer, ctx.clone())))
        .collect()
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
