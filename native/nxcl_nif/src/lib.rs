mod buffer;
mod codegen;
mod device_ctx;
mod kernel;
mod program;

use buffer::DeviceBuffer;
use device_ctx::{lock, DeviceCtx};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use rustler::{Binary, Env, NifResult, OwnedBinary, ResourceArc, Term};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

// Error convention: a fallible NIF returns `{:error, message}` and never
// raises. The NIFs that build a handle or read bytes return the bare value on
// success; the program NIFs return `{:ok, value}`. Elixir callers turn an error
// into an exception.

/// An error the NIF returns as `{:error, message}`.
fn term_err(message: impl Into<String>) -> rustler::Error {
    rustler::Error::Term(Box::new(message.into()))
}

pub struct NifDeviceCtx {
    inner: Arc<DeviceCtx>,
}

/// A device buffer handle. `buffer_free` empties the handle; the device memory
/// goes when the last handle (an aliasing result shares it) lets go.
pub struct NifDeviceBuffer {
    buffer: Mutex<Option<Arc<DeviceBuffer>>>,
    ctx: Arc<DeviceCtx>,
}

impl NifDeviceBuffer {
    fn new(buffer: Arc<DeviceBuffer>, ctx: Arc<DeviceCtx>) -> Self {
        Self {
            buffer: Mutex::new(Some(buffer)),
            ctx,
        }
    }

    fn get(&self) -> Result<Arc<DeviceBuffer>, String> {
        lock(&self.buffer)
            .clone()
            .ok_or_else(|| "buffer was deallocated".to_string())
    }
}

#[rustler::resource_impl]
impl rustler::Resource for NifDeviceCtx {}

/// An extra command queue on a device context. Each one runs a program at a
/// time, and different queues run at the same time.
pub struct NifQueue {
    queue: Mutex<CommandQueue>,
    ctx: Arc<DeviceCtx>,
}

#[rustler::resource_impl]
impl rustler::Resource for NifQueue {}

pub struct NifProgram {
    program: program::Program,
    ctx: Arc<DeviceCtx>,
}

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
    let ctx = DeviceCtx::new(platform, device_id).map_err(term_err)?;
    Ok(ResourceArc::new(NifDeviceCtx {
        inner: Arc::new(ctx),
    }))
}

/// A new command queue on the context, for `eval_program_on`.
#[rustler::nif(schedule = "DirtyIo")]
fn queue_create(ctx: ResourceArc<NifDeviceCtx>) -> NifResult<ResourceArc<NifQueue>> {
    // OpenCL 1.2 entry point, as in `DeviceCtx::new`.
    #[allow(deprecated)]
    let queue = CommandQueue::create_default(&ctx.inner.context, CL_QUEUE_PROFILING_ENABLE)
        .map_err(|e| term_err(format!("Failed to create command queue: {:?}", e)))?;
    Ok(ResourceArc::new(NifQueue {
        queue: Mutex::new(queue),
        ctx: ctx.inner.clone(),
    }))
}

#[rustler::nif(schedule = "DirtyIo")]
fn device_info(ctx: ResourceArc<NifDeviceCtx>) -> NifResult<Vec<(String, String)>> {
    let info = ctx.inner.device_info().map_err(term_err)?;
    let pair = |key: &str, value: String| (key.to_string(), value);
    Ok(vec![
        pair("name", info.name),
        pair("vendor", info.vendor),
        pair("version", info.version),
        pair("driver_version", info.driver_version),
        pair("device_type", info.device_type),
        pair("max_workgroup_size", info.max_workgroup_size.to_string()),
        pair("max_compute_units", info.max_compute_units.to_string()),
        pair("local_mem_size", info.local_mem_size.to_string()),
        pair("global_mem_size", info.global_mem_size.to_string()),
        pair("unified_memory", info.unified_memory.to_string()),
        pair("extensions", info.extensions),
    ])
}

// ── Buffer Operations ───────────────────────────────────

fn new_handle(ctx: &Arc<DeviceCtx>, buf: DeviceBuffer) -> ResourceArc<NifDeviceBuffer> {
    ResourceArc::new(NifDeviceBuffer::new(Arc::new(buf), ctx.clone()))
}

fn binary_from(bytes: &[u8]) -> NifResult<OwnedBinary> {
    let mut binary =
        OwnedBinary::new(bytes.len()).ok_or_else(|| term_err("Failed to allocate binary"))?;
    binary.as_mut_slice().copy_from_slice(bytes);
    Ok(binary)
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_create(
    ctx: ResourceArc<NifDeviceCtx>,
    size: usize,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    let buf = DeviceBuffer::create(&ctx.inner, size).map_err(term_err)?;
    Ok(new_handle(&ctx.inner, buf))
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_write(
    ctx: ResourceArc<NifDeviceCtx>,
    data: rustler::Binary,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    let buf = DeviceBuffer::create_with_data(&ctx.inner, data.as_slice()).map_err(term_err)?;
    Ok(new_handle(&ctx.inner, buf))
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_read(buf: ResourceArc<NifDeviceBuffer>, size: usize) -> NifResult<OwnedBinary> {
    let device = buf.get().map_err(term_err)?;
    let data = device.read(&buf.ctx, size).map_err(term_err)?;
    binary_from(&data)
}

/// Reads several buffers with one wait. `items` is `[{buffer, size}]`.
#[rustler::nif(schedule = "DirtyIo")]
fn buffer_read_many<'a>(
    env: Env<'a>,
    ctx: ResourceArc<NifDeviceCtx>,
    items: Vec<(ResourceArc<NifDeviceBuffer>, usize)>,
) -> NifResult<Vec<Binary<'a>>> {
    let devices = items
        .iter()
        .map(|(buf, size)| buf.get().map(|b| (b, *size)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(term_err)?;
    let data = DeviceBuffer::read_many(&ctx.inner, &devices).map_err(term_err)?;
    data.iter()
        .map(|bytes| Ok(binary_from(bytes)?.release(env)))
        .collect()
}

/// Drops the handle's hold on the device memory. Reading the handle afterwards is
/// an error. Returns false when the handle was already empty.
#[rustler::nif]
fn buffer_free(buf: ResourceArc<NifDeviceBuffer>) -> bool {
    lock(&buf.buffer).take().is_some()
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
        let program = program::Program::decode(&ctx.inner, wire)?;
        Ok(ResourceArc::new(NifProgram {
            program,
            ctx: ctx.inner.clone(),
        }))
    })
}

/// The device buffers behind the handles; `None` marks a slot nothing reads.
fn externals(
    buffers: &[Option<ResourceArc<NifDeviceBuffer>>],
) -> Result<Vec<Option<Arc<DeviceBuffer>>>, String> {
    buffers
        .iter()
        .map(|b| b.as_ref().map(|b| b.get()).transpose())
        .collect()
}

#[rustler::nif(schedule = "DirtyIo")]
fn eval_program(
    prog: ResourceArc<NifProgram>,
    buffers: Vec<Option<ResourceArc<NifDeviceBuffer>>>,
) -> Result<Vec<ResourceArc<NifDeviceBuffer>>, String> {
    guarded(|| {
        let outputs = prog
            .program
            .eval(&prog.ctx, &prog.ctx.queue, externals(&buffers)?)?;
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
        let outputs = prog
            .program
            .eval(&prog.ctx, &queue.queue, externals(&buffers)?)?;
        Ok(wrap_outputs(&prog.ctx, outputs))
    })
}

fn wrap_outputs(
    ctx: &Arc<DeviceCtx>,
    outputs: Vec<Arc<DeviceBuffer>>,
) -> Vec<ResourceArc<NifDeviceBuffer>> {
    outputs
        .into_iter()
        .map(|buffer| ResourceArc::new(NifDeviceBuffer::new(buffer, ctx.clone())))
        .collect()
}

/// Makes the child-process exit status reachable again if the VM ignored it.
///
/// The BEAM sets `SIGCHLD` to `SIG_IGN`, so the kernel reaps a child at once
/// and `wait4` never sees its exit. PoCL builds a kernel by running `clang` and
/// `ld` as child processes and waits on them forever. This sets the disposition
/// to `SIG_DFL` when, and only when, it is `SIG_IGN`; any handler another
/// library installed stays.
///
/// The change is process-wide. A child that any library forks later becomes a
/// zombie until its parent waits for it. The BEAM starts its own ports through
/// `erl_child_setup`, a separate process, and does not rely on the ignore.
///
/// This uses `sigaction` and not `signal`. `sigaction` reads the current
/// disposition without changing it, so there is no moment where the VM's
/// setting is replaced and then put back, and no handler loses its flags or
/// mask. `signal` could only read by setting. The calls take only a pointer to
/// a struct this function owns.
///
/// Unsafe because `libc::sigaction` is a raw system call over C structs. The
/// zeroed `sigaction` is valid (all-zero is `SIG_DFL` with no flags and an empty
/// mask on every platform `libc` supports), and both pointers outlive the call.
fn restore_sigchld() {
    // SAFETY: see above.
    unsafe {
        let mut current: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(libc::SIGCHLD, std::ptr::null(), &mut current) != 0 {
            return;
        }
        if current.sa_sigaction == libc::SIG_IGN {
            let mut default: libc::sigaction = std::mem::zeroed();
            default.sa_sigaction = libc::SIG_DFL;
            libc::sigemptyset(&mut default.sa_mask);
            libc::sigaction(libc::SIGCHLD, &default, std::ptr::null_mut());
        }
    }
}

fn on_load(_env: rustler::Env, _info: rustler::Term) -> bool {
    restore_sigchld();
    true
}

rustler::init!("Elixir.NxCL.Native", load = on_load);

#[cfg(test)]
mod tests {
    use super::restore_sigchld;

    /// The disposition of `SIGCHLD`, as the address the kernel holds.
    fn disposition() -> usize {
        // SAFETY: a zeroed `sigaction` is valid, and a null new action only reads.
        unsafe {
            let mut current: libc::sigaction = std::mem::zeroed();
            assert_eq!(
                libc::sigaction(libc::SIGCHLD, std::ptr::null(), &mut current),
                0
            );
            current.sa_sigaction
        }
    }

    fn set_disposition(handler: usize) {
        // SAFETY: as above; `handler` is `SIG_DFL`, `SIG_IGN` or the address of
        // an `extern "C"` function that takes the signal number.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler;
            libc::sigemptyset(&mut action.sa_mask);
            assert_eq!(
                libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut()),
                0
            );
        }
    }

    extern "C" fn handler(_signal: libc::c_int) {}

    // One test, because the disposition belongs to the whole process.
    #[test]
    fn an_ignored_sigchld_becomes_the_default_and_a_handler_stays() {
        let saved = disposition();

        set_disposition(libc::SIG_IGN);
        restore_sigchld();
        assert_eq!(disposition(), libc::SIG_DFL);

        let custom = handler as extern "C" fn(libc::c_int) as usize;
        set_disposition(custom);
        restore_sigchld();
        assert_eq!(disposition(), custom);

        set_disposition(libc::SIG_DFL);
        restore_sigchld();
        assert_eq!(disposition(), libc::SIG_DFL);

        set_disposition(saved);
    }
}
