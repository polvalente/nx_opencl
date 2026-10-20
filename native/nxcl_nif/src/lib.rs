mod buffer;
mod codegen;
mod device_ctx;
mod kernel;
mod program;

use buffer::DeviceBuffer;
use device_ctx::{lock, DeviceCtx};
use opencl3::command_queue::CommandQueue;
use rustler::{Binary, Env, NifResult, OwnedBinary, ResourceArc, Term};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

// A fallible NIF returns `{:error, message}` and does not raise on bad input.
// Only `compile_program` and the two eval NIFs also turn a panic into an error.
// Handle and read NIFs return the bare value on success; program NIFs return
// `{:ok, value}`.

/// An error the NIF returns as `{:error, message}`.
fn term_err(message: impl Into<String>) -> rustler::Error {
    rustler::Error::Term(Box::new(message.into()))
}

pub struct NifDeviceCtx {
    inner: Arc<DeviceCtx>,
}

/// A device buffer handle. The device memory goes when the last handle to it
/// (an aliasing result shares it) is freed or dropped.
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

/// An extra command queue on a device context. Queues run in parallel; each runs one program at a time.
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
    let queue = DeviceCtx::new_queue(&ctx.inner.context).map_err(term_err)?;
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

fn new_handle(ctx: &Arc<DeviceCtx>, buffer: Arc<DeviceBuffer>) -> ResourceArc<NifDeviceBuffer> {
    ResourceArc::new(NifDeviceBuffer::new(buffer, ctx.clone()))
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
    Ok(new_handle(&ctx.inner, Arc::new(buf)))
}

#[rustler::nif(schedule = "DirtyIo")]
fn buffer_write(
    ctx: ResourceArc<NifDeviceCtx>,
    data: Binary,
) -> NifResult<ResourceArc<NifDeviceBuffer>> {
    let buf = DeviceBuffer::create_with_data(&ctx.inner, data.as_slice()).map_err(term_err)?;
    Ok(new_handle(&ctx.inner, Arc::new(buf)))
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
        .map(|(buf, size)| {
            same_ctx(&buf.ctx, &ctx.inner)?;
            buf.get().map(|b| (b, *size))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(term_err)?;
    let data = DeviceBuffer::read_many(&ctx.inner, &devices).map_err(term_err)?;
    data.iter()
        .map(|bytes| Ok(binary_from(bytes)?.release(env)))
        .collect()
}

/// Drops the handle's hold on the device memory. Returns false if already empty.
#[rustler::nif]
fn buffer_free(buf: ResourceArc<NifDeviceBuffer>) -> bool {
    lock(&buf.buffer).take().is_some()
}

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

/// Fails unless both handles belong to one device context. A `cl_mem` from
/// another context is undefined behavior in OpenCL; some drivers reject it and
/// some do not.
fn same_ctx(a: &Arc<DeviceCtx>, b: &Arc<DeviceCtx>) -> Result<(), String> {
    if Arc::ptr_eq(a, b) {
        Ok(())
    } else {
        Err("the buffer belongs to another device context".to_string())
    }
}

/// The device buffers behind the handles, all from context `ctx`; `None` marks
/// a slot nothing reads.
fn externals(
    ctx: &Arc<DeviceCtx>,
    buffers: &[Option<ResourceArc<NifDeviceBuffer>>],
) -> Result<Vec<Option<Arc<DeviceBuffer>>>, String> {
    buffers
        .iter()
        .map(|b| {
            b.as_ref()
                .map(|b| same_ctx(&b.ctx, ctx).and_then(|_| b.get()))
                .transpose()
        })
        .collect()
}

#[rustler::nif(schedule = "DirtyIo")]
fn eval_program(
    prog: ResourceArc<NifProgram>,
    buffers: Vec<Option<ResourceArc<NifDeviceBuffer>>>,
) -> Result<Vec<ResourceArc<NifDeviceBuffer>>, String> {
    guarded(|| {
        let outputs =
            prog.program
                .eval(&prog.ctx, &prog.ctx.queue, externals(&prog.ctx, &buffers)?)?;
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
        let outputs =
            prog.program
                .eval(&prog.ctx, &queue.queue, externals(&prog.ctx, &buffers)?)?;
        Ok(wrap_outputs(&prog.ctx, outputs))
    })
}

fn wrap_outputs(
    ctx: &Arc<DeviceCtx>,
    outputs: Vec<Arc<DeviceBuffer>>,
) -> Vec<ResourceArc<NifDeviceBuffer>> {
    outputs
        .into_iter()
        .map(|buffer| new_handle(ctx, buffer))
        .collect()
}

/// Resets `SIGCHLD` to `SIG_DFL` when the VM left it at `SIG_IGN`.
///
/// With `SIG_IGN` the kernel reaps a child at once, so `wait4` fails with
/// `ECHILD` instead of returning the exit status. PoCL runs `clang` and `ld` as
/// children to build a kernel and cannot tell whether they succeeded. A handler
/// another library installed stays.
///
/// The change is process-wide: a child forked later by any library stays a
/// zombie until its parent waits. The BEAM starts its ports through
/// `erl_child_setup`, which does not rely on the ignore.
///
/// This uses `sigaction`, which reads the disposition without changing it, so no
/// handler loses its flags or mask. `signal` can only read by setting.
///
/// If the second `sigaction` fails the disposition stays as it was; there is
/// nothing better to do on load, so its result is ignored.
fn restore_sigchld() {
    // SAFETY: `libc::sigaction` is a raw system call over C structs. An
    // all-zero `sigaction` is valid (`SIG_DFL`, no flags, empty mask) on every
    // platform `libc` supports, and every pointer passed points to a local or
    // is null, and outlives the call.
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
