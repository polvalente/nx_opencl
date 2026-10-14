use opencl3::kernel::ExecuteKernel;
use opencl3::memory::ClMem;

use crate::buffer::DeviceBuffer;
use crate::device_ctx::DeviceCtx;
use crate::kernel::{
    CompiledKernel, ELEMENTWISE_SOURCE, MATMUL_SOURCE, REDUCE_SOURCE, SOFTMAX_SOURCE,
};

/// Run a binary elementwise operation (add, subtract, multiply, divide)
pub fn elementwise_binary(
    ctx: &DeviceCtx,
    op: &str,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    n: usize,
) -> Result<DeviceBuffer, String> {
    let kernel_name = format!("elementwise_{}", op);
    let kernel = CompiledKernel::compile(ctx, ELEMENTWISE_SOURCE, &kernel_name, "")?;

    let out = DeviceBuffer::create(ctx, n * 4)?; // f32 = 4 bytes

    let queue = ctx
        .queue
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    let global_work_size = round_up(n, 64);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel.kernel)
            .set_arg(&a.buffer.get())
            .set_arg(&b.buffer.get())
            .set_arg(&out.buffer.get())
            .set_arg(&(n as i32))
            .set_global_work_size(global_work_size)
            .set_local_work_size(64)
            .enqueue_nd_range(&queue)
    }
    .map_err(|e| format!("Failed to enqueue kernel: {:?}", e))?;

    kernel_event
        .wait()
        .map_err(|e| format!("Kernel wait error: {:?}", e))?;

    Ok(out)
}

/// Run a unary elementwise operation (negate, exp, log, tanh, sigmoid, relu, rsqrt, abs)
pub fn elementwise_unary(
    ctx: &DeviceCtx,
    op: &str,
    a: &DeviceBuffer,
    n: usize,
) -> Result<DeviceBuffer, String> {
    let kernel_name = format!("elementwise_{}", op);
    let kernel = CompiledKernel::compile(ctx, ELEMENTWISE_SOURCE, &kernel_name, "")?;

    let out = DeviceBuffer::create(ctx, n * 4)?;

    let queue = ctx
        .queue
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    let global_work_size = round_up(n, 64);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel.kernel)
            .set_arg(&a.buffer.get())
            .set_arg(&out.buffer.get())
            .set_arg(&(n as i32))
            .set_global_work_size(global_work_size)
            .set_local_work_size(64)
            .enqueue_nd_range(&queue)
    }
    .map_err(|e| format!("Failed to enqueue kernel: {:?}", e))?;

    kernel_event
        .wait()
        .map_err(|e| format!("Kernel wait error: {:?}", e))?;

    Ok(out)
}

/// Run matrix multiplication: C = A * B where A is [M x K] and B is [K x N]
pub fn matmul(
    ctx: &DeviceCtx,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<DeviceBuffer, String> {
    let ts = 8usize;
    let build_opts = format!("-DTS={}", ts);
    let kernel = CompiledKernel::compile(ctx, MATMUL_SOURCE, "matmul", &build_opts)?;

    let out = DeviceBuffer::create(ctx, m * n * 4)?;

    let queue = ctx
        .queue
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    let global_rows = round_up(m, ts);
    let global_cols = round_up(n, ts);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel.kernel)
            .set_arg(&a.buffer.get())
            .set_arg(&b.buffer.get())
            .set_arg(&out.buffer.get())
            .set_arg(&(m as i32))
            .set_arg(&(n as i32))
            .set_arg(&(k as i32))
            .set_global_work_sizes(&[global_rows, global_cols])
            .set_local_work_sizes(&[ts, ts])
            .enqueue_nd_range(&queue)
    }
    .map_err(|e| format!("Failed to enqueue matmul kernel: {:?}", e))?;

    kernel_event
        .wait()
        .map_err(|e| format!("Kernel wait error: {:?}", e))?;

    Ok(out)
}

/// Run a reduction operation (sum, max, min) over all elements.
/// Returns a single-element buffer.
pub fn reduce(
    ctx: &DeviceCtx,
    op: &str,
    input: &DeviceBuffer,
    n: usize,
) -> Result<DeviceBuffer, String> {
    let kernel_name = format!("reduce_{}", op);
    let kernel = CompiledKernel::compile(ctx, REDUCE_SOURCE, &kernel_name, "")?;

    let local_size: usize = 64;
    let global_size = round_up(n, local_size);
    let num_groups = global_size / local_size;

    // Output buffer for partial results (one per workgroup)
    let partials = DeviceBuffer::create(ctx, num_groups * 4)?;

    let queue = ctx
        .queue
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel.kernel)
            .set_arg(&input.buffer.get())
            .set_arg(&partials.buffer.get())
            .set_arg_local_buffer(local_size * 4) // scratch buffer
            .set_arg(&(n as i32))
            .set_global_work_size(global_size)
            .set_local_work_size(local_size)
            .enqueue_nd_range(&queue)
    }
    .map_err(|e| format!("Failed to enqueue reduce kernel: {:?}", e))?;

    kernel_event
        .wait()
        .map_err(|e| format!("Kernel wait error: {:?}", e))?;

    drop(queue);

    // If more than one workgroup, reduce the partials
    if num_groups > 1 {
        reduce(ctx, op, &partials, num_groups)
    } else {
        Ok(partials)
    }
}

/// Run softmax over the last dimension.
/// Input shape is [rows x cols], softmax is applied per row.
pub fn softmax(
    ctx: &DeviceCtx,
    input: &DeviceBuffer,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer, String> {
    let kernel = CompiledKernel::compile(ctx, SOFTMAX_SOURCE, "softmax", "")?;

    let out = DeviceBuffer::create(ctx, rows * cols * 4)?;

    let queue = ctx
        .queue
        .lock()
        .map_err(|e| format!("Lock error: {:?}", e))?;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel.kernel)
            .set_arg(&input.buffer.get())
            .set_arg(&out.buffer.get())
            .set_arg(&(cols as i32))
            .set_global_work_size(rows)
            .enqueue_nd_range(&queue)
    }
    .map_err(|e| format!("Failed to enqueue softmax kernel: {:?}", e))?;

    kernel_event
        .wait()
        .map_err(|e| format!("Kernel wait error: {:?}", e))?;

    Ok(out)
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}
