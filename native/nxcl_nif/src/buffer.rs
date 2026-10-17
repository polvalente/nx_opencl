use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_WRITE};
use opencl3::types::{CL_FALSE, CL_TRUE};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::device_ctx::DeviceCtx;

pub struct DeviceBuffer {
    pub buffer: Buffer<u8>,
    pub size: usize,
    live: Arc<AtomicUsize>,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn create(ctx: &DeviceCtx, size: usize) -> Result<Self, String> {
        // OpenCL rejects a zero-size buffer; an empty tensor gets one spare byte.
        let buffer = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_WRITE,
                size.max(1),
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| format!("Failed to create buffer: {:?}", e))?;

        ctx.live_buffers.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            buffer,
            size,
            live: ctx.live_buffers.clone(),
        })
    }

    pub fn create_with_data(ctx: &DeviceCtx, data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Self::create(ctx, 0);
        }
        let buffer = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr() as *mut c_void,
            )
        }
        .map_err(|e| format!("Failed to create buffer with data: {:?}", e))?;

        ctx.live_buffers.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            buffer,
            size: data.len(),
            live: ctx.live_buffers.clone(),
        })
    }

    pub fn read(&self, ctx: &DeviceCtx, size: usize) -> Result<Vec<u8>, String> {
        let read_size = std::cmp::min(size, self.size);
        if read_size == 0 {
            return Ok(Vec::new());
        }
        let mut output = vec![0u8; read_size];

        let queue = ctx
            .queue
            .lock()
            .map_err(|e| format!("Lock error: {:?}", e))?;

        unsafe { queue.enqueue_read_buffer(&self.buffer, CL_TRUE, 0, &mut output, &[]) }
            .map_err(|e| format!("Failed to read buffer: {:?}", e))?;

        Ok(output)
    }

    /// Reads several buffers with one wait: the reads queue without blocking and a
    /// single `finish` ends them. Each item is a buffer and a byte count.
    pub fn read_many(
        ctx: &DeviceCtx,
        items: &[(Arc<DeviceBuffer>, usize)],
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut outputs: Vec<Vec<u8>> = items
            .iter()
            .map(|(buf, size)| vec![0u8; std::cmp::min(*size, buf.size)])
            .collect();

        let queue = ctx
            .queue
            .lock()
            .map_err(|e| format!("Lock error: {:?}", e))?;

        let mut events = Vec::with_capacity(items.len());
        let mut failure = None;
        for ((buf, _), output) in items.iter().zip(outputs.iter_mut()) {
            if output.is_empty() {
                continue;
            }
            match unsafe { queue.enqueue_read_buffer(&buf.buffer, CL_FALSE, 0, output, &[]) } {
                Ok(event) => events.push(event),
                Err(e) => {
                    failure = Some(format!("Failed to read buffer: {:?}", e));
                    break;
                }
            }
        }

        // The output vectors must outlive every queued read, even after an error.
        let finished = queue.finish().map_err(|e| format!("Device error: {:?}", e));
        drop(events);
        if let Some(e) = failure {
            return Err(e);
        }
        finished?;
        Ok(outputs)
    }
}
