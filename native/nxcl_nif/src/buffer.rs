use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_WRITE};
use opencl3::types::{CL_FALSE, CL_TRUE};
use std::ffi::c_void;
use std::sync::Arc;

use crate::device_ctx::{cl_error, lock, DeviceCtx};

pub struct DeviceBuffer {
    pub buffer: Buffer<u8>,
    pub size: usize,
}

impl DeviceBuffer {
    pub fn create(ctx: &DeviceCtx, size: usize) -> Result<Self, String> {
        if size > isize::MAX as usize {
            return Err(format!("buffer of {} bytes is too large", size));
        }
        // OpenCL rejects a zero-size buffer; an empty tensor gets one spare byte.
        // SAFETY: the host pointer is null and no host-pointer flag is set, so
        // OpenCL allocates the memory itself and reads nothing from the host.
        let buffer = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_WRITE,
                size.max(1),
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| cl_error("Failed to create buffer", e))?;

        Ok(Self { buffer, size })
    }

    pub fn create_with_data(ctx: &DeviceCtx, data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Self::create(ctx, 0);
        }
        // SAFETY: `CL_MEM_COPY_HOST_PTR` copies `data.len()` bytes from a pointer
        // that is valid for that many bytes. OpenCL does not keep the pointer
        // after the call, and does not write through it.
        let buffer = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr() as *mut c_void,
            )
        }
        .map_err(|e| cl_error("Failed to create buffer with data", e))?;

        Ok(Self {
            buffer,
            size: data.len(),
        })
    }

    /// `size`, cut to the buffer's size.
    fn clamp(&self, size: usize) -> usize {
        std::cmp::min(size, self.size)
    }

    /// Reads up to `size` bytes through the context's own queue. A run on that
    /// queue may still be queued when this is called; the read comes after it.
    /// A run on another queue always finishes before it returns, so its output
    /// is ready for this read too.
    pub fn read(&self, ctx: &DeviceCtx, size: usize) -> Result<Vec<u8>, String> {
        let read_size = self.clamp(size);
        if read_size == 0 {
            return Ok(Vec::new());
        }
        let mut output = vec![0u8; read_size];

        let queue = lock(&ctx.queue);

        // SAFETY: `output` holds `read_size` bytes, at most the buffer's size,
        // and the read is blocking, so it finishes before `output` moves or drops.
        unsafe { queue.enqueue_read_buffer(&self.buffer, CL_TRUE, 0, &mut output, &[]) }
            .map_err(|e| cl_error("Failed to read buffer", e))?;

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
            .map(|(buf, size)| vec![0u8; buf.clamp(*size)])
            .collect();

        let queue = lock(&ctx.queue);

        let mut events = Vec::with_capacity(items.len());
        let mut failure = None;
        for ((buf, _), output) in items.iter().zip(outputs.iter_mut()) {
            if output.is_empty() {
                continue;
            }
            // SAFETY: the read is not blocking, so `output` must stay alive and
            // unmoved until the queue drains. The vectors in `outputs` are not
            // touched again until `finish` below has returned, on every path.
            match unsafe { queue.enqueue_read_buffer(&buf.buffer, CL_FALSE, 0, output, &[]) } {
                Ok(event) => events.push(event),
                Err(e) => {
                    failure = Some(cl_error("Failed to read buffer", e));
                    break;
                }
            }
        }

        // The output vectors must outlive every queued read, even after an error.
        let finished = queue.finish().map_err(|e| cl_error("Device error", e));
        drop(events);
        if let Err(e) = finished {
            // The driver may still write into the vectors, so they leak.
            std::mem::forget(outputs);
            return Err(failure.unwrap_or(e));
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(outputs),
        }
    }
}
