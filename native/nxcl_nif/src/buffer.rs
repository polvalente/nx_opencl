use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_WRITE};
use opencl3::types::CL_TRUE;
use std::ffi::c_void;

use crate::device_ctx::DeviceCtx;

pub struct DeviceBuffer {
    pub buffer: Buffer<u8>,
    pub size: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn create(ctx: &DeviceCtx, size: usize) -> Result<Self, String> {
        let buffer = unsafe {
            Buffer::<u8>::create(&ctx.context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
        }
        .map_err(|e| format!("Failed to create buffer: {:?}", e))?;

        Ok(Self { buffer, size })
    }

    pub fn create_with_data(ctx: &DeviceCtx, data: &[u8]) -> Result<Self, String> {
        let buffer = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr() as *mut c_void,
            )
        }
        .map_err(|e| format!("Failed to create buffer with data: {:?}", e))?;

        Ok(Self {
            buffer,
            size: data.len(),
        })
    }

    pub fn read(&self, ctx: &DeviceCtx, size: usize) -> Result<Vec<u8>, String> {
        let read_size = std::cmp::min(size, self.size);
        let mut output = vec![0u8; read_size];

        let queue = ctx
            .queue
            .lock()
            .map_err(|e| format!("Lock error: {:?}", e))?;

        unsafe { queue.enqueue_read_buffer(&self.buffer, CL_TRUE, 0, &mut output, &[]) }
            .map_err(|e| format!("Failed to read buffer: {:?}", e))?;

        Ok(output)
    }
}
