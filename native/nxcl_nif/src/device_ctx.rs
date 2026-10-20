use crate::codegen::{Caps, Limits};
use crate::kernel::{CompiledKernel, KernelKey};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{
    get_all_devices, Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
    CL_DEVICE_TYPE_GPU,
};
use opencl3::platform::get_platforms;
use opencl3::types::cl_device_id;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

pub struct DeviceCtx {
    pub context: Context,
    /// The context's own queue. Lock it with `lock`.
    pub queue: Mutex<CommandQueue>,
    pub device: Device,
    pub unified: bool,
    /// Extensions codegen checks before it emits a kernel.
    pub caps: Caps,
    /// Workgroup and local-memory limits codegen sizes cooperative kernels from.
    pub limits: Limits,
    /// Kernels built for this context, by source, entry and options. It only
    /// grows, with the number of distinct generated sources.
    pub kernels: Mutex<HashMap<KernelKey, Arc<CompiledKernel>>>,
}

/// Formats an OpenCL error as `what: message`. `Display` prints the error name
/// (`CL_INVALID_WORK_GROUP_SIZE`) and, for a build failure, the compiler log
/// as it is.
pub fn cl_error(what: &str, e: impl std::fmt::Display) -> String {
    format!("{}: {}", what, e)
}

/// Lock `m`, ignoring poison. The NIF entry points catch panics, and neither
/// the queue nor the kernel cache holds data a panic could leave inconsistent.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl DeviceCtx {
    /// Opens device `device_index`. With `platform` set, the index counts the
    /// devices of that platform; otherwise it counts every device, GPUs first.
    pub fn new(platform: Option<usize>, device_index: usize) -> Result<Self, String> {
        let device_ids = match platform {
            Some(p) => {
                let platforms =
                    get_platforms().map_err(|e| cl_error("Failed to get platforms", e))?;
                let platform = platforms.get(p).ok_or_else(|| {
                    format!(
                        "Platform index {} out of range (found {} platforms)",
                        p,
                        platforms.len()
                    )
                })?;
                pick_devices(platform.get_devices(CL_DEVICE_TYPE_GPU), || {
                    platform.get_devices(CL_DEVICE_TYPE_ALL)
                })?
            }
            None => pick_devices(get_all_devices(CL_DEVICE_TYPE_GPU), || {
                get_all_devices(CL_DEVICE_TYPE_ALL)
            })?,
        };

        if device_ids.is_empty() {
            return Err("No OpenCL devices found".to_string());
        }

        if device_index >= device_ids.len() {
            return Err(format!(
                "Device index {} out of range (found {} devices)",
                device_index,
                device_ids.len()
            ));
        }

        let device = Device::new(device_ids[device_index]);

        // OpenCL 2.0 deprecated this query, so a driver may reject it: such a
        // device counts as not unified, which only costs a copy.
        let unified = device.host_unified_memory().unwrap_or(false);
        // The other queries must work: a wrong answer would change the kernels
        // codegen emits.
        let query = |e| cl_error("Failed to query the device", e);
        let caps = Caps::from_extensions(&device.extensions().map_err(query)?);

        let max_wg = device.max_work_group_size().map_err(query)?;
        let sizes = device.max_work_item_sizes().map_err(query)?;
        let limits = Limits {
            max_wg,
            local_mem: device.local_mem_size().map_err(query)? as usize,
            // A driver that reports fewer than three dimensions leaves the rest
            // at the group size, the loosest value the group limit allows.
            max_item_sizes: std::array::from_fn(|d| sizes.get(d).copied().unwrap_or(max_wg)),
        };

        let context =
            Context::from_device(&device).map_err(|e| cl_error("Failed to create context", e))?;
        let queue = Self::new_queue(&context)?;

        Ok(Self {
            context,
            queue: Mutex::new(queue),
            device,
            unified,
            caps,
            limits,
            kernels: Mutex::new(HashMap::new()),
        })
    }

    /// A new command queue on `context`.
    pub fn new_queue(context: &Context) -> Result<CommandQueue, String> {
        // OpenCL 1.2 entry point: Apple's driver has no clCreateCommandQueueWithProperties.
        #[allow(deprecated)]
        CommandQueue::create_default(context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| cl_error("Failed to create command queue", e))
    }
}

/// The GPUs when there are any, else every device. `gpus` is the GPU query
/// and `all` the query for every type, run only when needed.
fn pick_devices(
    gpus: opencl3::Result<Vec<cl_device_id>>,
    all: impl FnOnce() -> opencl3::Result<Vec<cl_device_id>>,
) -> Result<Vec<cl_device_id>, String> {
    match gpus {
        Ok(ids) if !ids.is_empty() => Ok(ids),
        _ => all().map_err(|e| cl_error("Failed to get devices", e)),
    }
}

pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub driver_version: String,
    pub device_type: String,
    pub max_workgroup_size: usize,
    pub max_compute_units: u32,
    pub local_mem_size: u64,
    pub global_mem_size: u64,
    pub unified_memory: bool,
    pub extensions: String,
}

impl DeviceCtx {
    pub fn device_info(&self) -> Result<DeviceInfo, String> {
        let d = &self.device;
        let query = |e| cl_error("Failed to query the device", e);
        let device_type = match d.dev_type().map_err(query)? {
            t if t & CL_DEVICE_TYPE_GPU != 0 => "GPU",
            t if t & CL_DEVICE_TYPE_CPU != 0 => "CPU",
            t if t & CL_DEVICE_TYPE_ACCELERATOR != 0 => "ACCELERATOR",
            _ => "OTHER",
        }
        .to_string();
        Ok(DeviceInfo {
            name: d.name().map_err(query)?,
            vendor: d.vendor().map_err(query)?,
            version: d.version().map_err(query)?,
            driver_version: d.driver_version().map_err(query)?,
            device_type,
            max_workgroup_size: d.max_work_group_size().map_err(query)?,
            max_compute_units: d.max_compute_units().map_err(query)?,
            local_mem_size: d.local_mem_size().map_err(query)?,
            global_mem_size: d.global_mem_size().map_err(query)?,
            unified_memory: self.unified,
            extensions: d.extensions().map_err(query)?,
        })
    }
}
