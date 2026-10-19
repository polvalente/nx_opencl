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
    /// Kernels built for this context, by source, entry and options. The cache
    /// only grows. One context holds one kernel per distinct generated source,
    /// and a program re-uses those, so the count follows the number of distinct
    /// ops a session runs, not the number of runs.
    pub kernels: Mutex<HashMap<KernelKey, Arc<CompiledKernel>>>,
}

/// Lock `m`, ignoring poison. A panic while a lock was held (the NIF entry
/// points catch panics) must not make the queue or the kernel cache unusable;
/// neither holds data a half-finished panic could leave inconsistent.
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
                    get_platforms().map_err(|e| format!("Failed to get platforms: {:?}", e))?;
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

        let unified = device.host_unified_memory().unwrap_or(false);
        let caps = Caps::from_extensions(&device.extensions().unwrap_or_default());

        let max_wg = device.max_work_group_size().unwrap_or(1);
        let sizes = device.max_work_item_sizes().unwrap_or_default();
        let limits = Limits {
            max_wg,
            local_mem: device.local_mem_size().unwrap_or(0) as usize,
            // A driver that reports fewer than three dimensions leaves the rest
            // at the group size, the loosest value the group limit allows.
            max_item_sizes: std::array::from_fn(|d| sizes.get(d).copied().unwrap_or(max_wg)),
        };

        let context = Context::from_device(&device)
            .map_err(|e| format!("Failed to create context: {:?}", e))?;

        // OpenCL 1.2 entry point: Apple's driver has no clCreateCommandQueueWithProperties.
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| format!("Failed to create command queue: {:?}", e))?;

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
}

/// The GPUs when there are any, else every device. `gpus` is the GPU query
/// and `all` the query for every type, run only when needed.
fn pick_devices(
    gpus: opencl3::Result<Vec<cl_device_id>>,
    all: impl FnOnce() -> opencl3::Result<Vec<cl_device_id>>,
) -> Result<Vec<cl_device_id>, String> {
    match gpus {
        Ok(ids) if !ids.is_empty() => Ok(ids),
        _ => all().map_err(|e| format!("Failed to get devices: {:?}", e)),
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

/// Formats an OpenCL error for a device-info query.
fn dbg<E: std::fmt::Debug>(e: E) -> String {
    format!("{:?}", e)
}

impl DeviceCtx {
    pub fn device_info(&self) -> Result<DeviceInfo, String> {
        let d = &self.device;
        let device_type = match d.dev_type().map_err(dbg)? {
            t if t & CL_DEVICE_TYPE_GPU != 0 => "GPU",
            t if t & CL_DEVICE_TYPE_CPU != 0 => "CPU",
            t if t & CL_DEVICE_TYPE_ACCELERATOR != 0 => "ACCELERATOR",
            _ => "OTHER",
        }
        .to_string();
        Ok(DeviceInfo {
            name: d.name().map_err(dbg)?,
            vendor: d.vendor().map_err(dbg)?,
            version: d.version().map_err(dbg)?,
            driver_version: d.driver_version().map_err(dbg)?,
            device_type,
            max_workgroup_size: d.max_work_group_size().map_err(dbg)?,
            max_compute_units: d.max_compute_units().map_err(dbg)?,
            local_mem_size: d.local_mem_size().map_err(dbg)?,
            global_mem_size: d.global_mem_size().map_err(dbg)?,
            unified_memory: self.unified,
            extensions: d.extensions().map_err(dbg)?,
        })
    }
}
