use crate::codegen::Caps;
use crate::kernel::{CompiledKernel, KernelKey};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

pub struct DeviceCtx {
    pub context: Context,
    pub queue: Mutex<CommandQueue>,
    pub device: Device,
    pub unified: bool,
    /// Extensions codegen checks before it emits a kernel.
    pub caps: Caps,
    pub kernels: Mutex<HashMap<KernelKey, Arc<CompiledKernel>>>,
    pub stats: KernelStats,
    pub live_buffers: Arc<AtomicUsize>,
}

/// Counters the tests read through `kernel_cache_stats/1`.
#[derive(Default)]
pub struct KernelStats {
    pub builds: AtomicU64,
    pub hits: AtomicU64,
    pub launches: AtomicU64,
}

unsafe impl Send for DeviceCtx {}
unsafe impl Sync for DeviceCtx {}

impl DeviceCtx {
    pub fn new(device_index: usize) -> Result<Self, String> {
        // Try GPU devices first
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)
            .ok()
            .filter(|ids| !ids.is_empty())
            .map(Ok)
            .unwrap_or_else(|| get_all_devices(CL_DEVICE_TYPE_ALL))
            .map_err(|e| format!("Failed to get devices: {:?}", e))?;

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
            kernels: Mutex::new(HashMap::new()),
            stats: KernelStats::default(),
            live_buffers: Arc::new(AtomicUsize::new(0)),
        })
    }
}

pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub max_workgroup_size: usize,
    pub max_compute_units: u32,
    pub local_mem_size: u64,
    pub global_mem_size: u64,
    pub unified_memory: bool,
    pub extensions: String,
}

impl DeviceCtx {
    pub fn device_info(&self) -> Result<DeviceInfo, String> {
        let name = self.device.name().map_err(|e| format!("{:?}", e))?;
        let vendor = self.device.vendor().map_err(|e| format!("{:?}", e))?;
        let version = self.device.version().map_err(|e| format!("{:?}", e))?;
        let max_workgroup_size = self
            .device
            .max_work_group_size()
            .map_err(|e| format!("{:?}", e))?;
        let max_compute_units = self
            .device
            .max_compute_units()
            .map_err(|e| format!("{:?}", e))?;
        let local_mem_size = self
            .device
            .local_mem_size()
            .map_err(|e| format!("{:?}", e))?;
        let global_mem_size = self
            .device
            .global_mem_size()
            .map_err(|e| format!("{:?}", e))?;

        let extensions = self.device.extensions().map_err(|e| format!("{:?}", e))?;

        Ok(DeviceInfo {
            name,
            vendor,
            version,
            max_workgroup_size,
            max_compute_units,
            local_mem_size,
            global_mem_size,
            unified_memory: self.unified,
            extensions,
        })
    }
}
