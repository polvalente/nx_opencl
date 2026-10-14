//! Kernel generation. A decoded instruction (op, operand and output types and
//! shapes, attrs) goes in; OpenCL C source, an entry name, build options, work
//! sizes and the argument list come out. Each op family has one module here.
//! N-d index math sits in `index.rs`.

pub mod complex;
pub mod conv;
pub mod elementwise;
pub mod gemm;
pub mod index;
pub mod movement;
pub mod reduce;
pub mod scan;
pub mod sort;
pub mod types;
pub mod window;

use types::DType;

/// An instruction attr: an integer, an atom, or a binary.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    Atom(String),
    Bin(Vec<u8>),
}

/// An operand as codegen sees it. A scalar constant carries its element bytes;
/// the kernel takes it as a scalar argument, not a buffer.
#[derive(Clone, Debug)]
pub struct Operand {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub constant: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub op: String,
    pub operands: Vec<Operand>,
    pub out: DType,
    pub out_shape: Vec<usize>,
    /// Op attrs. The elementwise family reads none; movement ops read axes,
    /// lengths, strides and pad configs.
    pub attrs: Vec<Attr>,
}

/// One kernel argument, in order. `Operand(i)` is a buffer (the i-th operand).
#[derive(Clone, Debug, PartialEq)]
pub enum KernelArg {
    Operand(usize),
    Scalar(Vec<u8>),
    Output,
    /// The i-th scratch buffer of the plan.
    Scratch(usize),
}

#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub source: String,
    pub entry: String,
    pub build_options: String,
    pub global: Vec<usize>,
    pub local: Vec<usize>,
    pub args: Vec<KernelArg>,
}

/// The kernels of one instruction, in launch order, and the scratch buffers
/// they share. Scratch sizes are in bytes. Ops that need cross-thread
/// cooperation (reductions, scans, sorts) use more than one pass.
#[derive(Clone, Debug)]
pub struct Plan {
    pub passes: Vec<KernelSpec>,
    pub scratch: Vec<usize>,
}

impl Plan {
    pub fn single(spec: KernelSpec) -> Self {
        Self {
            passes: vec![spec],
            scratch: Vec::new(),
        }
    }
}

/// Workgroup limits of the device. Cooperative kernels size their groups and
/// local arrays from these; none of them is a constant in the templates.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// `CL_DEVICE_MAX_WORK_GROUP_SIZE`.
    pub max_wg: usize,
    /// `CL_DEVICE_LOCAL_MEM_SIZE`, in bytes.
    pub local_mem: usize,
}

/// The group size codegen aims for when the device allows it.
const PREFERRED_WG: usize = 256;

impl Limits {
    /// Largest power of two that is at most `PREFERRED_WG` and the device
    /// maximum, and whose local arrays (`item_bytes` per work item) fit in
    /// local memory. At least 1.
    pub fn group_size(&self, item_bytes: usize) -> usize {
        let mut wg = 1;
        while wg * 2 <= PREFERRED_WG.min(self.max_wg)
            && wg * 2 * item_bytes.max(1) <= self.local_mem
        {
            wg *= 2;
        }
        wg
    }

    /// Apple's OpenCL GPU: 256 threads, 32 KiB of local memory. For source tests.
    #[cfg(test)]
    pub fn apple() -> Self {
        Self {
            max_wg: 256,
            local_mem: 32 * 1024,
        }
    }
}

/// What the device can do, as far as codegen cares.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub fp64: bool,
    pub fp16: bool,
}

impl Caps {
    pub fn from_extensions(extensions: &str) -> Self {
        Self {
            fp64: extensions.split_whitespace().any(|e| e == "cl_khr_fp64"),
            fp16: extensions.split_whitespace().any(|e| e == "cl_khr_fp16"),
        }
    }

    /// A device with every extension codegen knows. For source tests.
    #[cfg(test)]
    pub fn all() -> Self {
        Self {
            fp64: true,
            fp16: true,
        }
    }

    pub fn has(&self, ext: types::Ext) -> bool {
        match ext {
            types::Ext::Fp64 => self.fp64,
            types::Ext::Fp16 => self.fp16,
        }
    }
}

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// True when `op` is built by this module's elementwise family.
pub fn is_elementwise(op: &str) -> bool {
    elementwise::handles(op)
}

/// True when `op` is built by the movement family.
pub fn is_movement(op: &str) -> bool {
    movement::handles(op)
}

/// True when `op` is a reduction (`sum`, `argmax`, generic `reduce`, ...).
pub fn is_reduction(op: &str) -> bool {
    reduce::handles(op)
}

/// True when `op` is a cumulative op.
pub fn is_scan(op: &str) -> bool {
    scan::handles(op)
}

/// True when `op` is a window op.
pub fn is_window(op: &str) -> bool {
    window::handles(op)
}

/// True when `op` is `sort` or `argsort`.
pub fn is_sort(op: &str) -> bool {
    sort::handles(op)
}

/// True when `op` is `batched_gemm` or `conv`.
pub fn is_gemm(op: &str) -> bool {
    gemm::handles(op) || conv::handles(op)
}

/// Generate the kernels for any op this module knows, one or several passes.
pub fn generate_plan(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    if reduce::handles(&instr.op) {
        reduce::generate(instr, caps, limits)
    } else if scan::handles(&instr.op) {
        scan::generate(instr, caps, limits)
    } else if window::handles(&instr.op) {
        window::generate(instr, caps)
    } else if sort::handles(&instr.op) {
        sort::generate(instr, caps, limits)
    } else if gemm::handles(&instr.op) {
        gemm::generate(instr, caps, limits)
    } else if conv::handles(&instr.op) {
        conv::generate(instr, caps, limits)
    } else {
        generate(instr, caps).map(Plan::single)
    }
}

pub fn generate(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    if elementwise::handles(&instr.op) {
        elementwise::generate(instr, caps)
    } else if movement::handles(&instr.op) {
        movement::generate(instr, caps)
    } else {
        Err(format!("no codegen for op {}", instr.op))
    }
}
