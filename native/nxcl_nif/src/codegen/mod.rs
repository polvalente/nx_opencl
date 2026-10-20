//! Kernel generation. A decoded instruction (op, operand and output types and
//! shapes, attrs) goes in; OpenCL C source, entry name, build options, work sizes
//! and arguments come out. Each op family has a module; N-d index math is in `index.rs`.

pub mod complex;
pub mod conv;
pub mod elementwise;
pub mod fft;
pub mod gemm;
pub mod index;
pub mod indexing;
pub mod linalg;
pub mod movement;
pub mod reduce;
pub mod scan;
pub mod sort;
pub mod types;
pub mod window;

use types::DType;

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

impl Instr {
    /// The types the instruction touches: the operands', then the output's.
    pub fn dtypes(&self) -> Vec<DType> {
        let mut dtypes: Vec<DType> = self.operands.iter().map(|o| o.dtype).collect();
        dtypes.push(self.out);
        dtypes
    }
}

/// One kernel argument, in order. `Operand(i)` is a buffer (the i-th operand).
#[derive(Clone, Debug, PartialEq)]
pub enum KernelArg {
    Operand(usize),
    Scalar(Vec<u8>),
    Output,
    Scratch(usize),
    /// A read-only table the host built (twiddles, chirp), uploaded at decode.
    Table(Vec<u8>),
    /// The program's error flag, one `int` the kernel sets to a nonzero code on a
    /// bad index. The run fails after the queue drains if it is set.
    ErrorFlag,
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

/// Workgroup limits of the device. Cooperative kernels size groups and local arrays from these.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// `CL_DEVICE_MAX_WORK_GROUP_SIZE`.
    pub max_wg: usize,
    /// `CL_DEVICE_LOCAL_MEM_SIZE`, in bytes.
    pub local_mem: usize,
    /// `CL_DEVICE_MAX_WORK_ITEM_SIZES`: the most items a group has along each
    /// of the first three dimensions.
    pub max_item_sizes: [usize; 3],
}

/// The group size codegen aims for when the device allows it.
const PREFERRED_WG: usize = 256;

impl Limits {
    /// Largest power of two that is at most `PREFERRED_WG` and the device
    /// maximum (in total and along the first dimension), and whose local
    /// arrays (`item_bytes` per work item) fit in local memory. At least 1.
    pub fn group_size(&self, item_bytes: usize) -> usize {
        let cap = PREFERRED_WG.min(self.max_wg).min(self.max_item_sizes[0]);
        let mut wg = 1;
        while wg * 2 <= cap && wg * 2 * item_bytes.max(1) <= self.local_mem {
            wg *= 2;
        }
        wg
    }

    /// Apple's OpenCL GPU: 256 threads, 32 KiB of local memory. For source tests.
    #[cfg(test)]
    pub fn apple() -> Self {
        Self::new(256, 32 * 1024)
    }

    /// Limits whose per-dimension size equals the group size. For source tests.
    #[cfg(test)]
    pub fn new(max_wg: usize, local_mem: usize) -> Self {
        Self {
            max_wg,
            local_mem,
            max_item_sizes: [max_wg; 3],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub fp64: bool,
    pub fp16: bool,
    /// `cl_khr_int64_base_atomics`: 64-bit `atom_add` and `atom_cmpxchg`.
    pub int64_atomics: bool,
}

impl Caps {
    pub fn from_extensions(extensions: &str) -> Self {
        Self {
            fp64: extensions.split_whitespace().any(|e| e == "cl_khr_fp64"),
            fp16: extensions.split_whitespace().any(|e| e == "cl_khr_fp16"),
            int64_atomics: extensions
                .split_whitespace()
                .any(|e| e == "cl_khr_int64_base_atomics"),
        }
    }

    /// A device with every extension codegen knows. For source tests.
    #[cfg(test)]
    pub fn all() -> Self {
        Self {
            fp64: true,
            fp16: true,
            int64_atomics: true,
        }
    }

    pub fn has(&self, ext: types::Ext) -> bool {
        match ext {
            types::Ext::Fp64 => self.fp64,
            types::Ext::Fp16 => self.fp16,
        }
    }
}

/// Elements in `shape`, or `usize::MAX` when the count does not fit (never a wrapped value).
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().fold(1usize, |n, &d| n.saturating_mul(d))
}

/// Largest value a size may take. Every size reaches a kernel as a `long`.
const MAX_SIZE: usize = i64::MAX as usize;

/// Elements in `shape`, or an error when the count does not fit in a `long`.
pub fn checked_element_count(shape: &[usize]) -> Result<usize, String> {
    let mut n = 1usize;
    for &d in shape {
        n = n
            .checked_mul(d)
            .filter(|&n| n <= MAX_SIZE)
            .ok_or_else(|| format!("shape {:?} has too many elements", shape))?;
    }
    Ok(n)
}

/// The product of `factors`, or an error naming `what` when it does not fit
/// in a `long`.
pub fn checked_product(what: &str, factors: &[usize]) -> Result<usize, String> {
    let mut n = 1usize;
    for &f in factors {
        n = n
            .checked_mul(f)
            .filter(|&n| n <= MAX_SIZE)
            .ok_or_else(|| format!("{} is too large (factors {:?})", what, factors))?;
    }
    Ok(n)
}

pub(super) fn entry_name(prefix: &str, parts: &[&str]) -> String {
    let mut name = prefix.to_string();
    for part in parts {
        name.push('_');
        name.push_str(part);
    }
    name
}

pub fn long_arg(v: usize) -> Result<Vec<u8>, String> {
    i64::try_from(v)
        .map(|v| v.to_ne_bytes().to_vec())
        .map_err(|_| format!("size {} does not fit in a long", v))
}

// Op-family predicates. The decoder sends an op to the first family that
// claims it.

pub fn is_elementwise(op: &str) -> bool {
    elementwise::handles(op)
}

pub fn is_movement(op: &str) -> bool {
    movement::handles(op)
}

/// `sum`, `argmax`, generic `reduce` and the like.
pub fn is_reduction(op: &str) -> bool {
    reduce::handles(op)
}

/// The cumulative ops.
pub fn is_scan(op: &str) -> bool {
    scan::handles(op)
}

pub fn is_window(op: &str) -> bool {
    window::handles(op)
}

/// `sort` and `argsort`.
pub fn is_sort(op: &str) -> bool {
    sort::handles(op)
}

/// `gather`, `indexed_add` and `indexed_put`.
pub fn is_indexing(op: &str) -> bool {
    indexing::handles(op)
}

/// `fft` and `ifft`.
pub fn is_fft(op: &str) -> bool {
    fft::handles(op)
}

/// `triangular_solve` and `cholesky`.
pub fn is_linalg(op: &str) -> bool {
    linalg::handles(op)
}

/// `batched_gemm` and `conv`.
pub fn is_gemm(op: &str) -> bool {
    gemm::handles(op) || conv::handles(op)
}

pub fn generate_plan(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    if reduce::handles(&instr.op) {
        reduce::generate(instr, caps, limits)
    } else if scan::handles(&instr.op) {
        scan::generate(instr, caps, limits)
    } else if window::handles(&instr.op) {
        window::generate(instr, caps)
    } else if sort::handles(&instr.op) {
        sort::generate(instr, caps, limits)
    } else if indexing::handles(&instr.op) {
        indexing::generate(instr, caps)
    } else if fft::handles(&instr.op) {
        fft::generate(instr, caps)
    } else if linalg::handles(&instr.op) {
        linalg::generate(instr, caps, limits)
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

/// Helpers the unit tests of the codegen modules share.
#[cfg(test)]
pub mod test_support {
    use super::*;

    pub fn buf(dtype: DType, shape: &[usize]) -> Operand {
        Operand {
            dtype,
            shape: shape.to_vec(),
            constant: None,
        }
    }

    pub fn ints(v: &[i64]) -> Vec<Attr> {
        v.iter().map(|&i| Attr::Int(i)).collect()
    }

    pub fn instr(
        op: &str,
        operands: Vec<Operand>,
        out: DType,
        out_shape: &[usize],
        attrs: Vec<Attr>,
    ) -> Instr {
        Instr {
            op: op.to_string(),
            operands,
            out,
            out_shape: out_shape.to_vec(),
            attrs,
        }
    }

    /// The plan on a device with every extension and Apple's limits.
    pub fn plan(i: &Instr) -> Plan {
        generate_plan(i, &Caps::all(), &Limits::apple()).unwrap()
    }

    /// The error of a plan that must fail.
    pub fn plan_err(i: &Instr) -> String {
        generate_plan(i, &Caps::all(), &Limits::apple()).unwrap_err()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::types::Kind;
    use super::*;

    #[test]
    fn element_counts_saturate_and_checked_counts_error() {
        assert_eq!(element_count(&[2, 3, 4]), 24);
        assert_eq!(element_count(&[usize::MAX, 2]), usize::MAX);
        // A zero anywhere still gives zero.
        assert_eq!(element_count(&[usize::MAX, usize::MAX, 0]), 0);
        assert_eq!(checked_element_count(&[2, 3, 4]), Ok(24));
        let err = checked_element_count(&[1 << 40, 1 << 40]).unwrap_err();
        assert!(err.contains("too many elements"), "{err}");
        assert!(checked_element_count(&[usize::MAX]).is_err());
    }

    #[test]
    fn checked_products_name_what_overflowed() {
        assert_eq!(checked_product("x", &[3, 4, 5]), Ok(60));
        let err = checked_product("the scratch size", &[usize::MAX, 2]).unwrap_err();
        assert!(err.contains("the scratch size is too large"), "{err}");
        assert!(checked_product("x", &[1 << 62, 4]).is_err());
    }

    #[test]
    fn long_args_hold_what_a_long_holds() {
        assert_eq!(long_arg(7).unwrap(), 7i64.to_ne_bytes().to_vec());
        #[cfg(target_pointer_width = "64")]
        assert!(long_arg(usize::MAX).unwrap_err().contains("long"));
    }

    #[test]
    fn entry_names_join_the_parts() {
        assert_eq!(entry_name("rd", &["sum", "f32"]), "rd_sum_f32");
    }

    #[test]
    fn an_unknown_op_is_an_error() {
        let f32 = DType::new(Kind::F, 32);
        let i = instr("no_such_op", vec![buf(f32, &[2])], f32, &[2], vec![]);
        assert!(plan_err(&i).contains("no codegen for op no_such_op"));
    }

    #[test]
    fn the_group_size_obeys_every_limit() {
        let mut l = Limits::new(256, 32 * 1024);
        assert_eq!(l.group_size(4), 256);
        // The first dimension may be narrower than the total.
        l.max_item_sizes = [64, 64, 64];
        assert_eq!(l.group_size(4), 64);
        // Local memory bounds it too: 8 bytes per item in 256 bytes.
        assert_eq!(Limits::new(256, 256).group_size(8), 32);
        assert_eq!(Limits::new(1, 0).group_size(0), 1);
    }
}
