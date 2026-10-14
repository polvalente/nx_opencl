//! Kernel generation. A decoded instruction (op, operand and output types and
//! shapes, attrs) goes in; OpenCL C source, an entry name, build options, work
//! sizes and the argument list come out. Each op family has one module here.
//! Later stages add index math in this file, the one place they do so.

pub mod complex;
pub mod elementwise;
pub mod types;

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
    /// Op attrs. The elementwise family reads none; later families do.
    #[allow(dead_code)]
    pub attrs: Vec<Attr>,
}

/// One kernel argument, in order. `Operand(i)` is a buffer (the i-th operand).
#[derive(Clone, Debug, PartialEq)]
pub enum KernelArg {
    Operand(usize),
    Scalar(Vec<u8>),
    Output,
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

pub fn generate(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    if elementwise::handles(&instr.op) {
        elementwise::generate(instr, caps)
    } else {
        Err(format!("no codegen for op {}", instr.op))
    }
}
