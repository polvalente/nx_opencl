//! Straight-line program executor.
//!
//! `Program::decode` turns the wire term built by `NxCL.Executor` into steps and
//! builds every kernel through the device cache. `Program::eval` enqueues the
//! steps in order on the one in-order queue and waits once, in `finish`.
//!
//! Wire term: `{inputs, captures, consts, [instr], [out_ref]}` where
//! `inputs = [{type, shape} | nil]`, `captures = [{type, shape}]`,
//! `consts = [{type, value}]`, `instr = {op, [ref], out_type, out_shape, [attr]}`,
//! `type = {kind, bits}`, `shape` is a tuple and `ref = {kind, index}`. Rust
//! keeps the type and shape of every slot, so an instruction needs no operand
//! types of its own.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::ClMem;
use rustler::types::tuple::get_tuple;
use rustler::{Binary, Term};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::buffer::DeviceBuffer;
use crate::codegen::types::{encode_const, type_info, ConstVal, DType, Kind};
use crate::codegen::{self, Attr, Instr, KernelArg, Operand};
use crate::device_ctx::DeviceCtx;
use crate::kernel::{self, CompiledKernel, MATMUL_SOURCE};

const MATMUL_TILE: usize = 8;
const MAX_DIMS: usize = 3;

/// One kernel argument.
enum Arg {
    Slot(usize),
    /// The output buffer of the step.
    Out,
    /// A scalar passed by value: the bytes of one element, 1, 2, 4 or 8 long.
    Scalar(Vec<u8>),
}

enum Step {
    /// The result shares the operand's buffer.
    Alias { src: usize, out: usize },
    Launch {
        kernel: Arc<CompiledKernel>,
        args: Vec<Arg>,
        out: usize,
        out_bytes: usize,
        global: Vec<usize>,
        local: Vec<usize>,
    },
}

pub struct Program {
    n_external: usize,
    steps: Vec<Step>,
    outputs: Vec<usize>,
    /// Slots to drop after each step: buffers whose last reader just ran.
    free_after: Vec<Vec<usize>>,
    /// External slots no step or output reads; dropped before the first step.
    free_first: Vec<usize>,
    /// Constant slots a step or output reads as a buffer; each run uploads them.
    const_buffers: Vec<(usize, Vec<u8>)>,
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn decode_attr(term: Term) -> Result<Attr, String> {
    if let Ok(i) = term.decode::<i64>() {
        return Ok(Attr::Int(i));
    }
    if let Ok(b) = term.decode::<Binary>() {
        return Ok(Attr::Bin(b.as_slice().to_vec()));
    }
    term.atom_to_string()
        .map(Attr::Atom)
        .map_err(|_| "attr is not an integer, atom or binary".to_string())
}

fn attr_int(attrs: &[Attr], i: usize, what: &str) -> Result<i64, String> {
    match attrs.get(i) {
        Some(Attr::Int(v)) => Ok(*v),
        _ => Err(format!("attr {} ({}) must be an integer", i, what)),
    }
}

fn attr_size(attrs: &[Attr], i: usize, what: &str) -> Result<usize, String> {
    usize::try_from(attr_int(attrs, i, what)?)
        .map_err(|_| format!("attr {} ({}) is negative", i, what))
}

fn attr_str(attrs: &[Attr], i: usize, what: &str) -> Result<String, String> {
    match attrs.get(i) {
        Some(Attr::Bin(b)) => {
            String::from_utf8(b.clone()).map_err(|_| format!("attr {} ({}) is not UTF-8", i, what))
        }
        _ => Err(format!("attr {} ({}) must be a binary", i, what)),
    }
}

fn to_i32(v: i64, what: &str) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| format!("{} {} does not fit in 32 bits", what, v))
}

/// What the decoder knows about a slot: its type and shape, and its value if
/// it is a scalar constant. An input the function never reads has no type.
struct SlotInfo {
    dtype: Option<DType>,
    shape: Vec<usize>,
    constant: Option<Vec<u8>>,
}

impl SlotInfo {
    fn dtype(&self) -> Result<DType, String> {
        self.dtype
            .ok_or_else(|| "reads an input the program did not declare".to_string())
    }
}

fn decode_type(term: Term) -> Result<DType, String> {
    let (kind, bits): (Term, u32) = term
        .decode()
        .map_err(|_| "type must be {kind, bits}".to_string())?;
    let kind = kind
        .atom_to_string()
        .map_err(|_| "type kind must be an atom".to_string())?;
    DType::from_wire(&kind, bits)
}

fn decode_shape(term: Term) -> Result<Vec<usize>, String> {
    get_tuple(term)
        .map_err(|_| "shape must be a tuple".to_string())?
        .into_iter()
        .map(|d| {
            d.decode::<usize>()
                .map_err(|_| "shape holds a non-integer".to_string())
        })
        .collect()
}

fn decode_const_value(term: Term) -> Result<ConstVal, String> {
    // A complex constant is a `{re, im}` pair.
    if let Ok((re, im)) = term.decode::<(Term, Term)>() {
        return match (decode_const_value(re)?, decode_const_value(im)?) {
            (ConstVal::Complex(..), _) | (_, ConstVal::Complex(..)) => {
                Err("a complex constant holds two real parts".to_string())
            }
            (re, im) => Ok(ConstVal::Complex(part_value(re), part_value(im))),
        };
    }
    if let Ok(i) = term.decode::<i64>() {
        return Ok(ConstVal::Int(i as i128));
    }
    if let Ok(u) = term.decode::<u64>() {
        return Ok(ConstVal::Int(u as i128));
    }
    if let Ok(f) = term.decode::<f64>() {
        return Ok(ConstVal::Float(f));
    }
    match term.atom_to_string().as_deref() {
        Ok("nan") => Ok(ConstVal::Float(f64::NAN)),
        Ok("infinity") => Ok(ConstVal::Float(f64::INFINITY)),
        Ok("neg_infinity") => Ok(ConstVal::Float(f64::NEG_INFINITY)),
        _ => Err(
            "constant must be a number, nan, infinity, neg_infinity or a {re, im} pair".to_string(),
        ),
    }
}

fn part_value(v: ConstVal) -> f64 {
    match v {
        ConstVal::Int(i) => i as f64,
        ConstVal::Float(f) => f,
        ConstVal::Complex(re, _) => re,
    }
}

fn byte_size(dtype: DType, shape: &[usize]) -> Result<usize, String> {
    Ok(type_info(dtype)?.bytes * codegen::element_count(shape))
}

struct Layout {
    n_inputs: usize,
    n_captures: usize,
    n_consts: usize,
    n_external: usize,
    /// Info per slot: the externals, then one entry per decoded instruction.
    infos: Vec<SlotInfo>,
}

impl Layout {
    fn slot(&self, term: Term, defined: usize) -> Result<usize, String> {
        let (kind, index): (Term, usize) = term
            .decode()
            .map_err(|_| "ref must be a {kind, index} tuple".to_string())?;
        let kind = kind
            .atom_to_string()
            .map_err(|_| "ref kind must be an atom".to_string())?;
        let (base, limit) = match kind.as_str() {
            "input" => (0, self.n_inputs),
            "capture" => (self.n_inputs, self.n_captures),
            "const" => (self.n_inputs + self.n_captures, self.n_consts),
            "result" => (self.n_external, defined),
            other => return Err(format!("unknown ref kind {}", other)),
        };
        if index >= limit {
            return Err(format!("ref {{{}, {}}} is out of range", kind, index));
        }
        Ok(base + index)
    }
}

impl Program {
    pub fn decode(ctx: &DeviceCtx, wire: Term) -> Result<Self, String> {
        type Parts<'a> = (
            Vec<Term<'a>>,
            Vec<Term<'a>>,
            Vec<Term<'a>>,
            Vec<Term<'a>>,
            Vec<Term<'a>>,
        );
        let (inputs, captures, consts, instrs, outs): Parts = wire.decode().map_err(|_| {
            "program must be {inputs, captures, consts, instrs, outputs}".to_string()
        })?;

        let mut infos = Vec::new();
        for (i, t) in inputs.iter().enumerate() {
            let info = match t.decode::<Option<(Term, Term)>>() {
                Ok(None) => SlotInfo {
                    dtype: None,
                    shape: Vec::new(),
                    constant: None,
                },
                Ok(Some((ty, shape))) => SlotInfo {
                    dtype: Some(decode_type(ty).map_err(|e| format!("input {}: {}", i, e))?),
                    shape: decode_shape(shape).map_err(|e| format!("input {}: {}", i, e))?,
                    constant: None,
                },
                Err(_) => return Err(format!("input {} must be {{type, shape}} or nil", i)),
            };
            infos.push(info);
        }
        for (i, t) in captures.iter().enumerate() {
            let (ty, shape): (Term, Term) = t
                .decode()
                .map_err(|_| format!("capture {} must be {{type, shape}}", i))?;
            infos.push(SlotInfo {
                dtype: Some(decode_type(ty).map_err(|e| format!("capture {}: {}", i, e))?),
                shape: decode_shape(shape).map_err(|e| format!("capture {}: {}", i, e))?,
                constant: None,
            });
        }
        for (i, t) in consts.iter().enumerate() {
            let (ty, value): (Term, Term) = t
                .decode()
                .map_err(|_| format!("constant {} must be {{type, value}}", i))?;
            let dtype = decode_type(ty).map_err(|e| format!("constant {}: {}", i, e))?;
            let value = decode_const_value(value).map_err(|e| format!("constant {}: {}", i, e))?;
            infos.push(SlotInfo {
                dtype: Some(dtype),
                shape: Vec::new(),
                constant: Some(
                    encode_const(dtype, value).map_err(|e| format!("constant {}: {}", i, e))?,
                ),
            });
        }

        let (n_inputs, n_captures, n_consts) = (inputs.len(), captures.len(), consts.len());
        let mut layout = Layout {
            n_inputs,
            n_captures,
            n_consts,
            n_external: n_inputs + n_captures + n_consts,
            infos,
        };

        let mut steps = Vec::with_capacity(instrs.len());
        for (i, instr) in instrs.into_iter().enumerate() {
            let step = decode_step(ctx, &mut layout, i, instr)
                .map_err(|e| format!("instruction {}: {}", i, e))?;
            steps.push(step);
        }

        let defined = steps.len();
        let outputs = outs
            .into_iter()
            .map(|t| layout.slot(t, defined))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("output: {}", e))?;

        let n_external = layout.n_external;
        let (free_after, free_first) = liveness(n_external, &steps, &outputs);

        // A constant that a step reads as a buffer, or that is an output,
        // needs a device buffer. Elementwise kernels take it by value instead.
        let const_start = n_inputs + n_captures;
        let mut read = vec![false; n_external];
        for step in &steps {
            match step {
                Step::Alias { src, .. } => {
                    if *src < n_external {
                        read[*src] = true;
                    }
                }
                Step::Launch { args, .. } => {
                    for a in args {
                        if let Arg::Slot(s) = a {
                            if *s < n_external {
                                read[*s] = true;
                            }
                        }
                    }
                }
            }
        }
        for &o in &outputs {
            if o < n_external {
                read[o] = true;
            }
        }
        let const_buffers = (const_start..n_external)
            .filter(|&s| read[s])
            .filter_map(|s| layout.infos[s].constant.clone().map(|bytes| (s, bytes)))
            .collect();

        Ok(Self {
            n_external,
            steps,
            outputs,
            free_after,
            free_first,
            const_buffers,
        })
    }

    /// Run the program. `externals` holds inputs, captures and constants in
    /// slot order; a slot nothing reads may be `None`.
    pub fn eval(
        &self,
        ctx: &DeviceCtx,
        externals: Vec<Option<Arc<DeviceBuffer>>>,
    ) -> Result<Vec<Arc<DeviceBuffer>>, String> {
        if externals.len() != self.n_external {
            return Err(format!(
                "expected {} input buffers, got {}",
                self.n_external,
                externals.len()
            ));
        }

        let mut slots: Vec<Option<Arc<DeviceBuffer>>> = externals;
        slots.resize(self.n_external + self.steps.len(), None);
        for (s, bytes) in &self.const_buffers {
            slots[*s] = Some(Arc::new(DeviceBuffer::create_with_data(ctx, bytes)?));
        }
        for &s in &self.free_first {
            slots[s] = None;
        }

        let queue = ctx
            .queue
            .lock()
            .map_err(|e| format!("Lock error: {:?}", e))?;

        // No wait between steps. The queue is in order, and OpenCL keeps a
        // buffer alive until every queued command that uses it has finished,
        // so dropping a slot at its last reader is safe.
        let run = (|| -> Result<(), String> {
            for (i, step) in self.steps.iter().enumerate() {
                match step {
                    Step::Alias { src, out } => {
                        slots[*out] = slots[*src].clone();
                    }
                    Step::Launch {
                        kernel,
                        args,
                        out,
                        out_bytes,
                        global,
                        local,
                    } => {
                        let out_buf = Arc::new(DeviceBuffer::create(ctx, *out_bytes)?);
                        // An empty output needs no launch.
                        if *out_bytes == 0 {
                            slots[*out] = Some(out_buf);
                            for &s in &self.free_after[i] {
                                slots[s] = None;
                            }
                            continue;
                        }
                        let mut bufs = Vec::with_capacity(args.len());
                        for a in args {
                            if let Arg::Slot(o) = a {
                                bufs.push(slots[*o].clone().ok_or_else(|| {
                                    "an input buffer the program reads was not given".to_string()
                                })?);
                            }
                        }
                        let mut next = bufs.iter();
                        let mut exec = ExecuteKernel::new(&kernel.kernel);
                        unsafe {
                            for a in args {
                                match a {
                                    Arg::Slot(_) => {
                                        let b = next.next().expect("one buffer per slot arg");
                                        exec.set_arg(&b.buffer.get());
                                    }
                                    Arg::Out => {
                                        exec.set_arg(&out_buf.buffer.get());
                                    }
                                    Arg::Scalar(bytes) => set_scalar(&mut exec, bytes)?,
                                }
                            }
                            exec.set_global_work_sizes(global);
                            if !local.is_empty() {
                                exec.set_local_work_sizes(local);
                            }
                            // The event is dropped; `finish` waits for it.
                            exec.enqueue_nd_range(&queue)
                                .map_err(|e| format!("Failed to enqueue kernel: {:?}", e))?;
                        }
                        ctx.stats.launches.fetch_add(1, Ordering::Relaxed);
                        slots[*out] = Some(out_buf);
                    }
                }
                for &s in &self.free_after[i] {
                    slots[s] = None;
                }
            }
            Ok(())
        })();

        // Drain the queue even after an error, before any buffer goes away.
        let finished = queue.finish().map_err(|e| format!("Device error: {:?}", e));
        run?;
        finished?;

        self.outputs
            .iter()
            .map(|&s| {
                slots[s]
                    .clone()
                    .ok_or_else(|| "an output buffer was not given".to_string())
            })
            .collect()
    }
}

fn liveness(n_external: usize, steps: &[Step], outputs: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let total = n_external + steps.len();
    let mut last_read: Vec<Option<usize>> = vec![None; total];
    for (i, step) in steps.iter().enumerate() {
        match step {
            Step::Alias { src, .. } => last_read[*src] = Some(i),
            Step::Launch { args, .. } => {
                for a in args {
                    if let Arg::Slot(o) = a {
                        last_read[*o] = Some(i);
                    }
                }
            }
        }
    }
    let is_output = |s: usize| outputs.contains(&s);

    let mut free_after = vec![Vec::new(); steps.len()];
    let mut free_first = Vec::new();
    for s in 0..total {
        if is_output(s) {
            continue;
        }
        match (last_read[s], s >= n_external) {
            (Some(i), _) => free_after[i].push(s),
            (None, true) => free_after[s - n_external].push(s),
            (None, false) => free_first.push(s),
        }
    }
    (free_after, free_first)
}

/// Set a scalar argument from the bytes of one element.
///
/// # Safety
/// The argument index and the kernel must match the value size.
unsafe fn set_scalar(exec: &mut ExecuteKernel, bytes: &[u8]) -> Result<(), String> {
    match bytes.len() {
        1 => exec.set_arg(&bytes[0]),
        2 => exec.set_arg(&u16::from_ne_bytes([bytes[0], bytes[1]])),
        4 => exec.set_arg(&u32::from_ne_bytes(bytes.try_into().unwrap())),
        8 => exec.set_arg(&u64::from_ne_bytes(bytes.try_into().unwrap())),
        16 => exec.set_arg(&u128::from_ne_bytes(bytes.try_into().unwrap())),
        n => return Err(format!("scalar of {} bytes is not supported", n)),
    };
    Ok(())
}

fn decode_step(
    ctx: &DeviceCtx,
    layout: &mut Layout,
    index: usize,
    instr: Term,
) -> Result<Step, String> {
    let (op, operands, out_type, out_shape, attrs): (Term, Vec<Term>, Term, Term, Vec<Term>) =
        instr.decode().map_err(|_| {
            "instruction must be {op, operands, out_type, out_shape, attrs}".to_string()
        })?;
    let op = op
        .atom_to_string()
        .map_err(|_| "op must be an atom".to_string())?;
    let operands = operands
        .into_iter()
        .map(|t| layout.slot(t, index))
        .collect::<Result<Vec<_>, _>>()?;
    let attrs = attrs
        .into_iter()
        .map(decode_attr)
        .collect::<Result<Vec<_>, _>>()?;
    let out_dtype = decode_type(out_type)?;
    let out_shape = decode_shape(out_shape)?;
    let out = layout.n_external + index;
    // Every instruction defines exactly one slot.
    layout.infos.push(SlotInfo {
        dtype: Some(out_dtype),
        shape: out_shape.clone(),
        constant: None,
    });
    let out_bytes = byte_size(out_dtype, &out_shape)?;

    let expect = |n: usize| -> Result<(), String> {
        if operands.len() == n {
            Ok(())
        } else {
            Err(format!(
                "{} takes {} operands, got {}",
                op,
                n,
                operands.len()
            ))
        }
    };

    let alias_of = |src: usize, layout: &Layout| -> Result<Step, String> {
        let info = &layout.infos[src];
        let in_bytes = byte_size(info.dtype()?, &info.shape)?;
        if in_bytes != out_bytes {
            return Err(format!(
                "{} changes the byte size from {} to {}",
                op, in_bytes, out_bytes
            ));
        }
        Ok(Step::Alias { src, out })
    };

    match op.as_str() {
        name if codegen::is_movement(name) => launch_generated(
            ctx, layout, name, &operands, out_dtype, out_shape, attrs, out,
        ),
        name if codegen::is_elementwise(name) => {
            // `as_type` to the same type moves no data.
            if name == "as_type" && expect(1).is_ok() {
                let info = &layout.infos[operands[0]];
                if info.dtype == Some(out_dtype) && info.shape == out_shape {
                    return alias_of(operands[0], layout);
                }
            }
            launch_generated(
                ctx, layout, name, &operands, out_dtype, out_shape, attrs, out,
            )
        }
        "bitcast" => {
            expect(1)?;
            alias_of(operands[0], layout)
        }
        "dot" => {
            expect(2)?;
            let f32t = DType::new(Kind::F, 32);
            let all_f32 = operands
                .iter()
                .all(|&o| layout.infos[o].dtype == Some(f32t))
                && out_dtype == f32t;
            if !all_f32 {
                return Err("dot runs on f32 only until stage 08".to_string());
            }
            let m = attr_size(&attrs, 0, "m")?;
            let n = attr_size(&attrs, 1, "n")?;
            let k = attr_size(&attrs, 2, "k")?;
            let opts = format!("-DTS={}", MATMUL_TILE);
            let kernel = kernel::cached(ctx, MATMUL_SOURCE, "matmul", &opts)?;
            let mut args: Vec<Arg> = operands.iter().map(|&o| Arg::Slot(o)).collect();
            args.push(Arg::Out);
            for v in [m, n, k] {
                args.push(Arg::Scalar(
                    to_i32(v as i64, "dot size")?.to_ne_bytes().to_vec(),
                ));
            }
            Ok(Step::Launch {
                kernel,
                args,
                out,
                out_bytes,
                global: vec![round_up(m, MATMUL_TILE), round_up(n, MATMUL_TILE)],
                local: vec![MATMUL_TILE, MATMUL_TILE],
            })
        }
        "reshape" | "squeeze" => {
            expect(1)?;
            alias_of(operands[0], layout)
        }
        "launch" => decode_launch(ctx, operands, out, out_bytes, &attrs),
        other => Err(format!("unknown op {}", other)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_generated(
    ctx: &DeviceCtx,
    layout: &Layout,
    op: &str,
    operands: &[usize],
    out_dtype: DType,
    out_shape: Vec<usize>,
    attrs: Vec<Attr>,
    out: usize,
) -> Result<Step, String> {
    let mut ops = Vec::with_capacity(operands.len());
    for &o in operands {
        let info = &layout.infos[o];
        ops.push(Operand {
            dtype: info.dtype()?,
            shape: info.shape.clone(),
            constant: info.constant.clone(),
        });
    }
    let out_bytes = byte_size(out_dtype, &out_shape)?;
    let instr = Instr {
        op: op.to_string(),
        operands: ops,
        out: out_dtype,
        out_shape,
        attrs,
    };
    let spec = codegen::generate(&instr, &ctx.caps)?;
    let kernel = kernel::cached(ctx, &spec.source, &spec.entry, &spec.build_options)?;
    let args = spec
        .args
        .into_iter()
        .map(|a| match a {
            KernelArg::Operand(k) => Arg::Slot(operands[k]),
            KernelArg::Scalar(bytes) => Arg::Scalar(bytes),
            KernelArg::Output => Arg::Out,
        })
        .collect();
    Ok(Step::Launch {
        kernel,
        args,
        out,
        out_bytes,
        global: spec.global,
        local: spec.local,
    })
}

/// Attrs: source, entry, build options, dims, global sizes, local sizes, scalars.
/// Local sizes all zero means the driver picks. The kernel takes the operand
/// buffers, then the output buffer, then the scalars as `int`.
fn decode_launch(
    ctx: &DeviceCtx,
    operands: Vec<usize>,
    out: usize,
    out_bytes: usize,
    attrs: &[Attr],
) -> Result<Step, String> {
    let source = attr_str(attrs, 0, "source")?;
    let entry = attr_str(attrs, 1, "entry")?;
    let opts = attr_str(attrs, 2, "build options")?;
    let dims = attr_size(attrs, 3, "dims")?;
    if dims == 0 || dims > MAX_DIMS {
        return Err(format!("dims must be 1 to {}, got {}", MAX_DIMS, dims));
    }
    let mut global = Vec::with_capacity(dims);
    let mut local = Vec::with_capacity(dims);
    for d in 0..dims {
        let g = attr_size(attrs, 4 + d, "global size")?;
        if g == 0 {
            return Err("global size must be positive".to_string());
        }
        global.push(g);
        local.push(attr_size(attrs, 4 + dims + d, "local size")?);
    }
    if local.iter().all(|&l| l == 0) {
        local.clear();
    } else if local.contains(&0) {
        return Err("local sizes must be all zero or all positive".to_string());
    }
    let mut args: Vec<Arg> = operands.into_iter().map(Arg::Slot).collect();
    args.push(Arg::Out);
    for i in (4 + 2 * dims)..attrs.len() {
        let v = to_i32(attr_int(attrs, i, "scalar")?, "scalar")?;
        args.push(Arg::Scalar(v.to_ne_bytes().to_vec()));
    }
    let kernel = kernel::cached(ctx, &source, &entry, &opts)?;
    Ok(Step::Launch {
        kernel,
        args,
        out,
        out_bytes,
        global,
        local,
    })
}
