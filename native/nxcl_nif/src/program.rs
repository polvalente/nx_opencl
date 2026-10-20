//! Straight-line program executor.
//!
//! `Program::decode` turns the wire term from `NxCL.Compiler.Executor.Wire`
//! into steps and builds every kernel through the device cache. `Program::eval`
//! enqueues the steps in order. On the context's own queue it returns after a
//! flush and a read of an output waits; on another queue, or when a kernel
//! reports an error code, it waits once, in `finish`.
//!
//! Wire term: `{inputs, captures, consts, [instr], [out_ref]}` with
//! `inputs = [{type, shape} | nil]`, `captures = [{type, shape}]`,
//! `consts = [{type, value}]`, `instr = {op, [ref], out_type, out_shape, [attr]}`,
//! `type = {kind, bits}`, `ref = {kind, index}`. The decoder keeps each slot's
//! type and shape, so an instruction carries no operand types.
//!
//! A shape whose element count or byte size does not fit in a `long` is an
//! error, so codegen can pass dimensions as kernel arguments without a wrap.

use opencl3::command_queue::CommandQueue;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::ClMem;
use rustler::types::tuple::get_tuple;
use rustler::{Binary, Term};
use std::sync::{Arc, Mutex};

use crate::buffer::DeviceBuffer;
use crate::codegen::types::{encode_const, type_info, ConstVal, DType};
use crate::codegen::{self, Attr, Instr, KernelArg, Operand};
use crate::device_ctx::{cl_error, lock, DeviceCtx};
use crate::kernel::{self, CompiledKernel};

/// One kernel argument.
enum Arg {
    Slot(usize),
    /// The output buffer of the step.
    Out,
    /// A scratch buffer the passes of the step share.
    Scratch(usize),
    /// A scalar passed by value: the bytes of one element, 1, 2, 4, 8 or 16 long.
    Scalar(Vec<u8>),
    /// A table the host built, uploaded when the program is decoded.
    Table(Arc<DeviceBuffer>),
    /// The one-`int` error flag of the run; a kernel sets it on a bad index.
    ErrorFlag,
}

/// One kernel launch.
struct Pass {
    kernel: Arc<CompiledKernel>,
    args: Vec<Arg>,
    global: Vec<usize>,
    local: Vec<usize>,
}

/// One or more launches in order. `scratch` holds the byte size of each
/// scratch buffer the passes read and write; they live for the step.
struct Launch {
    passes: Vec<Pass>,
    scratch: Vec<usize>,
    out: usize,
    out_bytes: usize,
}

enum Step {
    /// The result shares the operand's buffer.
    Alias {
        src: usize,
        out: usize,
    },
    Launch(Launch),
}

impl Step {
    /// The slots the step reads as buffers.
    fn reads(&self) -> Vec<usize> {
        match self {
            Step::Alias { src, .. } => vec![*src],
            Step::Launch(l) => l
                .passes
                .iter()
                .flat_map(|p| p.args.iter())
                .filter_map(|a| match a {
                    Arg::Slot(s) => Some(*s),
                    _ => None,
                })
                .collect(),
        }
    }

    fn uses_flag(&self) -> bool {
        match self {
            Step::Alias { .. } => false,
            Step::Launch(l) => l
                .passes
                .iter()
                .any(|p| p.args.iter().any(|a| matches!(a, Arg::ErrorFlag))),
        }
    }
}

pub struct Program {
    n_external: usize,
    steps: Vec<Step>,
    outputs: Vec<usize>,
    /// Slots to drop after each step: buffers whose last reader just ran.
    free_after: Vec<Vec<usize>>,
    /// External slots no step or output reads; dropped before the first step.
    free_first: Vec<usize>,
    /// Constant slots a step or output reads as a buffer, uploaded once at
    /// decode. No kernel writes to an input, so every run shares them.
    const_buffers: Vec<(usize, Arc<DeviceBuffer>)>,
    /// True when a kernel reads the error flag: the run allocates it and checks it.
    uses_flag: bool,
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

/// What the decoder knows about a slot: its type and shape, and its value if
/// it is a scalar constant. An input the function never reads has no type.
struct SlotInfo {
    dtype: Option<DType>,
    shape: Vec<usize>,
    constant: Option<Vec<u8>>,
}

impl SlotInfo {
    /// A slot with a type and a shape and no constant value.
    fn buffer(dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            dtype: Some(dtype),
            shape,
            constant: None,
        }
    }

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

/// A shape tuple. Its element count must fit in a `long`.
fn decode_shape(term: Term) -> Result<Vec<usize>, String> {
    let shape = get_tuple(term)
        .map_err(|_| "shape must be a tuple".to_string())?
        .into_iter()
        .map(|d| {
            d.decode::<usize>()
                .map_err(|_| "shape holds a non-integer".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    codegen::checked_element_count(&shape)?;
    Ok(shape)
}

/// A real part of a complex constant, as `f64`. An integer beyond 2^53 loses
/// its low bits, as it would in the float type it becomes.
fn part_value(v: ConstVal) -> Result<f64, String> {
    match v {
        ConstVal::Int(i) => Ok(i as f64),
        ConstVal::Float(f) => Ok(f),
        ConstVal::Complex(..) => Err("a complex constant holds two real parts".to_string()),
    }
}

fn decode_const_value(term: Term) -> Result<ConstVal, String> {
    // A complex constant is a `{re, im}` pair.
    if let Ok((re, im)) = term.decode::<(Term, Term)>() {
        let (re, im) = (
            part_value(decode_const_value(re)?)?,
            part_value(decode_const_value(im)?)?,
        );
        return Ok(ConstVal::Complex(re, im));
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

/// Bytes of a buffer of `dtype` and `shape`, or an error when the size does not
/// fit in a `long`.
fn byte_size(dtype: DType, shape: &[usize]) -> Result<usize, String> {
    let elements = codegen::checked_element_count(shape)?;
    let bytes = type_info(dtype)?.bytes;
    codegen::checked_product("the buffer size", &[bytes, elements])
        .map_err(|_| format!("a {} buffer of shape {:?} is too large", dtype, shape))
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

type WireParts<'a> = (
    Vec<Term<'a>>,
    Vec<Term<'a>>,
    Vec<Term<'a>>,
    Vec<Term<'a>>,
    Vec<Term<'a>>,
);

/// The slots the wire declares before any instruction: inputs, captures and
/// constants.
fn decode_externals(inputs: &[Term], captures: &[Term], consts: &[Term]) -> Result<Layout, String> {
    let mut infos = Vec::with_capacity(inputs.len() + captures.len() + consts.len());
    for (i, t) in inputs.iter().enumerate() {
        let info = match t.decode::<Option<(Term, Term)>>() {
            Ok(None) => SlotInfo {
                dtype: None,
                shape: Vec::new(),
                constant: None,
            },
            Ok(Some((ty, shape))) => SlotInfo::buffer(
                decode_type(ty).map_err(|e| format!("input {}: {}", i, e))?,
                decode_shape(shape).map_err(|e| format!("input {}: {}", i, e))?,
            ),
            Err(_) => return Err(format!("input {} must be {{type, shape}} or nil", i)),
        };
        infos.push(info);
    }
    for (i, t) in captures.iter().enumerate() {
        let (ty, shape): (Term, Term) = t
            .decode()
            .map_err(|_| format!("capture {} must be {{type, shape}}", i))?;
        infos.push(SlotInfo::buffer(
            decode_type(ty).map_err(|e| format!("capture {}: {}", i, e))?,
            decode_shape(shape).map_err(|e| format!("capture {}: {}", i, e))?,
        ));
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
    Ok(Layout {
        n_inputs,
        n_captures,
        n_consts,
        n_external: n_inputs + n_captures + n_consts,
        infos,
    })
}

/// The device buffers for the constant slots a step or output reads. An
/// elementwise kernel takes a constant by value instead, so those need none.
fn upload_constants(
    ctx: &DeviceCtx,
    layout: &Layout,
    steps: &[Step],
    outputs: &[usize],
) -> Result<Vec<(usize, Arc<DeviceBuffer>)>, String> {
    let const_start = layout.n_inputs + layout.n_captures;
    let mut read = vec![false; layout.n_external];
    let reads = steps
        .iter()
        .flat_map(|step| step.reads())
        .chain(outputs.iter().copied());
    for s in reads {
        if let Some(flag) = read.get_mut(s) {
            *flag = true;
        }
    }
    let mut buffers = Vec::new();
    for (s, info) in layout
        .infos
        .iter()
        .enumerate()
        .take(layout.n_external)
        .skip(const_start)
    {
        if let (true, Some(bytes)) = (read[s], &info.constant) {
            buffers.push((s, Arc::new(DeviceBuffer::create_with_data(ctx, bytes)?)));
        }
    }
    Ok(buffers)
}

impl Program {
    pub fn decode(ctx: &DeviceCtx, wire: Term) -> Result<Self, String> {
        let (inputs, captures, consts, instrs, outs): WireParts = wire.decode().map_err(|_| {
            "program must be {inputs, captures, consts, instrs, outputs}".to_string()
        })?;

        let mut layout = decode_externals(&inputs, &captures, &consts)?;

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
        let const_buffers = upload_constants(ctx, &layout, &steps, &outputs)?;
        let uses_flag = steps.iter().any(Step::uses_flag);

        Ok(Self {
            n_external,
            steps,
            outputs,
            free_after,
            free_first,
            const_buffers,
            uses_flag,
        })
    }

    /// Run the program. `externals` holds inputs, captures and constants in
    /// slot order; a slot nothing reads may be `None`.
    pub fn eval(
        &self,
        ctx: &DeviceCtx,
        queue: &Mutex<CommandQueue>,
        externals: Vec<Option<Arc<DeviceBuffer>>>,
    ) -> Result<Vec<Arc<DeviceBuffer>>, String> {
        if externals.len() != self.n_external {
            return Err(format!(
                "expected {} input buffers, got {}",
                self.n_external,
                externals.len()
            ));
        }

        let mut slots = self.prepare_slots(externals);
        let flag = if self.uses_flag {
            Some(DeviceBuffer::create_with_data(ctx, &[0u8; 4])?)
        } else {
            None
        };

        let on_default_queue = std::ptr::eq(queue, &ctx.queue);
        // The default queue returns without waiting. A read of an output uses the
        // default queue (`buffer_read` and `buffer_read_many` use no other), so
        // it runs after this run's commands. Any other queue ends in `finish`.
        let lazy = !self.uses_flag && on_default_queue;
        if !on_default_queue {
            // An input may be the output of an earlier run on the default
            // queue that has not finished. This queue does not order itself after
            // that queue, so wait for it first. The lock is released before the
            // run takes this queue's lock, so the two are never held together.
            lock(&ctx.queue)
                .finish()
                .map_err(|e| cl_error("Device error", e))?;
        }
        let queue = lock(queue);

        let run = self.run_steps(ctx, &queue, flag.as_ref(), &mut slots);

        // Drain the queue even after an error, before any buffer goes away.
        let finished = if lazy && run.is_ok() {
            // Flush so the device starts now; a read waits for the rest.
            queue.flush()
        } else {
            queue.finish()
        }
        .map_err(|e| cl_error("Device error", e));
        drop(queue);
        run?;
        finished?;
        if let Some(flag) = &flag {
            check_flag(ctx, flag)?;
        }

        self.outputs
            .iter()
            .map(|&s| {
                slots[s]
                    .clone()
                    .ok_or_else(|| "an output buffer was not given".to_string())
            })
            .collect()
    }

    /// The slots of a run: the externals, the shared constants, and room for
    /// each step's result. Slots nothing reads are dropped at once.
    fn prepare_slots(
        &self,
        externals: Vec<Option<Arc<DeviceBuffer>>>,
    ) -> Vec<Option<Arc<DeviceBuffer>>> {
        let mut slots = externals;
        slots.resize(self.n_external + self.steps.len(), None);
        for (s, buf) in &self.const_buffers {
            slots[*s] = Some(buf.clone());
        }
        for &s in &self.free_first {
            slots[s] = None;
        }
        slots
    }

    /// Enqueue every step in order, with no wait between. OpenCL keeps a buffer
    /// alive until every queued command that uses it finishes, so dropping a slot
    /// at its last reader is safe.
    fn run_steps(
        &self,
        ctx: &DeviceCtx,
        queue: &CommandQueue,
        flag: Option<&DeviceBuffer>,
        slots: &mut [Option<Arc<DeviceBuffer>>],
    ) -> Result<(), String> {
        for (i, step) in self.steps.iter().enumerate() {
            match step {
                Step::Alias { src, out } => slots[*out] = slots[*src].clone(),
                Step::Launch(launch) => launch.run(ctx, queue, flag, slots)?,
            }
            for &s in &self.free_after[i] {
                slots[s] = None;
            }
        }
        Ok(())
    }
}

/// Reads the error flag after a run and turns a nonzero code into an error.
fn check_flag(ctx: &DeviceCtx, flag: &DeviceBuffer) -> Result<(), String> {
    let bytes = flag.read(ctx, 4)?;
    let code = i32::from_ne_bytes(bytes.try_into().map_err(|_| "short flag read")?);
    if code != 0 {
        return Err(codegen::indexing::error_message(code));
    }
    Ok(())
}

impl Launch {
    fn run(
        &self,
        ctx: &DeviceCtx,
        queue: &CommandQueue,
        flag: Option<&DeviceBuffer>,
        slots: &mut [Option<Arc<DeviceBuffer>>],
    ) -> Result<(), String> {
        let out_buf = Arc::new(DeviceBuffer::create(ctx, self.out_bytes)?);
        // An empty output needs no launch. Codegen still builds the kernels, with
        // global size clamped to 1.
        if self.out_bytes > 0 {
            let scratch = self
                .scratch
                .iter()
                .map(|&n| DeviceBuffer::create(ctx, n))
                .collect::<Result<Vec<_>, _>>()?;
            for pass in &self.passes {
                pass.enqueue(queue, slots, &out_buf, &scratch, flag)?;
            }
        }
        slots[self.out] = Some(out_buf);
        Ok(())
    }
}

impl Pass {
    fn enqueue(
        &self,
        queue: &CommandQueue,
        slots: &[Option<Arc<DeviceBuffer>>],
        out: &DeviceBuffer,
        scratch: &[DeviceBuffer],
        flag: Option<&DeviceBuffer>,
    ) -> Result<(), String> {
        // Held from the first `set_arg` to the enqueue: the kernel object's
        // arguments are one set, and another queue may share this object.
        let _launch = lock(&self.kernel.launch);
        let mut exec = ExecuteKernel::new(&self.kernel.kernel);
        for a in &self.args {
            // SAFETY: `set_arg` reads `size_of` the value it is given. Each
            // value has the type the kernel parameter has: a `cl_mem` for a
            // buffer, and for a scalar the bytes of one `long`, `int` or
            // element the parameter declares. Generated kernels have this by
            // construction (codegen writes the parameters and `KernelArg`s
            // from one table).
            let buf: &DeviceBuffer = match a {
                Arg::Scalar(bytes) => {
                    unsafe { set_scalar(&mut exec, bytes)? };
                    continue;
                }
                Arg::Slot(s) => slots[*s]
                    .as_deref()
                    .ok_or_else(|| "an input buffer the program reads was not given".to_string())?,
                Arg::Out => out,
                Arg::Scratch(n) => scratch
                    .get(*n)
                    .ok_or_else(|| format!("scratch buffer {} was not planned", n))?,
                Arg::Table(t) => t,
                Arg::ErrorFlag => flag.ok_or("the run has no error flag")?,
            };
            unsafe { exec.set_arg(&buf.buffer.get()) };
        }
        exec.set_global_work_sizes(&self.global);
        if !self.local.is_empty() {
            exec.set_local_work_sizes(&self.local);
        }
        // SAFETY: every argument was set above, and each buffer is alive (the
        // slots, scratch and flag outlive this call, and OpenCL holds a buffer
        // until the command that uses it ends). The event is dropped: `finish`
        // or a read of an output waits for the command.
        unsafe {
            exec.enqueue_nd_range(queue)
                .map_err(|e| cl_error("Failed to enqueue kernel", e))?;
        }
        Ok(())
    }
}

fn liveness(n_external: usize, steps: &[Step], outputs: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let total = n_external + steps.len();
    let mut last_read: Vec<Option<usize>> = vec![None; total];
    for (i, step) in steps.iter().enumerate() {
        for s in step.reads() {
            last_read[s] = Some(i);
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

/// `bytes` as an array of `N` bytes.
fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], String> {
    bytes
        .try_into()
        .map_err(|_| format!("scalar of {} bytes is not {} bytes", bytes.len(), N))
}

/// Set a scalar argument from the bytes of one element.
///
/// # Safety
/// The next argument of the kernel must be a scalar parameter of
/// `bytes.len()` bytes, the size `set_arg` reads from the value it is given.
unsafe fn set_scalar(exec: &mut ExecuteKernel, bytes: &[u8]) -> Result<(), String> {
    match bytes.len() {
        1 => exec.set_arg(&bytes[0]),
        2 => exec.set_arg(&u16::from_ne_bytes(array(bytes)?)),
        4 => exec.set_arg(&u32::from_ne_bytes(array(bytes)?)),
        8 => exec.set_arg(&u64::from_ne_bytes(array(bytes)?)),
        16 => exec.set_arg(&u128::from_ne_bytes(array(bytes)?)),
        n => return Err(format!("scalar of {} bytes is not supported", n)),
    };
    Ok(())
}

/// An instruction after its operands and attrs are decoded.
struct Decoded {
    op: String,
    operands: Vec<usize>,
    out_dtype: DType,
    out_shape: Vec<usize>,
    attrs: Vec<Attr>,
    /// The slot the instruction defines.
    out: usize,
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
    layout
        .infos
        .push(SlotInfo::buffer(out_dtype, out_shape.clone()));
    let d = Decoded {
        op,
        operands,
        out_dtype,
        out_shape,
        attrs,
        out,
    };
    route(ctx, layout, d)
}

/// Picks the decoder for the op: a codegen family or an alias.
fn route(ctx: &DeviceCtx, layout: &Layout, d: Decoded) -> Result<Step, String> {
    let out_bytes = byte_size(d.out_dtype, &d.out_shape)?;
    let expect = |n: usize| -> Result<(), String> {
        if d.operands.len() == n {
            Ok(())
        } else {
            Err(format!(
                "{} takes {} operands, got {}",
                d.op,
                n,
                d.operands.len()
            ))
        }
    };
    let alias_of = |src: usize| -> Result<Step, String> {
        let info = &layout.infos[src];
        let in_bytes = byte_size(info.dtype()?, &info.shape)?;
        if in_bytes != out_bytes {
            return Err(format!(
                "{} changes the byte size from {} to {}",
                d.op, in_bytes, out_bytes
            ));
        }
        Ok(Step::Alias { src, out: d.out })
    };

    match d.op.as_str() {
        name if codegen::is_movement(name)
            || codegen::is_reduction(name)
            || codegen::is_scan(name)
            || codegen::is_window(name)
            || codegen::is_sort(name)
            || codegen::is_indexing(name)
            || codegen::is_linalg(name)
            || codegen::is_fft(name)
            || codegen::is_gemm(name) =>
        {
            launch_generated(ctx, layout, &d, out_bytes)
        }
        name if codegen::is_elementwise(name) => {
            // `as_type` to the same type moves no data.
            if name == "as_type" && expect(1).is_ok() {
                let info = &layout.infos[d.operands[0]];
                if info.dtype == Some(d.out_dtype) && info.shape == d.out_shape {
                    return alias_of(d.operands[0]);
                }
            }
            launch_generated(ctx, layout, &d, out_bytes)
        }
        "bitcast" | "reshape" | "squeeze" => {
            expect(1)?;
            alias_of(d.operands[0])
        }
        other => Err(format!("unknown op {}", other)),
    }
}

fn launch_generated(
    ctx: &DeviceCtx,
    layout: &Layout,
    d: &Decoded,
    out_bytes: usize,
) -> Result<Step, String> {
    let mut ops = Vec::with_capacity(d.operands.len());
    for &o in &d.operands {
        let info = &layout.infos[o];
        ops.push(Operand {
            dtype: info.dtype()?,
            shape: info.shape.clone(),
            constant: info.constant.clone(),
        });
    }
    let instr = Instr {
        op: d.op.clone(),
        operands: ops,
        out: d.out_dtype,
        out_shape: d.out_shape.clone(),
        attrs: d.attrs.clone(),
    };
    let plan = codegen::generate_plan(&instr, &ctx.caps, &ctx.limits)?;
    plan_step(ctx, &d.operands, plan, d.out, out_bytes)
}

/// Build the kernels of a plan and the launch step that runs them.
fn plan_step(
    ctx: &DeviceCtx,
    operands: &[usize],
    plan: codegen::Plan,
    out: usize,
    out_bytes: usize,
) -> Result<Step, String> {
    let mut passes = Vec::with_capacity(plan.passes.len());
    // A table that several passes share (one `Arc`) is uploaded once.
    let mut tables: Vec<(Arc<[u8]>, Arc<DeviceBuffer>)> = Vec::new();
    for spec in plan.passes {
        let kernel = kernel::cached(ctx, &spec.source, &spec.entry, &spec.build_options)?;
        let mut args = Vec::with_capacity(spec.args.len());
        for a in spec.args {
            args.push(match a {
                KernelArg::Operand(k) => Arg::Slot(*operands.get(k).ok_or_else(|| {
                    format!("the kernel reads operand {} of {}", k, operands.len())
                })?),
                KernelArg::Scalar(bytes) => Arg::Scalar(bytes),
                KernelArg::Output => Arg::Out,
                KernelArg::Scratch(n) => {
                    if n >= plan.scratch.len() {
                        return Err(format!(
                            "the kernel reads scratch buffer {} of {}",
                            n,
                            plan.scratch.len()
                        ));
                    }
                    Arg::Scratch(n)
                }
                KernelArg::ErrorFlag => Arg::ErrorFlag,
                KernelArg::Table(bytes) => {
                    let buf = match tables.iter().find(|(b, _)| Arc::ptr_eq(b, &bytes)) {
                        Some((_, buf)) => buf.clone(),
                        None => {
                            let buf = Arc::new(DeviceBuffer::create_with_data(ctx, &bytes)?);
                            tables.push((bytes, buf.clone()));
                            buf
                        }
                    };
                    Arg::Table(buf)
                }
            });
        }
        kernel.expect_args(args.len())?;
        passes.push(Pass {
            kernel,
            args,
            global: spec.global,
            local: spec.local,
        });
    }
    Ok(Step::Launch(Launch {
        passes,
        scratch: plan.scratch,
        out,
        out_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::types::Kind;

    #[test]
    fn byte_sizes_that_overflow_name_the_buffer() {
        let f32 = DType::new(Kind::F, 32);
        assert_eq!(byte_size(f32, &[3, 4]).unwrap(), 48);
        assert_eq!(byte_size(f32, &[0, 1 << 40]).unwrap(), 0);
        let err = byte_size(f32, &[1 << 40, 1 << 40]).unwrap_err();
        assert!(err.contains("too many elements"), "{err}");
        // The element count fits in a long, the byte count does not.
        let err = byte_size(DType::new(Kind::C, 128), &[1usize << 62]).unwrap_err();
        assert!(err.contains("c128") && err.contains("too large"), "{err}");
    }

    #[test]
    fn scalar_bytes_of_the_wrong_width_are_errors() {
        assert_eq!(array::<4>(&[1, 2, 3, 4]).unwrap(), [1, 2, 3, 4]);
        assert!(array::<8>(&[1, 2, 3, 4]).unwrap_err().contains("4 bytes"));
    }

    #[test]
    fn a_complex_part_is_a_real_number() {
        assert_eq!(part_value(ConstVal::Int(3)).unwrap(), 3.0);
        assert_eq!(part_value(ConstVal::Float(1.5)).unwrap(), 1.5);
        assert!(part_value(ConstVal::Complex(1.0, 2.0)).is_err());
    }
}
