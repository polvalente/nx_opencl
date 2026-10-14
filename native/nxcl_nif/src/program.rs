//! Straight-line program executor.
//!
//! `Program::decode` turns the wire term built by `NxCL.Executor` into steps and
//! builds every kernel through the device cache. `Program::eval` enqueues the
//! steps in order on the one in-order queue and waits once, in `finish`.
//!
//! Wire term: `{n_inputs, n_captures, n_consts, [instr], [out_ref]}` where
//! `instr = {op, [ref], out_bytes, [attr]}` and `ref = {kind, index}`.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::ClMem;
use rustler::{Binary, Term};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::buffer::DeviceBuffer;
use crate::device_ctx::DeviceCtx;
use crate::kernel::{self, CompiledKernel, ELEMENTWISE_SOURCE, MATMUL_SOURCE};

const UNARY_OPS: [&str; 8] = [
    "negate", "exp", "log", "tanh", "sigmoid", "abs", "relu", "rsqrt",
];
const BINARY_OPS: [&str; 4] = ["add", "subtract", "multiply", "divide"];
const MATMUL_TILE: usize = 8;
const ELEMENTWISE_LOCAL: usize = 64;
const MAX_DIMS: usize = 3;

enum Attr {
    Int(i64),
    Atom,
    Bin(Vec<u8>),
}

enum Step {
    /// The result shares the operand's buffer.
    Alias { src: usize, out: usize },
    Launch {
        kernel: Arc<CompiledKernel>,
        operands: Vec<usize>,
        out: usize,
        out_bytes: usize,
        global: Vec<usize>,
        local: Vec<usize>,
        scalars: Vec<i32>,
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
        .map(|_| Attr::Atom)
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

struct Layout {
    n_inputs: usize,
    n_captures: usize,
    n_consts: usize,
    n_external: usize,
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
        let (n_inputs, n_captures, n_consts, instrs, outs): (
            usize,
            usize,
            usize,
            Vec<Term>,
            Vec<Term>,
        ) = wire.decode().map_err(|_| {
            "program must be {n_inputs, n_captures, n_consts, instrs, outputs}".to_string()
        })?;
        let layout = Layout {
            n_inputs,
            n_captures,
            n_consts,
            n_external: n_inputs + n_captures + n_consts,
        };

        let mut steps = Vec::with_capacity(instrs.len());
        for (i, instr) in instrs.into_iter().enumerate() {
            let step = decode_step(ctx, &layout, i, instr)
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
        Ok(Self {
            n_external,
            steps,
            outputs,
            free_after,
            free_first,
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
                        operands,
                        out,
                        out_bytes,
                        global,
                        local,
                        scalars,
                    } => {
                        let out_buf = Arc::new(DeviceBuffer::create(ctx, *out_bytes)?);
                        let mut args = Vec::with_capacity(operands.len());
                        for &o in operands {
                            args.push(slots[o].clone().ok_or_else(|| {
                                "an input buffer the program reads was not given".to_string()
                            })?);
                        }
                        let mut exec = ExecuteKernel::new(&kernel.kernel);
                        unsafe {
                            for a in &args {
                                exec.set_arg(&a.buffer.get());
                            }
                            exec.set_arg(&out_buf.buffer.get());
                            for s in scalars {
                                exec.set_arg(s);
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
            Step::Launch { operands, .. } => {
                for &o in operands {
                    last_read[o] = Some(i);
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

fn decode_step(
    ctx: &DeviceCtx,
    layout: &Layout,
    index: usize,
    instr: Term,
) -> Result<Step, String> {
    let (op, operands, out_bytes, attrs): (Term, Vec<Term>, usize, Vec<Term>) = instr
        .decode()
        .map_err(|_| "instruction must be {op, operands, out_bytes, attrs}".to_string())?;
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
    let out = layout.n_external + index;

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

    match op.as_str() {
        name if UNARY_OPS.contains(&name) || BINARY_OPS.contains(&name) => {
            expect(if UNARY_OPS.contains(&name) { 1 } else { 2 })?;
            let entry = format!("elementwise_{}", name);
            let kernel = kernel::cached(ctx, ELEMENTWISE_SOURCE, &entry, "")?;
            let n = to_i32((out_bytes / 4) as i64, "element count")?;
            Ok(Step::Launch {
                kernel,
                operands,
                out,
                out_bytes,
                global: vec![round_up(n as usize, ELEMENTWISE_LOCAL)],
                local: vec![ELEMENTWISE_LOCAL],
                scalars: vec![n],
            })
        }
        "dot" => {
            expect(2)?;
            let m = attr_size(&attrs, 0, "m")?;
            let n = attr_size(&attrs, 1, "n")?;
            let k = attr_size(&attrs, 2, "k")?;
            let opts = format!("-DTS={}", MATMUL_TILE);
            let kernel = kernel::cached(ctx, MATMUL_SOURCE, "matmul", &opts)?;
            Ok(Step::Launch {
                kernel,
                operands,
                out,
                out_bytes,
                global: vec![round_up(m, MATMUL_TILE), round_up(n, MATMUL_TILE)],
                local: vec![MATMUL_TILE, MATMUL_TILE],
                scalars: vec![
                    to_i32(m as i64, "m")?,
                    to_i32(n as i64, "n")?,
                    to_i32(k as i64, "k")?,
                ],
            })
        }
        "reshape" | "squeeze" => {
            expect(1)?;
            Ok(Step::Alias {
                src: operands[0],
                out,
            })
        }
        "launch" => decode_launch(ctx, operands, out, out_bytes, &attrs),
        other => Err(format!("unknown op {}", other)),
    }
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
    let mut scalars = Vec::new();
    for i in (4 + 2 * dims)..attrs.len() {
        scalars.push(to_i32(attr_int(attrs, i, "scalar")?, "scalar")?);
    }
    let kernel = kernel::cached(ctx, &source, &entry, &opts)?;
    Ok(Step::Launch {
        kernel,
        operands,
        out,
        out_bytes,
        global,
        local,
        scalars,
    })
}
