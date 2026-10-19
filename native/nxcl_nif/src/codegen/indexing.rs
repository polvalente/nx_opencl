//! Indexing kernels: `gather`, `indexed_add` and `indexed_put`.
//!
//! Index rules (the reference is `Nx.BinaryBackend`):
//!
//! * An index outside `[0, dim)` is an error there. A kernel cannot raise, so it
//!   sets the program's error flag and keeps going with a clamped index (`gather`)
//!   or skips the update (`indexed_*`); the run then fails with
//!   `index out of bounds`. No result of such a run reaches the caller.
//! * `indexed_put` with repeated indices: the reference keeps the last update (the
//!   highest row). Here three passes give the same answer on every device: fill a
//!   winner scratch with -1, `atomic_max` the row number of each update into the
//!   winner of every target element it covers, then one thread per target element
//!   reads the update of its winner. No two threads write the same output.
//! * `indexed_add`: `s32`/`u32` use `atomic_add`; `f32` and each `c64` part use an
//!   `atomic_cmpxchg` loop on the bit pattern. `s64`/`u64`/`f64`/`c128` do the same
//!   on `ulong` and need `cl_khr_int64_base_atomics`; a device without it gets an
//!   error that names the extension, before any launch. `s8`/`s16`/`u8`/`u16`/`f16`/
//!   `bf16` have no atomics of their width: they widen into a 32-bit scratch, add
//!   there, and narrow once in a last pass. Float sums depend on the order the
//!   threads run, so repeated indices give results that differ in the last bits.
//!
//! Layout: the target's gathered `axes` act as the leading axes. The other (rest)
//! axes keep their order. `gather` writes `[indices batch..., rest...]`; an update
//! row is a window over the rest axes that starts at 0.

use super::index::{self, dims, strides};
use super::movement::{ints, type_names, Kernel};
use super::types::{type_info, Half, Kind, TypeInfo, BF16_HELPERS};
use super::{checked_product, element_count, Caps, Instr, KernelArg, KernelSpec, Plan};

const OPS: [&str; 3] = ["gather", "indexed_add", "indexed_put"];

const CODE_GATHER: i32 = 1;
const CODE_ADD: i32 = 2;
const CODE_PUT: i32 = 3;

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

/// The error text for a nonzero error flag.
pub fn error_message(code: i32) -> String {
    let op = match code {
        CODE_GATHER => "gather",
        CODE_ADD => "indexed_add",
        CODE_PUT => "indexed_put",
        _ => "an indexing op",
    };
    format!("index out of bounds in {}", op)
}

// ── Geometry ────────────────────────────────────────────

/// The target and its split into gathered and rest axes.
struct Geo {
    axes: Vec<usize>,
    rest: Vec<usize>,
    shape: Vec<usize>,
    strides: Vec<i64>,
}

impl Geo {
    fn new(instr: &Instr, shape: &[usize]) -> Result<Self, String> {
        let axes: Vec<usize> = ints(&instr.attrs, &instr.op)?
            .into_iter()
            .map(|a| usize::try_from(a).map_err(|_| format!("{}: negative axis", instr.op)))
            .collect::<Result<_, _>>()?;
        let rank = shape.len();
        if axes.is_empty() || axes.iter().any(|&a| a >= rank) {
            return Err(format!("{}: bad axes for rank {}", instr.op, rank));
        }
        if axes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!("{}: axes must be sorted and distinct", instr.op));
        }
        let rest = (0..rank).filter(|a| !axes.contains(a)).collect();
        Ok(Self {
            axes,
            rest,
            shape: shape.to_vec(),
            strides: strides(shape),
        })
    }

    fn k(&self) -> usize {
        self.axes.len()
    }

    fn r(&self) -> usize {
        self.rest.len()
    }

    /// Dimension and stride of each gathered axis, and the stride of each rest axis.
    fn gathered(&self) -> (Vec<i64>, Vec<i64>) {
        let d = self.axes.iter().map(|&a| self.shape[a] as i64).collect();
        let s = self.axes.iter().map(|&a| self.strides[a]).collect();
        (d, s)
    }

    fn rest_strides(&self) -> Vec<i64> {
        self.rest.iter().map(|&a| self.strides[a]).collect()
    }
}

fn check_index_type(instr: &Instr, k: usize) -> Result<TypeInfo, String> {
    let info = type_info(instr.operands[k].dtype)?;
    if info.is_float() || info.is_complex() {
        return Err(format!("{}: indices must be integers", instr.op));
    }
    Ok(info)
}

fn expect(instr: &Instr, n: usize) -> Result<(), String> {
    if instr.operands.len() == n {
        Ok(())
    } else {
        Err(format!(
            "{} takes {} operands, got {}",
            instr.op,
            n,
            instr.operands.len()
        ))
    }
}

/// Declares the error flag argument as `err`.
fn flag(k: &mut Kernel) {
    k.params.push("__global int* err".to_string());
    k.args.push(KernelArg::ErrorFlag);
}

/// Declares the scratch buffer `n` as `name` of C type `ty`.
fn scratch(k: &mut Kernel, ty: &str, name: &str, n: usize) {
    k.params.push(format!("volatile __global {ty}* {name}"));
    k.args.push(KernelArg::Scratch(n));
}

/// Source that reads each index of row `row` (an expression) into `v0..`, flags
/// and clamps one that is out of range. Needs the dims as `gd0..`.
fn index_reads(k: &Kernel, ix: usize, row: &str, count: usize) -> Result<String, String> {
    let info = type_info(k.instr.operands[ix].dtype)?;
    let mut s = String::new();
    for a in 0..count {
        let read = k.read(ix, &format!("{row} * {count} + {a}"));
        if !info.is_signed() && info.dtype.bits == 64 {
            s.push_str(&format!(
                "  ulong u{a} = {read};\n  long v{a} = (long)u{a};\n  if (u{a} >= (ulong)gd{a}) {{ bad = 1; v{a} = gd{a} - 1; }}\n"
            ));
        } else {
            s.push_str(&format!(
                "  long v{a} = (long)({read});\n  if (v{a} < 0 || v{a} >= gd{a}) {{ bad = 1; v{a} = v{a} < 0 ? 0 : gd{a} - 1; }}\n"
            ));
        }
    }
    Ok(s)
}

/// Adds `off += v{a} * gs{a}` for each gathered axis.
fn index_offset(count: usize) -> String {
    (0..count)
        .map(|a| format!("  off += v{a} * gs{a};\n"))
        .collect()
}

// ── gather ──────────────────────────────────────────────

/// Operands: the tensor, the indices `[batch..., k]`. Attrs: the gathered axes.
fn gather(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    expect(instr, 2)?;
    let (tensor, indices) = (&instr.operands[0], &instr.operands[1]);
    if tensor.dtype != instr.out {
        return Err(format!(
            "gather: the tensor is {}, the output {}",
            tensor.dtype, instr.out
        ));
    }
    check_index_type(instr, 1)?;
    index::check_rank(&instr.op, tensor.shape.len().max(instr.out_shape.len()))?;
    let g = Geo::new(instr, &tensor.shape)?;
    let (k, r) = (g.k(), g.r());
    let (batch, last) = match indices.shape.split_last() {
        Some((&last, batch)) => (batch, last),
        None => return Err("gather: indices need rank 1 or more".to_string()),
    };
    if last != k {
        return Err(format!(
            "gather: the last indices axis is {}, expected {}",
            last, k
        ));
    }
    let rest_dims: Vec<usize> = g.rest.iter().map(|&a| g.shape[a]).collect();
    let want: Vec<usize> = batch
        .iter()
        .copied()
        .chain(rest_dims.iter().copied())
        .collect();
    if instr.out_shape != want {
        return Err(format!("gather: the output shape should be {:?}", want));
    }

    let (gd, gs) = g.gathered();
    let mut kern = Kernel::new(instr);
    kern.operand(0)?;
    kern.operand(1)?;
    kern.output()?;
    flag(&mut kern);
    let rest_count = checked_product("the gather window size", &rest_dims)?;
    kern.long("rc".to_string(), rest_count as i64);
    kern.longs("rd", &dims(&rest_dims));
    kern.longs("rs", &g.rest_strides());
    kern.longs("gd", &gd);
    kern.longs("gs", &gs);

    let mut body = String::from(
        "  size_t i = get_global_id(0);\n  long b = (long)i / rc;\n  long q = (long)i % rc;\n  int bad = 0;\n  long off = 0;\n",
    );
    for d in (0..r).rev() {
        body.push_str(&format!("  off += (q % rd{d}) * rs{d};\n  q /= rd{d};\n"));
    }
    body.push_str(&index_reads(&kern, 1, "b", k)?);
    body.push_str(&index_offset(k));
    body.push_str(&format!("  if (bad) *err = {CODE_GATHER};\n"));
    body.push_str(&format!("  out[i] = {};\n", kern.read(0, "off")));
    let entry = format!("ix_gather_k{k}_r{r}_{}", type_names(instr));
    let mut spec = kern.finish(caps, entry, "", &body)?;
    spec.global = vec![element_count(&instr.out_shape).max(1)];
    Ok(Plan::single(spec))
}

// ── Shared by indexed_add and indexed_put ───────────────

/// Checked shapes of an indexed op: operands `[target, indices {n, k}, updates {n, window...}]`.
struct Indexed {
    geo: Geo,
    /// Elements per update row.
    row_len: usize,
    /// Elements in all the updates: `rows * row_len`.
    updates: usize,
    window: Vec<usize>,
}

fn indexed(instr: &Instr) -> Result<Indexed, String> {
    expect(instr, 3)?;
    let (target, indices, updates) = (&instr.operands[0], &instr.operands[1], &instr.operands[2]);
    for (name, o) in [("target", target), ("updates", updates)] {
        if o.dtype != instr.out {
            return Err(format!(
                "{}: the {} is {}, the output {}",
                instr.op, name, o.dtype, instr.out
            ));
        }
    }
    check_index_type(instr, 1)?;
    if instr.operands.iter().any(|o| o.constant.is_some()) {
        return Err(format!("{}: operands must be buffers", instr.op));
    }
    index::check_rank(&instr.op, target.shape.len())?;
    if instr.out_shape != target.shape {
        return Err(format!(
            "{}: the output must have the target's shape",
            instr.op
        ));
    }
    let geo = Geo::new(instr, &target.shape)?;
    let (k, r) = (geo.k(), geo.r());
    let rows = match indices.shape.as_slice() {
        [n, last] if *last == k => *n,
        _ => {
            return Err(format!(
                "{}: indices must have shape {{n, {}}}, got {:?}",
                instr.op, k, indices.shape
            ))
        }
    };
    if rows > i32::MAX as usize {
        return Err(format!("{}: too many updates", instr.op));
    }
    if updates.shape.len() != 1 + r || updates.shape[0] != rows {
        return Err(format!(
            "{}: updates must have rank {} and {} rows",
            instr.op,
            1 + r,
            rows
        ));
    }
    let window = updates.shape[1..].to_vec();
    for (d, &w) in window.iter().enumerate() {
        if w > geo.shape[geo.rest[d]] {
            return Err(format!(
                "{}: update axis {} is larger than the target's",
                instr.op,
                d + 1
            ));
        }
    }
    let row_len = checked_product("the update row size", &window)?;
    Ok(Indexed {
        row_len,
        updates: checked_product("the update count", &[rows, row_len])?,
        geo,
        window,
    })
}

/// A pass with one thread per update element. `body` runs when the index is in
/// range, with `off` the target offset and `i` the update's flat position.
fn update_pass(
    instr: &Instr,
    caps: &Caps,
    ix: &Indexed,
    spec: PassSpec<'_>,
    declare: impl FnOnce(&mut Kernel) -> Result<(), String>,
    body: &str,
    code: i32,
) -> Result<KernelSpec, String> {
    let mut k = Kernel::new(instr);
    declare(&mut k)?;
    flag(&mut k);
    let (gd, gs) = ix.geo.gathered();
    k.long("nu".to_string(), ix.updates as i64);
    k.long("uc".to_string(), ix.row_len as i64);
    k.longs("ud", &dims(&ix.window));
    k.longs("rs", &ix.geo.rest_strides());
    k.longs("gd", &gd);
    k.longs("gs", &gs);
    let (kk, r) = (ix.geo.k(), ix.geo.r());
    let mut src = String::from(
        "  size_t i = get_global_id(0);\n  if ((long)i >= nu) return;\n  long n = (long)i / uc;\n  long q = (long)i % uc;\n  int bad = 0;\n  long off = 0;\n",
    );
    for d in (0..r).rev() {
        src.push_str(&format!("  off += (q % ud{d}) * rs{d};\n  q /= ud{d};\n"));
    }
    src.push_str(&index_reads(&k, 1, "n", kk)?);
    src.push_str(&index_offset(kk));
    src.push_str(&format!("  if (bad) {{ *err = {code}; return; }}\n"));
    src.push_str(body);
    let entry = format!("{}_k{kk}_r{r}_{}", spec.name, type_names(instr));
    let mut out = k.finish(caps, entry, spec.helpers, &src)?;
    out.global = vec![ix.updates.max(1)];
    Ok(out)
}

struct PassSpec<'a> {
    name: &'a str,
    helpers: &'a str,
}

/// A pass with one thread per target element: `body` sees `i` and the
/// coordinates `c0..` of the target element.
fn target_pass(
    instr: &Instr,
    caps: &Caps,
    ix: &Indexed,
    name: &str,
    helpers: &str,
    declare: impl FnOnce(&mut Kernel) -> Result<(), String>,
    body: &str,
) -> Result<KernelSpec, String> {
    let mut k = Kernel::new(instr);
    declare(&mut k)?;
    k.long("uc".to_string(), ix.row_len as i64);
    k.longs("o", &dims(&ix.geo.shape));
    let ws = strides(&ix.window);
    k.longs("us", &ws);
    let mut src = index::coords(ix.geo.shape.len());
    src.push_str(body);
    let entry = format!("{}_r{}_{}", name, ix.geo.shape.len(), type_names(instr));
    let mut out = k.finish(caps, entry, helpers, &src)?;
    out.global = vec![element_count(&ix.geo.shape).max(1)];
    Ok(out)
}

/// `us{d}` of rest axis `d` applies to the coordinate `c{rest[d]}`.
fn window_offset(ix: &Indexed) -> String {
    ix.geo
        .rest
        .iter()
        .enumerate()
        .map(|(d, a)| format!(" + c{a} * us{d}"))
        .collect()
}

// ── indexed_put ─────────────────────────────────────────

fn indexed_put(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    let ix = indexed(instr)?;
    let tc = element_count(&ix.geo.shape);

    // Pass 1: winner[t] = -1.
    let mut k = Kernel::new(instr);
    scratch(&mut k, "int", "win", 0);
    let mut init = k.finish(
        caps,
        format!("ix_put_init_{}", instr.out.name()),
        "",
        "  win[get_global_id(0)] = -1;\n",
    )?;
    init.global = vec![tc.max(1)];

    // Pass 2: each update claims the target element it covers; the highest row wins.
    let claim = update_pass(
        instr,
        caps,
        &ix,
        PassSpec {
            name: "ix_put_claim",
            helpers: "",
        },
        |k| {
            k.operand(1)?;
            scratch(k, "int", "win", 0);
            Ok(())
        },
        "  atomic_max(&win[off], (int)n);\n",
        CODE_PUT,
    )?;

    // Pass 3: one thread per target element copies it, or its winner's update.
    let body = format!(
        "  int w = win[i];\n  long u = (long)w * uc{};\n  out[i] = w >= 0 ? {} : {};\n",
        window_offset(&ix),
        "in2[u]",
        "in0[i]"
    );
    let write = target_pass(
        instr,
        caps,
        &ix,
        "ix_put_write",
        "",
        |k| {
            k.operand(0)?;
            k.operand(2)?;
            k.output()?;
            scratch(k, "int", "win", 0);
            Ok(())
        },
        &body,
    )?;
    Ok(Plan {
        passes: vec![init, claim, write],
        scratch: vec![checked_product("the winner buffer", &[tc, 4])?],
    })
}

// ── indexed_add ─────────────────────────────────────────

const ATOMICS64: &str = "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n";

const ADD_F32: &str = "\
static void atomic_add_f32(volatile __global uint* p, float v) {
  uint seen;
  uint old = *p;
  do { seen = old; old = atomic_cmpxchg(p, seen, as_uint(as_float(seen) + v)); } while (old != seen);
}
";

const ADD_F64: &str = "\
static void atomic_add_f64(volatile __global ulong* p, double v) {
  ulong seen;
  ulong old = *p;
  do { seen = old; old = atom_cmpxchg(p, seen, as_ulong(as_double(seen) + v)); } while (old != seen);
}
";

/// How a type adds into memory.
enum Adder {
    /// Atomics on the output buffer itself.
    Direct,
    /// Widen into a 32-bit scratch, add there, narrow once.
    Widen { acc: &'static str },
}

fn adder(info: &TypeInfo) -> Adder {
    if info.dtype.bits < 32 && !info.is_complex() {
        let acc = match (info.is_float(), info.is_signed()) {
            (true, _) => "float",
            (false, true) => "int",
            (false, false) => "uint",
        };
        Adder::Widen { acc }
    } else {
        Adder::Direct
    }
}

/// The helper source and the statement that adds `v` (a C expression of the
/// buffer type) at `off` of the memory `base` (a pointer of the element type).
fn add_statement(info: &TypeInfo, base: &str, v: &str) -> (String, String) {
    let bits = info.dtype.bits;
    match (info.dtype.kind, bits) {
        (Kind::S, 32) | (Kind::U, 32) => (
            String::new(),
            format!(
                "atomic_add((volatile __global {}*)({base} + off), {v});",
                info.storage
            ),
        ),
        (Kind::S, 64) | (Kind::U, 64) => (
            ATOMICS64.to_string(),
            format!(
                "atom_add((volatile __global {}*)({base} + off), {v});",
                info.storage
            ),
        ),
        (Kind::F, 32) | (Kind::Bf, _) | (Kind::F, 16) => (
            ADD_F32.to_string(),
            format!("atomic_add_f32((volatile __global uint*)({base} + off), {v});"),
        ),
        (Kind::F, 64) => (
            format!("{ATOMICS64}{ADD_F64}"),
            format!("atomic_add_f64((volatile __global ulong*)({base} + off), {v});"),
        ),
        (Kind::C, 64) => (
            ADD_F32.to_string(),
            format!(
                "float2 cv = {v};\n  __global float* cp = (__global float*)({base}) + 2 * off;\n  atomic_add_f32((volatile __global uint*)cp, cv.x);\n  atomic_add_f32((volatile __global uint*)(cp + 1), cv.y);"
            ),
        ),
        (Kind::C, 128) => (
            format!("{ATOMICS64}{ADD_F64}"),
            format!(
                "double2 cv = {v};\n  __global double* cp = (__global double*)({base}) + 2 * off;\n  atomic_add_f64((volatile __global ulong*)cp, cv.x);\n  atomic_add_f64((volatile __global ulong*)(cp + 1), cv.y);"
            ),
        ),
        // Small integers add as 32-bit values in the scratch.
        _ => (
            String::new(),
            format!("atomic_add({base} + off, {v});"),
        ),
    }
}

/// C expression that widens the buffer element `x` into the scratch type.
fn widen(info: &TypeInfo, acc: &str, x: &str) -> String {
    match info.half {
        Some(Half::Bf16) => format!("bf16_to_f32({x})"),
        None => format!("({acc})({x})"),
    }
}

fn narrow(info: &TypeInfo, x: &str) -> String {
    match info.half {
        Some(Half::Bf16) => format!("f32_to_bf16({x})"),
        None => format!("({})({x})", info.storage),
    }
}

fn indexed_add(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    let ix = indexed(instr)?;
    let info = type_info(instr.out)?;
    let tc = element_count(&ix.geo.shape);
    // `c64` is two 32-bit parts; `c128` two 64-bit parts.
    let wide_value = matches!(
        (info.dtype.kind, info.dtype.bits),
        (Kind::S | Kind::U | Kind::F, 64) | (Kind::C, 128)
    );
    if wide_value && !caps.int64_atomics {
        return Err(format!(
            "indexed_add on {} needs cl_khr_int64_base_atomics, which the device lacks",
            instr.out
        ));
    }

    match adder(&info) {
        Adder::Direct => {
            let copy = target_pass(
                instr,
                caps,
                &ix,
                "ix_add_copy",
                "",
                |k| {
                    k.operand(0)?;
                    k.output()?;
                    Ok(())
                },
                "  out[i] = in0[i];\n",
            )?;
            let (helpers, stmt) = add_statement(&info, "out", "upd");
            let body = format!("  {} upd = in2[i];\n  {stmt}\n", info.storage);
            let add = update_pass(
                instr,
                caps,
                &ix,
                PassSpec {
                    name: "ix_add",
                    helpers: &helpers,
                },
                |k| {
                    k.operand(1)?;
                    k.operand(2)?;
                    k.output()?;
                    Ok(())
                },
                &body,
                CODE_ADD,
            )?;
            Ok(Plan {
                passes: vec![copy, add],
                scratch: Vec::new(),
            })
        }
        Adder::Widen { acc } => {
            let helpers = if info.half == Some(Half::Bf16) {
                format!("{BF16_HELPERS}{ADD_F32}")
            } else if acc == "float" {
                ADD_F32.to_string()
            } else {
                String::new()
            };
            let widen_body = format!("  acc[i] = {};\n", widen(&info, acc, "in0[i]"));
            let init = target_pass(
                instr,
                caps,
                &ix,
                "ix_add_widen",
                &helpers,
                |k| {
                    k.operand(0)?;
                    scratch(k, acc, "acc", 0);
                    Ok(())
                },
                &widen_body,
            )?;
            let value = widen(&info, acc, "in2[i]");
            let (_, stmt) = if acc == "float" {
                add_statement(&info, "acc", &value)
            } else {
                (String::new(), format!("atomic_add(acc + off, {value});"))
            };
            let add = update_pass(
                instr,
                caps,
                &ix,
                PassSpec {
                    name: "ix_add",
                    helpers: &helpers,
                },
                |k| {
                    k.operand(1)?;
                    k.operand(2)?;
                    scratch(k, acc, "acc", 0);
                    Ok(())
                },
                &format!("  {stmt}\n"),
                CODE_ADD,
            )?;
            let narrow_body = format!("  out[i] = {};\n", narrow(&info, "acc[i]"));
            let fin = target_pass(
                instr,
                caps,
                &ix,
                "ix_add_narrow",
                &helpers,
                |k| {
                    scratch(k, acc, "acc", 0);
                    k.output()?;
                    Ok(())
                },
                &narrow_body,
            )?;
            Ok(Plan {
                passes: vec![init, add, fin],
                scratch: vec![checked_product("the winner buffer", &[tc, 4])?],
            })
        }
    }
}

// ── Entry point ─────────────────────────────────────────

pub fn generate(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    match instr.op.as_str() {
        "gather" => gather(instr, caps),
        "indexed_add" => indexed_add(instr, caps),
        "indexed_put" => indexed_put(instr, caps),
        other => Err(format!("no indexing codegen for op {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{buf, ints, plan};
    use super::super::types::{DType, Kind};
    use super::*;

    const S32: DType = DType::new(Kind::S, 32);
    const S64: DType = DType::new(Kind::S, 64);
    const U64: DType = DType::new(Kind::U, 64);
    const U8: DType = DType::new(Kind::U, 8);
    const S8: DType = DType::new(Kind::S, 8);
    const U16: DType = DType::new(Kind::U, 16);
    const U32: DType = DType::new(Kind::U, 32);
    const F32: DType = DType::new(Kind::F, 32);
    const F64: DType = DType::new(Kind::F, 64);
    const BF16: DType = DType::new(Kind::Bf, 16);
    const C64: DType = DType::new(Kind::C, 64);
    const C128: DType = DType::new(Kind::C, 128);

    fn gather_instr(ix: DType, tensor: &[usize], indices: &[usize], out: &[usize]) -> Instr {
        Instr {
            op: "gather".to_string(),
            operands: vec![buf(F32, tensor), buf(ix, indices)],
            out: F32,
            out_shape: out.to_vec(),
            attrs: ints(&[0]),
        }
    }

    /// `[target {4, 3}, indices {5, 1}, updates {5, 3}]` on axis 0.
    fn indexed_instr(op: &str, dtype: DType) -> Instr {
        Instr {
            op: op.to_string(),
            operands: vec![buf(dtype, &[4, 3]), buf(S32, &[5, 1]), buf(dtype, &[5, 3])],
            out: dtype,
            out_shape: vec![4, 3],
            attrs: ints(&[0]),
        }
    }

    fn no_atomics() -> Caps {
        Caps {
            int64_atomics: false,
            ..Caps::all()
        }
    }

    #[test]
    fn gather_has_one_pass_with_the_error_flag() {
        let p = plan(&gather_instr(S32, &[4, 3], &[2, 3, 1], &[2, 3, 3]));
        assert_eq!(p.passes.len(), 1);
        assert!(p.scratch.is_empty());
        let spec = &p.passes[0];
        assert_eq!(spec.global, vec![18]);
        assert!(spec.args.contains(&KernelArg::ErrorFlag));
        assert!(spec.source.contains("if (bad) *err = 1;"));
        assert!(spec.entry.starts_with("ix_gather_k1_r1_"));
    }

    #[test]
    fn gather_source_does_not_depend_on_the_sizes() {
        let a = plan(&gather_instr(S32, &[4, 3], &[2, 1], &[2, 3]));
        let b = plan(&gather_instr(S32, &[40, 30], &[7, 9, 1], &[7, 9, 30]));
        assert_eq!(a.passes[0].source, b.passes[0].source);
    }

    #[test]
    fn a_u64_index_compares_as_unsigned() {
        let p = plan(&gather_instr(U64, &[4, 3], &[2, 1], &[2, 3]));
        assert!(p.passes[0].source.contains("u0 >= (ulong)gd0"));
        let p = plan(&gather_instr(S64, &[4, 3], &[2, 1], &[2, 3]));
        assert!(p.passes[0].source.contains("v0 < 0 || v0 >= gd0"));
    }

    #[test]
    fn gather_checks_its_shapes_and_index_type() {
        let bad_out = gather_instr(S32, &[4, 3], &[2, 1], &[2, 4]);
        assert!(generate(&bad_out, &Caps::all())
            .unwrap_err()
            .contains("output shape"));
        let bad_last = gather_instr(S32, &[4, 3], &[2, 2], &[2, 3]);
        assert!(generate(&bad_last, &Caps::all())
            .unwrap_err()
            .contains("last indices axis"));
        let float_ix = gather_instr(F32, &[4, 3], &[2, 1], &[2, 3]);
        assert!(generate(&float_ix, &Caps::all())
            .unwrap_err()
            .contains("integers"));
        let mut axes = gather_instr(S32, &[4, 3], &[2, 1], &[2, 3]);
        axes.attrs = ints(&[2]);
        assert!(generate(&axes, &Caps::all())
            .unwrap_err()
            .contains("bad axes"));
        axes.attrs = ints(&[1, 0]);
        axes.operands[1] = buf(S32, &[2, 2]);
        assert!(generate(&axes, &Caps::all())
            .unwrap_err()
            .contains("sorted"));
    }

    #[test]
    fn put_runs_fill_claim_write_over_one_winner_scratch() {
        let p = plan(&indexed_instr("indexed_put", F32));
        assert_eq!(p.passes.len(), 3);
        assert_eq!(p.scratch, vec![4 * 12]);
        assert!(p.passes[0].source.contains("win[get_global_id(0)] = -1;"));
        assert!(p.passes[1]
            .source
            .contains("atomic_max(&win[off], (int)n);"));
        assert!(p.passes[1].args.contains(&KernelArg::ErrorFlag));
        assert!(p.passes[2].source.contains("w >= 0 ?"));
        assert_eq!(p.passes[0].global, vec![12]);
        assert_eq!(p.passes[1].global, vec![15]);
        assert_eq!(p.passes[2].global, vec![12]);
        // The write pass reads the winner, so the claim pass finishes first.
        for pass in &p.passes {
            assert!(pass.args.contains(&KernelArg::Scratch(0)));
        }
    }

    #[test]
    fn put_needs_no_64_bit_atomics() {
        let i = indexed_instr("indexed_put", S64);
        assert!(generate(&i, &no_atomics()).is_ok());
    }

    #[test]
    fn add_on_32_bit_ints_is_atomic_add_on_the_output() {
        for dtype in [S32, U32] {
            let p = plan(&indexed_instr("indexed_add", dtype));
            assert_eq!(p.passes.len(), 2);
            assert!(p.scratch.is_empty());
            assert!(p.passes[1].source.contains("atomic_add((volatile __global"));
            assert!(!p.passes[1].source.contains("int64_base_atomics"));
        }
    }

    #[test]
    fn add_on_f32_is_a_compare_and_swap_loop() {
        let p = plan(&indexed_instr("indexed_add", F32));
        let src = &p.passes[1].source;
        assert!(src.contains("atomic_cmpxchg(p, seen"));
        assert!(src.contains("atomic_add_f32((volatile __global uint*)(out + off), upd);"));
    }

    #[test]
    fn add_on_c64_loops_once_per_part() {
        let p = plan(&indexed_instr("indexed_add", C64));
        let src = &p.passes[1].source;
        assert!(src.contains("atomic_add_f32((volatile __global uint*)cp, cv.x);"));
        assert!(src.contains("atomic_add_f32((volatile __global uint*)(cp + 1), cv.y);"));
    }

    #[test]
    fn add_on_small_types_widens_adds_and_narrows() {
        for (dtype, acc) in [(S8, "int"), (U16, "uint"), (BF16, "float")] {
            let p = plan(&indexed_instr("indexed_add", dtype));
            assert_eq!(p.passes.len(), 3, "{}", dtype);
            assert_eq!(p.scratch, vec![4 * 12]);
            assert!(p.passes[0].source.contains(&format!("{acc}* acc")));
        }
        let bf = plan(&indexed_instr("indexed_add", BF16));
        assert!(bf.passes[0].source.contains("bf16_to_f32(in0[i])"));
        assert!(bf.passes[2].source.contains("f32_to_bf16(acc[i])"));
        let small = plan(&indexed_instr("indexed_add", S8));
        assert!(small.passes[2].source.contains("out[i] = (char)(acc[i]);"));
    }

    #[test]
    fn add_on_64_bit_types_needs_the_atomics_extension() {
        for dtype in [S64, U64, F64, C128] {
            let i = indexed_instr("indexed_add", dtype);
            let err = generate(&i, &no_atomics()).unwrap_err();
            assert!(err.contains("cl_khr_int64_base_atomics"), "{}", err);
            assert!(err.contains(&dtype.name()), "{}", err);
            let p = generate(&i, &Caps::all()).unwrap();
            assert!(p.passes[1]
                .source
                .contains("#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"));
        }
        for dtype in [S32, F32, C64, S8] {
            assert!(generate(&indexed_instr("indexed_add", dtype), &no_atomics()).is_ok());
        }
    }

    #[test]
    fn an_update_pass_ignores_threads_past_the_updates() {
        let p = plan(&indexed_instr("indexed_add", S32));
        assert!(p.passes[1].source.contains("if ((long)i >= nu) return;"));
    }

    #[test]
    fn indexed_ops_check_shapes() {
        let mut i = indexed_instr("indexed_add", F32);
        i.operands[2] = buf(F32, &[4, 3]);
        assert!(generate(&i, &Caps::all()).unwrap_err().contains("rows"));
        let mut i = indexed_instr("indexed_add", F32);
        i.operands[2] = buf(F32, &[5, 4]);
        assert!(generate(&i, &Caps::all()).unwrap_err().contains("larger"));
        let mut i = indexed_instr("indexed_put", F32);
        i.operands[2] = buf(S32, &[5, 3]);
        assert!(generate(&i, &Caps::all())
            .unwrap_err()
            .contains("updates is"));
    }

    #[test]
    fn error_codes_name_the_op() {
        assert_eq!(error_message(1), "index out of bounds in gather");
        assert_eq!(error_message(2), "index out of bounds in indexed_add");
        assert_eq!(error_message(3), "index out of bounds in indexed_put");
    }

    #[test]
    fn every_listed_op_is_handled() {
        for op in OPS {
            assert!(handles(op));
        }
        assert!(!handles("slice"));
    }

    #[test]
    fn u8_values_go_through_the_widen_path_without_the_extension() {
        let i = indexed_instr("indexed_add", U8);
        assert!(generate(&i, &no_atomics()).is_ok());
    }
}
