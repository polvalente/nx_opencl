//! Reduction kernels: `sum`, `product`, `reduce_max`, `reduce_min`, `all`,
//! `any`, `argmax`, `argmin` and the generic `reduce`.
//!
//! One template serves every op. The strategy comes from the reduced extent
//! `R` and the number of outputs:
//!
//! - `Thread`: `R` is small. One work item per output loops over the extent.
//! - `Group`: one workgroup per output. Each item folds a strided share of the
//!   extent, then the group folds the items in a tree over local memory.
//! - `Sliced(g)`: few outputs and a long extent. `g` workgroups per output
//!   each fold a slice and write a partial; a second launch folds the `g`
//!   partials per output with the same tree.
//!
//! No atomics. The accumulator has the output type (`s32` stays `s32` and wraps,
//! `f32` stays `f32`; `bf16` and `f16` accumulate in `f32`). Reduced axes may sit
//! anywhere: the kernel maps the flat reduced index to an input offset with
//! strides, so no transpose copy is made. The group size and the local array
//! sizes come from the device limits.
//!
//! `argmax` and `argmin` fold `(value, index)` pairs under a total order: a NaN
//! beats every number, the later of two NaNs wins (what `Nx.BinaryBackend` does),
//! and equal values go to the lowest or highest index by `tie_break`.
//!
//! A generic `reduce` arrives as a list of semantic steps over the two
//! accumulator values. A body that is a single `add`, `multiply`, `max` or `min`
//! is the same combiner as `sum`, `product`, `reduce_max` or `reduce_min`, so it
//! generates the same source; the lowering folds its initial accumulator in
//! afterwards. Any other body need not be associative, so a tree would change
//! the answer. It takes the initial accumulator as a second operand and folds
//! the extent in order, one work item per output, as `Nx` does.

use super::complex;
use super::elementwise::{math_of, scalar_binary, ScalarCode, BF16_HELPERS};
use super::index::{self, strides};
use super::movement::{ints, pragmas, Kernel};
use super::types::{type_info, DType, Half, TypeInfo};
use super::{element_count, Attr, Caps, Instr, KernelArg, KernelSpec, Limits, Plan};

const OPS: [&str; 9] = [
    "sum",
    "product",
    "reduce_max",
    "reduce_min",
    "all",
    "any",
    "argmax",
    "argmin",
    "reduce",
];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

/// Extents up to this size run on one work item per output.
pub const SMALL_EXTENT: usize = 64;
/// Outputs at or above this count fill the device without slicing the extent.
const FILL_GROUPS: usize = 64;
/// Elements one work item reads before the tree when the extent is sliced.
const PER_ITEM: usize = 16;
/// Most slices of one extent.
const MAX_SLICES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Thread,
    Group,
    Sliced(usize),
}

/// The strategy for `n_out` outputs of extent `extent`, with groups of `wg` items.
pub fn strategy(extent: usize, n_out: usize, wg: usize) -> Strategy {
    if extent <= SMALL_EXTENT {
        Strategy::Thread
    } else if n_out >= FILL_GROUPS {
        Strategy::Group
    } else {
        let slices = extent.div_ceil(wg * PER_ITEM).clamp(1, MAX_SLICES);
        if slices == 1 {
            Strategy::Group
        } else {
            Strategy::Sliced(slices)
        }
    }
}

// ── Steps and combiners ─────────────────────────────────

/// One step of a generic `reduce` body: `op(v[a], v[b])`. Variables 0 and 1
/// are the two accumulator parameters; step `k` defines variable `2 + k`. The
/// last step is the result.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub op: String,
    pub a: usize,
    pub b: usize,
}

/// Reads `[n, (op, a, b) * n]` from the front of `attrs`. Returns the steps
/// and the number of attrs used.
pub fn parse_steps(attrs: &[Attr]) -> Result<(Vec<Step>, usize), String> {
    let n = match attrs.first() {
        Some(Attr::Int(n)) if *n > 0 => *n as usize,
        _ => return Err("reduce: the step count must be a positive integer".to_string()),
    };
    let mut steps = Vec::with_capacity(n);
    for k in 0..n {
        let at = 1 + 3 * k;
        let (op, a, b) = match (attrs.get(at), attrs.get(at + 1), attrs.get(at + 2)) {
            (Some(Attr::Atom(op)), Some(Attr::Int(a)), Some(Attr::Int(b))) => (op, *a, *b),
            _ => return Err(format!("reduce: step {} must be (op, a, b)", k)),
        };
        let var = |v: i64| -> Result<usize, String> {
            usize::try_from(v)
                .ok()
                .filter(|&v| v < 2 + k)
                .ok_or_else(|| format!("reduce: step {} reads variable {} before it exists", k, v))
        };
        steps.push(Step {
            op: op.clone(),
            a: var(a)?,
            b: var(b)?,
        });
    }
    Ok((steps, 1 + 3 * n))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canon {
    Sum,
    Product,
    Max,
    Min,
}

impl Canon {
    pub fn from_op(op: &str) -> Option<Self> {
        match op {
            "sum" | "window_sum" | "cumulative_sum" => Some(Canon::Sum),
            "product" | "window_product" | "cumulative_product" => Some(Canon::Product),
            "reduce_max" | "window_max" | "cumulative_max" => Some(Canon::Max),
            "reduce_min" | "window_min" | "cumulative_min" => Some(Canon::Min),
            _ => None,
        }
    }

    fn binary(&self) -> &'static str {
        match self {
            Canon::Sum => "add",
            Canon::Product => "multiply",
            Canon::Max => "max",
            Canon::Min => "min",
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            Canon::Sum => "sum",
            Canon::Product => "prod",
            Canon::Max => "max",
            Canon::Min => "min",
        }
    }
}

/// A body of one `add`, `multiply`, `max` or `min` over the two parameters is
/// the canonical combiner; any other body is generic.
pub fn canon_of_steps(steps: &[Step]) -> Option<Canon> {
    match steps {
        [s] if (s.a, s.b) == (0, 1) || (s.a, s.b) == (1, 0) => match s.op.as_str() {
            "add" => Some(Canon::Sum),
            "multiply" => Some(Canon::Product),
            "max" => Some(Canon::Max),
            "min" => Some(Canon::Min),
            _ => None,
        },
        _ => None,
    }
}

pub enum CombKind {
    Canon(Canon),
    All,
    Any,
    Steps(Vec<Step>),
}

/// `comb(new, acc)` as a C function, and the identity element when the
/// combiner has one. Without an identity the kernels track whether a slot
/// holds a value.
pub struct Combiner {
    /// The accumulator type: a math type.
    pub acc: TypeInfo,
    pub code: ScalarCode,
    pub identity: Option<String>,
    pub tag: String,
}

fn zero_of(info: &TypeInfo) -> String {
    if info.is_complex() {
        complex::zero(info)
    } else {
        format!("({})0", info.math)
    }
}

fn one_of(info: &TypeInfo) -> String {
    if info.is_complex() {
        complex::one(info)
    } else {
        format!("({})1", info.math)
    }
}

/// Smallest value of the accumulator type (the identity of `max`).
pub fn lowest(info: &TypeInfo) -> String {
    if info.is_float() {
        "(-INFINITY)".to_string()
    } else {
        info.min.to_string()
    }
}

/// Largest value of the accumulator type (the identity of `min`).
pub fn highest(info: &TypeInfo) -> String {
    if info.is_float() {
        "INFINITY".to_string()
    } else {
        info.max.to_string()
    }
}

/// The identity element of canonical op `canon` over `info`.
pub fn identity_of(canon: Canon, info: &TypeInfo) -> Result<String, String> {
    if info.is_complex() && matches!(canon, Canon::Max | Canon::Min) {
        return Err("max and min do not accept complex values".to_string());
    }
    Ok(match canon {
        Canon::Sum => zero_of(info),
        Canon::Product => one_of(info),
        Canon::Max => lowest(info),
        Canon::Min => highest(info),
    })
}

/// Builds the combiner for `kind` over the accumulator type `acc_dtype`.
pub fn combiner(kind: &CombKind, acc_dtype: DType) -> Result<Combiner, String> {
    let acc = type_info(acc_dtype)?;
    match kind {
        CombKind::Canon(canon) => {
            let identity = identity_of(*canon, &acc)?;
            let code = if acc.is_complex() {
                // Complex sum and product use the helper stage 05 wrote.
                let name = if *canon == Canon::Sum {
                    "sum"
                } else {
                    "product"
                };
                let mut code = ScalarCode::default();
                let needs = complex::accumulate_needs(name)?;
                code.cfns.insert(
                    complex::prefix(&acc),
                    (acc, needs.iter().map(|s| s.to_string()).collect()),
                );
                code.funcs = format!(
                    "static {t} comb({t} in0, {t} in1) {{ return {e}; }}\n",
                    t = acc.math,
                    e = complex::accumulate(&acc, name, "in1", "in0")?
                );
                code
            } else {
                scalar_binary(canon.binary(), acc_dtype, "comb")?
            };
            Ok(Combiner {
                acc,
                code,
                identity: Some(identity),
                tag: canon.tag().to_string(),
            })
        }
        CombKind::All | CombKind::Any => {
            let all = matches!(kind, CombKind::All);
            let (sym, id) = if all { ("&&", 1) } else { ("||", 0) };
            let code = ScalarCode {
                funcs: format!(
                    "static {t} comb({t} in0, {t} in1) {{ return ({t})(in1 {sym} in0); }}\n",
                    t = acc.math
                ),
                ..ScalarCode::default()
            };
            Ok(Combiner {
                acc,
                code,
                identity: Some(format!("({})({})", acc.math, id)),
                tag: (if all { "all" } else { "any" }).to_string(),
            })
        }
        CombKind::Steps(steps) => {
            let mut code = ScalarCode::default();
            let mut body = String::new();
            let var = |v: usize| -> String {
                match v {
                    0 => "in0".to_string(),
                    1 => "in1".to_string(),
                    n => format!("v{}", n),
                }
            };
            for (k, step) in steps.iter().enumerate() {
                code.merge(scalar_binary(&step.op, acc_dtype, &format!("step{}", k))?);
                body.push_str(&format!(
                    "  {t} v{n} = step{k}({a}, {b});\n",
                    t = acc.math,
                    n = 2 + k,
                    a = var(step.a),
                    b = var(step.b)
                ));
            }
            code.funcs.push_str(&format!(
                "static {t} comb({t} in0, {t} in1) {{\n{body}  return v{last};\n}}\n",
                t = acc.math,
                last = 1 + steps.len()
            ));
            let tag = steps
                .iter()
                .map(|s| format!("{}_{}_{}", s.op, s.a, s.b))
                .collect::<Vec<_>>()
                .join("_");
            Ok(Combiner {
                acc,
                code,
                identity: None,
                tag: format!("gen_{}", tag),
            })
        }
    }
}

// ── Loads and stores ────────────────────────────────────

/// The value of buffer element `raw` (type `from`) as a C expression of the
/// math type `to`.
pub fn load_as(from: &TypeInfo, raw: &str, to: &TypeInfo) -> Result<String, String> {
    let value = match from.half {
        Some(Half::Bf16) => format!("bf16_to_f32({})", raw),
        None if from.storage == "half" => format!("(float)({})", raw),
        None => raw.to_string(),
    };
    let own = math_of(from.dtype)?;
    if own.math == to.math {
        Ok(value)
    } else if own.is_complex() || to.is_complex() || (own.is_float() && !to.is_float()) {
        Err(format!(
            "cannot read a {} value as {}",
            from.dtype, to.dtype
        ))
    } else {
        Ok(format!("(({})({}))", to.math, value))
    }
}

/// The expression `expr` (math type of `out`) as a buffer element of `out`.
pub fn store_as(out: &TypeInfo, expr: &str) -> String {
    match out.half {
        Some(Half::Bf16) => format!("f32_to_bf16((float)({}))", expr),
        None if out.storage == "half" => format!("(half)({})", expr),
        None => format!("({})({})", out.storage, expr),
    }
}

/// `BF16_HELPERS` when any type needs them.
pub fn bf16_helpers(types: &[&TypeInfo]) -> &'static str {
    if types.iter().any(|t| t.half == Some(Half::Bf16)) {
        BF16_HELPERS
    } else {
        ""
    }
}

/// Size of one C value of `info.math`, for local and scratch sizes.
pub fn math_bytes(info: &TypeInfo) -> usize {
    match info.math {
        "char" | "uchar" => 1,
        "short" | "ushort" => 2,
        "int" | "uint" | "float" => 4,
        "long" | "ulong" | "double" | "float2" => 8,
        _ => 16,
    }
}

// ── Geometry ────────────────────────────────────────────

/// The kept and reduced axes of an input, with their sizes and input strides.
pub struct Geometry {
    pub kept: Vec<(usize, i64)>,
    pub reduced: Vec<(usize, i64)>,
    pub extent: usize,
    pub n_out: usize,
}

impl Geometry {
    pub fn new(op: &str, input: &[usize], out: &[usize], axes: &[i64]) -> Result<Self, String> {
        index::check_rank(op, input.len())?;
        let st = strides(input);
        let mut is_red = vec![false; input.len()];
        for &a in axes {
            let a = usize::try_from(a)
                .ok()
                .filter(|&a| a < input.len())
                .ok_or_else(|| format!("{}: axis {} is out of range", op, a))?;
            is_red[a] = true;
        }
        let mut kept = Vec::new();
        let mut reduced = Vec::new();
        for d in 0..input.len() {
            if is_red[d] {
                reduced.push((input[d], st[d]));
            } else {
                kept.push((input[d], st[d]));
            }
        }
        let extent: usize = reduced.iter().map(|r| r.0).product();
        let n_out: usize = kept.iter().map(|k| k.0).product();
        if element_count(out) != n_out {
            return Err(format!(
                "{}: the output has {} elements, the kept axes {}",
                op,
                element_count(out),
                n_out
            ));
        }
        Ok(Self {
            kept,
            reduced,
            extent,
            n_out,
        })
    }

    /// Source: `base` from the flat output id `o`.
    fn base_code(&self) -> String {
        let mut s = String::from("  long base = 0;\n");
        if !self.kept.is_empty() {
            s.push_str("  long rem = o;\n");
        }
        for d in (0..self.kept.len()).rev() {
            s.push_str(&format!(
                "  base += (rem % kd{d}) * ks{d};\n  rem /= kd{d};\n"
            ));
        }
        s
    }

    /// Source: `off` from the flat reduced index `j`.
    fn offset_code(&self) -> String {
        match self.reduced.len() {
            0 => "    long off = 0;\n".to_string(),
            1 => "    long off = j * rs0;\n".to_string(),
            n => {
                let mut s = String::from("    long q = j;\n    long off = 0;\n");
                for d in (0..n).rev() {
                    s.push_str(&format!(
                        "    off += (q % rd{d}) * rs{d};\n    q /= rd{d};\n"
                    ));
                }
                s
            }
        }
    }

    /// Declares `kd`, `ks`, `rd`, `rs` as scalar arguments.
    fn declare(&self, k: &mut Kernel) {
        let col = |v: &[(usize, i64)], f: fn(&(usize, i64)) -> i64| -> Vec<i64> {
            v.iter().map(f).collect()
        };
        k.longs("kd", &col(&self.kept, |p| p.0 as i64));
        k.longs("ks", &col(&self.kept, |p| p.1));
        k.longs("rd", &col(&self.reduced, |p| p.0 as i64));
        k.longs("rs", &col(&self.reduced, |p| p.1));
    }

    fn shape_tag(&self) -> String {
        format!("k{}_r{}", self.kept.len(), self.reduced.len())
    }
}

// ── The op, decoded ─────────────────────────────────────

enum Kind {
    /// Fold values with a combiner.
    Fold {
        kind: CombKind,
        /// `all` and `any` fold truth values of the input.
        truth: bool,
        /// Operand 1 is the initial accumulator and the fold runs in order.
        init: bool,
    },
    /// `(value, index)` pairs.
    Arg { max: bool, low: bool },
}

struct Decoded {
    kind: Kind,
    axes: Vec<i64>,
    tag: String,
}

fn decode(instr: &Instr) -> Result<Decoded, String> {
    let op = instr.op.as_str();
    let canon = |c| Kind::Fold {
        kind: CombKind::Canon(c),
        truth: false,
        init: false,
    };
    match op {
        "sum" | "product" | "reduce_max" | "reduce_min" => Ok(Decoded {
            kind: canon(Canon::from_op(op).expect("canonical op")),
            axes: ints(&instr.attrs, "reduction axes")?,
            tag: op.to_string(),
        }),
        "all" | "any" => Ok(Decoded {
            kind: Kind::Fold {
                kind: if op == "all" {
                    CombKind::All
                } else {
                    CombKind::Any
                },
                truth: true,
                init: false,
            },
            axes: ints(&instr.attrs, "reduction axes")?,
            tag: op.to_string(),
        }),
        "argmax" | "argmin" => {
            let low = match instr.attrs.first() {
                Some(Attr::Atom(t)) if t == "low" => true,
                Some(Attr::Atom(t)) if t == "high" => false,
                _ => return Err(format!("{}: tie_break must be low or high", op)),
            };
            Ok(Decoded {
                kind: Kind::Arg {
                    max: op == "argmax",
                    low,
                },
                axes: ints(&instr.attrs[1..], "reduction axes")?,
                tag: format!("{}_{}", op, if low { "low" } else { "high" }),
            })
        }
        _ => {
            let (steps, used) = parse_steps(&instr.attrs)?;
            let axes = ints(&instr.attrs[used..], "reduction axes")?;
            let with_init = instr.operands.len() == 2;
            let (kind, tag) = match (canon_of_steps(&steps), with_init) {
                (Some(c), false) => (canon(c), c.tag().to_string()),
                (None, false) => {
                    return Err("reduce: a body other than add, multiply, max or min needs an initial accumulator operand".to_string())
                }
                (_, true) => {
                    let tag = combiner(&CombKind::Steps(steps.clone()), instr.out)?.tag;
                    (
                        Kind::Fold {
                            kind: CombKind::Steps(steps),
                            truth: false,
                            init: true,
                        },
                        tag,
                    )
                }
            };
            Ok(Decoded { kind, axes, tag })
        }
    }
}

// ── Kernel text ─────────────────────────────────────────

/// Everything the passes share.
struct Ctx<'a> {
    instr: &'a Instr,
    geo: Geometry,
    in_info: TypeInfo,
    out_info: TypeInfo,
    /// Accumulator type (fold) or compared value type (arg).
    acc: TypeInfo,
    head: String,
    wg: usize,
    tag: String,
}

impl Ctx<'_> {
    fn entry(&self, pass: &str) -> String {
        format!(
            "rd_{}_{}_{}_{}_{}",
            self.tag,
            self.in_info.dtype.name(),
            self.out_info.dtype.name(),
            self.geo.shape_tag(),
            pass
        )
    }

    /// Value of input element `idx` for folding.
    fn value(&self, truth: bool, raw: &str) -> Result<String, String> {
        if truth {
            return Ok(if self.in_info.is_complex() {
                format!(
                    "(uchar)({}({}))",
                    complex::fn_name(&self.in_info, "nonzero"),
                    raw
                )
            } else {
                let own = math_of(self.in_info.dtype)?;
                format!("(uchar)({} != 0)", load_as(&self.in_info, raw, &own)?)
            });
        }
        load_as(&self.in_info, raw, &self.acc)
    }

    fn spec(
        &self,
        pass: &str,
        k: Kernel,
        body: String,
        global: usize,
        local: Vec<usize>,
    ) -> KernelSpec {
        let entry = self.entry(pass);
        let source = format!(
            "{head}__kernel void {entry}({params}) {{\n{body}}}\n",
            head = self.head,
            params = k.params.join(", ")
        );
        KernelSpec {
            source,
            entry,
            build_options: String::new(),
            global: vec![global],
            local,
            args: k.args,
        }
    }
}

/// Statement that folds `value` into `acc`: `acc = comb(value, acc)`, or the
/// first value when the combiner has no identity.
fn fold_stmt(has_identity: bool, ty: &str, value: &str, indent: &str) -> String {
    if has_identity {
        format!("{indent}acc = comb({value}, acc);\n")
    } else {
        format!(
            "{indent}{{ {ty} v = {value};\n{indent}  if (has) acc = comb(v, acc); else {{ acc = v; has = 1; }} }}\n"
        )
    }
}

/// Source of the fold prologue: declares `acc` (and `has`).
fn acc_decl(ctx: &Ctx, comb: &Combiner) -> String {
    match &comb.identity {
        Some(id) => format!("  {} acc = {};\n", ctx.acc.math, id),
        None => format!("  {} acc;\n  int has = 0;\n", ctx.acc.math),
    }
}

/// The tree over local memory: folds `sh[0..WG]` into `sh[0]`.
fn tree(comb: &Combiner, wg: usize) -> String {
    let t = comb.acc.math;
    if comb.identity.is_some() {
        format!(
            "  __local {t} sh[{wg}];\n  sh[lid] = acc;\n  barrier(CLK_LOCAL_MEM_FENCE);\n  for (int s = {wg} / 2; s > 0; s >>= 1) {{\n    if (lid < s) sh[lid] = comb(sh[lid + s], sh[lid]);\n    barrier(CLK_LOCAL_MEM_FENCE);\n  }}\n"
        )
    } else {
        format!(
            "  __local {t} sh[{wg}];\n  __local uchar shh[{wg}];\n  sh[lid] = acc;\n  shh[lid] = (uchar)has;\n  barrier(CLK_LOCAL_MEM_FENCE);\n  for (int s = {wg} / 2; s > 0; s >>= 1) {{\n    if (lid < s && shh[lid + s]) {{\n      if (shh[lid]) sh[lid] = comb(sh[lid + s], sh[lid]);\n      else {{ sh[lid] = sh[lid + s]; shh[lid] = 1; }}\n    }}\n    barrier(CLK_LOCAL_MEM_FENCE);\n  }}\n"
        )
    }
}

fn fold_head(ctx: &Ctx, comb: &Combiner, caps_pragmas: &str) -> String {
    format!(
        "{caps_pragmas}{bf}{code}",
        bf = bf16_helpers(&[&ctx.in_info, &ctx.out_info]),
        code = comb.code.render()
    )
}

/// How a pass that reads the input runs.
#[derive(Clone, Copy)]
enum Mode {
    /// One work item per output.
    Thread,
    /// `slices` groups per output; with more than one, each writes a partial.
    Group { slices: usize },
}

/// The pass that reads the input.
fn fold_input_pass(
    ctx: &Ctx,
    comb: &Combiner,
    truth: bool,
    init: bool,
    mode: Mode,
) -> Result<KernelSpec, String> {
    let to_partials = matches!(mode, Mode::Group { slices } if slices > 1);
    let mut k = Kernel::new(ctx.instr);
    k.operand(0)?;
    if init {
        k.operand(1)?;
    }
    // The destination: the output, or the partials buffer.
    if to_partials {
        k.params.push(format!("__global {}* dst", ctx.acc.math));
        k.args.push(KernelArg::Scratch(0));
    } else {
        k.output()?;
    }
    k.long("R".to_string(), ctx.geo.extent as i64);
    if let Mode::Group { slices } = mode {
        k.long("G".to_string(), slices as i64);
    }
    ctx.geo.declare(&mut k);

    let load = ctx.value(truth, &k.read(0, "base + off"))?;
    let store = |expr: &str| {
        if to_partials {
            expr.to_string()
        } else {
            store_as(&ctx.out_info, expr)
        }
    };
    let dst = if to_partials { "dst" } else { "out" };
    // With an initial accumulator the fold starts from it and needs no flag.
    let (decl, seeded) = if init {
        let start = load_as(
            &type_info(ctx.instr.operands[1].dtype)?,
            &k.read(1, "0"),
            &ctx.acc,
        )?;
        (format!("  {} acc = {};\n", ctx.acc.math, start), true)
    } else {
        (acc_decl(ctx, comb), comb.identity.is_some())
    };
    let fold = fold_stmt(seeded, ctx.acc.math, &load, "    ");
    let offs = ctx.geo.offset_code();
    let base = ctx.geo.base_code();

    match mode {
        Mode::Thread => {
            let body = format!(
                "  long o = (long)get_global_id(0);\n{base}{decl}  for (long j = 0; j < R; j++) {{\n{offs}{fold}  }}\n  {dst}[o] = {};\n",
                store("acc")
            );
            Ok(ctx.spec("t", k, body, ctx.geo.n_out, Vec::new()))
        }
        Mode::Group { slices } => {
            let wg = ctx.wg;
            let body = format!(
                "  long lid = (long)get_local_id(0);\n  long grp = (long)get_group_id(0);\n  long o = grp / G;\n  long g = grp % G;\n{base}{decl}  for (long j = g * {wg} + lid; j < R; j += G * {wg}) {{\n{offs}{fold}  }}\n{tree}  if (lid == 0) {dst}[grp] = {s};\n",
                tree = tree(comb, wg),
                s = store("sh[0]")
            );
            Ok(ctx.spec("g", k, body, ctx.geo.n_out * slices * wg, vec![wg]))
        }
    }
}

/// The second pass: one group per output folds the `slices` partials.
fn fold_partials_pass(ctx: &Ctx, comb: &Combiner, slices: usize) -> Result<KernelSpec, String> {
    let mut k = Kernel::new(ctx.instr);
    k.params
        .push(format!("__global const {}* pv", ctx.acc.math));
    k.args.push(KernelArg::Scratch(0));
    k.output()?;
    k.long("G".to_string(), slices as i64);
    let wg = ctx.wg;
    let decl = acc_decl(ctx, comb);
    let fold = fold_stmt(
        comb.identity.is_some(),
        ctx.acc.math,
        "pv[o * G + q]",
        "    ",
    );
    let body = format!(
        "  long lid = (long)get_local_id(0);\n  long o = (long)get_group_id(0);\n{decl}  for (long q = lid; q < G; q += {wg}) {{\n{fold}  }}\n{tree}  if (lid == 0) out[o] = {s};\n",
        tree = tree(comb, wg),
        s = store_as(&ctx.out_info, "sh[0]")
    );
    Ok(ctx.spec("p2", k, body, ctx.geo.n_out * wg, vec![wg]))
}

// ── Argmax and argmin ───────────────────────────────────

fn better_fn(ctx: &Ctx, max: bool, low: bool) -> String {
    let t = ctx.acc.math;
    let cmp = if max { ">" } else { "<" };
    let nan = if ctx.acc.is_float() {
        "  if (isnan(v)) return isnan(w) ? i > k : 1;\n  if (isnan(w)) return 0;\n"
    } else {
        ""
    };
    let tie = if low { "i < k" } else { "i > k" };
    format!(
        "static int better({t} v, long i, {t} w, long k) {{\n{nan}  if (v {cmp} w) return 1;\n  if (w {cmp} v) return 0;\n  return {tie};\n}}\n"
    )
}

fn arg_tree(ctx: &Ctx) -> String {
    let (t, wg) = (ctx.acc.math, ctx.wg);
    format!(
        "  __local {t} shv[{wg}];\n  __local long shi[{wg}];\n  shv[lid] = bv;\n  shi[lid] = bi;\n  barrier(CLK_LOCAL_MEM_FENCE);\n  for (int s = {wg} / 2; s > 0; s >>= 1) {{\n    if (lid < s && shi[lid + s] >= 0 && (shi[lid] < 0 || better(shv[lid + s], shi[lid + s], shv[lid], shi[lid]))) {{\n      shv[lid] = shv[lid + s];\n      shi[lid] = shi[lid + s];\n    }}\n    barrier(CLK_LOCAL_MEM_FENCE);\n  }}\n"
    )
}

fn arg_input_pass(ctx: &Ctx, mode: Mode) -> Result<KernelSpec, String> {
    let to_partials = matches!(mode, Mode::Group { slices } if slices > 1);
    let t = ctx.acc.math;
    let mut k = Kernel::new(ctx.instr);
    k.operand(0)?;
    if to_partials {
        k.params.push(format!("__global {t}* dv"));
        k.args.push(KernelArg::Scratch(0));
        k.params.push("__global long* di".to_string());
        k.args.push(KernelArg::Scratch(1));
    } else {
        k.output()?;
    }
    k.long("R".to_string(), ctx.geo.extent as i64);
    if let Mode::Group { slices } = mode {
        k.long("G".to_string(), slices as i64);
    }
    ctx.geo.declare(&mut k);
    let load = load_as(&ctx.in_info, &k.read(0, "base + off"), &ctx.acc)?;
    let offs = ctx.geo.offset_code();
    let base = ctx.geo.base_code();
    let fold = format!(
        "{offs}    {t} v = {load};\n    if (bi < 0 || better(v, j, bv, bi)) {{ bv = v; bi = j; }}\n"
    );
    match mode {
        Mode::Thread => {
            let body = format!(
                "  long o = (long)get_global_id(0);\n{base}  {t} bv;\n  long bi = -1;\n  for (long j = 0; j < R; j++) {{\n{fold}  }}\n  out[o] = {};\n",
                store_as(&ctx.out_info, "bi")
            );
            Ok(ctx.spec("t", k, body, ctx.geo.n_out, Vec::new()))
        }
        Mode::Group { slices } => {
            let wg = ctx.wg;
            let write = if to_partials {
                "  if (lid == 0) { dv[grp] = shv[0]; di[grp] = shi[0]; }\n".to_string()
            } else {
                format!(
                    "  if (lid == 0) out[grp] = {};\n",
                    store_as(&ctx.out_info, "shi[0]")
                )
            };
            let body = format!(
                "  long lid = (long)get_local_id(0);\n  long grp = (long)get_group_id(0);\n  long o = grp / G;\n  long g = grp % G;\n{base}  {t} bv;\n  long bi = -1;\n  for (long j = g * {wg} + lid; j < R; j += G * {wg}) {{\n{fold}  }}\n{tree}{write}",
                tree = arg_tree(ctx)
            );
            Ok(ctx.spec("g", k, body, ctx.geo.n_out * slices * wg, vec![wg]))
        }
    }
}

fn arg_partials_pass(ctx: &Ctx, slices: usize) -> Result<KernelSpec, String> {
    let t = ctx.acc.math;
    let mut k = Kernel::new(ctx.instr);
    k.params.push(format!("__global const {t}* pv"));
    k.args.push(KernelArg::Scratch(0));
    k.params.push("__global const long* pi".to_string());
    k.args.push(KernelArg::Scratch(1));
    k.output()?;
    k.long("G".to_string(), slices as i64);
    let wg = ctx.wg;
    let body = format!(
        "  long lid = (long)get_local_id(0);\n  long o = (long)get_group_id(0);\n  {t} bv;\n  long bi = -1;\n  for (long q = lid; q < G; q += {wg}) {{\n    {t} v = pv[o * G + q];\n    long j = pi[o * G + q];\n    if (bi < 0 || better(v, j, bv, bi)) {{ bv = v; bi = j; }}\n  }}\n{tree}  if (lid == 0) out[o] = {s};\n",
        tree = arg_tree(ctx),
        s = store_as(&ctx.out_info, "shi[0]")
    );
    Ok(ctx.spec("p2", k, body, ctx.geo.n_out * wg, vec![wg]))
}

// ── Generation ──────────────────────────────────────────

pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    let decoded = decode(instr)?;
    let init = matches!(decoded.kind, Kind::Fold { init: true, .. });
    let want = if init { 2 } else { 1 };
    if instr.operands.len() != want {
        return Err(format!(
            "{} takes {} operands, got {}",
            instr.op,
            want,
            instr.operands.len()
        ));
    }
    let input = &instr.operands[0];
    let geo = Geometry::new(&instr.op, &input.shape, &instr.out_shape, &decoded.axes)?;
    let in_info = type_info(input.dtype)?;
    let out_info = type_info(instr.out)?;
    let pragma_text = pragmas(instr, caps)?;

    let acc = match &decoded.kind {
        Kind::Fold { .. } => math_of(instr.out)?,
        Kind::Arg { .. } => {
            if in_info.is_complex() {
                return Err(format!("{} does not accept complex values", instr.op));
            }
            math_of(input.dtype)?
        }
    };
    // Per item: the accumulator, plus a `long` index and a flag for the rest.
    let item_bytes = math_bytes(&acc) + 9;
    let wg = limits.group_size(item_bytes);
    let mut ctx = Ctx {
        instr,
        geo,
        in_info,
        out_info,
        acc,
        head: String::new(),
        wg,
        tag: decoded.tag.clone(),
    };

    let comb = match &decoded.kind {
        Kind::Fold { kind, .. } => Some(combiner(kind, acc.dtype)?),
        Kind::Arg { .. } => None,
    };
    match (&decoded.kind, &comb) {
        (Kind::Fold { truth, .. }, Some(c)) => {
            ctx.head = fold_head(&ctx, c, &pragma_text);
            ctx.tag = c.tag.clone();
            // `all` and `any` read truth values of the input type.
            if *truth && ctx.in_info.is_complex() {
                ctx.head
                    .push_str(&complex::preamble(&ctx.in_info, &["nonzero"]));
            }
        }
        (Kind::Arg { max, low }, _) => {
            ctx.head = format!(
                "{pragma_text}{}{}",
                bf16_helpers(&[&ctx.in_info, &ctx.out_info]),
                better_fn(&ctx, *max, *low)
            );
        }
        _ => {}
    }

    let n_out = ctx.geo.n_out;
    let extent = ctx.geo.extent;

    // An empty extent has no elements to read: fill with the identity. With an
    // initial accumulator the thread fold runs zero times and writes it.
    if extent == 0 && !init {
        return empty_extent(&ctx, comb.as_ref());
    }

    let truth = matches!(decoded.kind, Kind::Fold { truth: true, .. });
    let is_arg = matches!(decoded.kind, Kind::Arg { .. });
    let first = |mode: Mode| -> Result<KernelSpec, String> {
        match &comb {
            Some(c) => fold_input_pass(&ctx, c, truth, init, mode),
            None => arg_input_pass(&ctx, mode),
        }
    };
    let strat = if init {
        Strategy::Thread
    } else {
        strategy(extent, n_out, wg)
    };
    match strat {
        Strategy::Thread => Ok(Plan::single(first(Mode::Thread)?)),
        Strategy::Group => Ok(Plan::single(first(Mode::Group { slices: 1 })?)),
        Strategy::Sliced(g) => {
            let second = match &comb {
                Some(c) => fold_partials_pass(&ctx, c, g)?,
                None => arg_partials_pass(&ctx, g)?,
            };
            let mut scratch = vec![n_out * g * math_bytes(&ctx.acc)];
            if is_arg {
                scratch.push(n_out * g * 8);
            }
            Ok(Plan {
                passes: vec![first(Mode::Group { slices: g })?, second],
                scratch,
            })
        }
    }
}

/// A reduction over an empty extent: every output is the identity, if the
/// op has one.
fn empty_extent(ctx: &Ctx, comb: Option<&Combiner>) -> Result<Plan, String> {
    let id = comb.and_then(|c| c.identity.clone()).ok_or_else(|| {
        format!(
            "{} over an empty axis has no identity element",
            ctx.instr.op
        )
    })?;
    let mut k = Kernel::new(ctx.instr);
    k.output()?;
    let body = format!(
        "  size_t i = get_global_id(0);\n  out[i] = {};\n",
        store_as(&ctx.out_info, &id)
    );
    Ok(Plan::single(ctx.spec(
        "empty",
        k,
        body,
        ctx.geo.n_out.max(1),
        Vec::new(),
    )))
}

#[cfg(test)]
mod tests {
    use super::super::types::Kind as K;
    use super::super::Operand;
    use super::*;

    const S32: DType = DType::new(K::S, 32);
    const S64: DType = DType::new(K::S, 64);
    const U8: DType = DType::new(K::U, 8);
    const U64: DType = DType::new(K::U, 64);
    const F32: DType = DType::new(K::F, 32);
    const F64: DType = DType::new(K::F, 64);
    const BF16: DType = DType::new(K::Bf, 16);
    const C64: DType = DType::new(K::C, 64);

    fn buf(dtype: DType, shape: &[usize]) -> Operand {
        Operand {
            dtype,
            shape: shape.to_vec(),
            constant: None,
        }
    }

    fn ints(v: &[i64]) -> Vec<Attr> {
        v.iter().map(|&i| Attr::Int(i)).collect()
    }

    fn instr(
        op: &str,
        input: DType,
        shape: &[usize],
        out: DType,
        out_shape: &[usize],
        attrs: Vec<Attr>,
    ) -> Instr {
        Instr {
            op: op.to_string(),
            operands: vec![buf(input, shape)],
            out,
            out_shape: out_shape.to_vec(),
            attrs,
        }
    }

    fn plan(i: &Instr) -> Plan {
        generate(i, &Caps::all(), &Limits::apple()).unwrap()
    }

    fn sources(p: &Plan) -> Vec<&str> {
        p.passes.iter().map(|s| s.source.as_str()).collect()
    }

    #[test]
    fn strategy_follows_extent_and_output_count() {
        let wg = 256;
        // Up to 64 elements one thread folds a line.
        assert_eq!(strategy(1, 1, wg), Strategy::Thread);
        assert_eq!(strategy(63, 1, wg), Strategy::Thread);
        assert_eq!(strategy(64, 1, wg), Strategy::Thread);
        // Past that, a group folds a line; 65 elements need one slice only.
        assert_eq!(strategy(65, 1, wg), Strategy::Group);
        assert_eq!(strategy(4096, 1, wg), Strategy::Group);
        // 4097 elements at 16 per item need two slices.
        assert_eq!(strategy(4097, 1, wg), Strategy::Sliced(2));
        assert_eq!(strategy(1 << 20, 1, wg), Strategy::Sliced(256));
        // Many outputs fill the device with one group each, however long the line.
        assert_eq!(strategy(4097, 64, wg), Strategy::Group);
        assert_eq!(strategy(1 << 20, 4096, wg), Strategy::Group);
        assert_eq!(strategy(4097, 63, wg), Strategy::Sliced(2));
        // The slice count stops at 1024.
        assert_eq!(strategy(1 << 30, 1, wg), Strategy::Sliced(1024));
        // A smaller group gives more slices.
        assert_eq!(strategy(4097, 1, 64), Strategy::Sliced(5));
    }

    #[test]
    fn accumulator_has_the_width_of_the_nx_output_type() {
        let acc = |out: DType| {
            combiner(&CombKind::Canon(Canon::Sum), out)
                .unwrap()
                .acc
                .math
        };
        assert_eq!(acc(S32), "int");
        assert_eq!(acc(S64), "long");
        assert_eq!(acc(U64), "ulong");
        assert_eq!(acc(F32), "float");
        assert_eq!(acc(F64), "double");
        // Half-width floats accumulate in float.
        assert_eq!(acc(BF16), "float");
        assert_eq!(acc(C64), "float2");

        // `sum` of `u8` has output type `u64`: the kernel folds in `ulong`.
        let p = plan(&instr("sum", U8, &[100], U64, &[], ints(&[0])));
        assert!(sources(&p)[0].contains("ulong acc"));
    }

    #[test]
    fn identities_per_op_and_dtype() {
        let id = |c, d| identity_of(c, &type_info(d).unwrap()).unwrap();
        assert_eq!(id(Canon::Sum, F32), "(float)0");
        assert_eq!(id(Canon::Product, F32), "(float)1");
        assert_eq!(id(Canon::Max, F32), "(-INFINITY)");
        assert_eq!(id(Canon::Min, F32), "INFINITY");
        assert_eq!(id(Canon::Sum, S64), "(long)0");
        // Integers use the limits of their own type.
        for d in [S32, S64, U8, U64] {
            let info = type_info(d).unwrap();
            assert_eq!(id(Canon::Max, d), info.min);
            assert_eq!(id(Canon::Min, d), info.max);
        }
        // Complex values have no order, so max and min have no identity.
        let c = type_info(C64).unwrap();
        assert!(identity_of(Canon::Max, &c).is_err());
        assert!(identity_of(Canon::Sum, &c).is_ok());
    }

    #[test]
    fn argmax_orders_nan_first_and_ties_by_tie_break() {
        let src = |op: &str, tie: &str| {
            let a = vec![Attr::Atom(tie.to_string()), Attr::Int(0)];
            sources(&plan(&instr(op, F32, &[100], S64, &[], a)))[0].to_string()
        };
        // NaN beats a number; of two NaNs the later index wins.
        assert!(src("argmax", "low").contains("if (isnan(v)) return isnan(w) ? i > k : 1;"));
        assert!(src("argmax", "low").contains("if (isnan(w)) return 0;"));
        // Ties go to the lowest or highest index.
        assert!(src("argmax", "low").contains("return i < k;"));
        assert!(src("argmax", "high").contains("return i > k;"));
        assert!(src("argmax", "low").contains("v > w"));
        assert!(src("argmin", "low").contains("v < w"));
        // Integers have no NaN arm.
        let a = vec![Attr::Atom("low".to_string()), Attr::Int(0)];
        let int = plan(&instr("argmax", S32, &[100], S64, &[], a));
        assert!(!sources(&int)[0].contains("isnan"));
        // The tie break is part of the kernel hash.
        assert_ne!(src("argmax", "low"), src("argmax", "high"));
    }

    #[test]
    fn reduce_with_an_add_body_generates_the_source_of_sum() {
        let sum = plan(&instr("sum", F32, &[3, 100], F32, &[3], ints(&[1])));
        // One step, `add(v0, v1)`, then the axes.
        let attrs = vec![
            Attr::Int(1),
            Attr::Atom("add".to_string()),
            Attr::Int(0),
            Attr::Int(1),
            Attr::Int(1),
        ];
        let reduce = plan(&instr("reduce", F32, &[3, 100], F32, &[3], attrs));
        assert_eq!(sources(&sum), sources(&reduce));
        assert_eq!(
            sum.passes.iter().map(|p| &p.entry).collect::<Vec<_>>(),
            reduce.passes.iter().map(|p| &p.entry).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_body_without_an_initial_value_must_be_a_monoid() {
        let attrs = vec![
            Attr::Int(2),
            Attr::Atom("multiply".to_string()),
            Attr::Int(0),
            Attr::Int(0),
            Attr::Atom("add".to_string()),
            Attr::Int(2),
            Attr::Int(1),
            Attr::Int(1),
        ];
        let err = generate(
            &instr("reduce", F32, &[3, 100], F32, &[3], attrs),
            &Caps::all(),
            &Limits::apple(),
        )
        .unwrap_err();
        assert!(err.contains("initial accumulator"), "{err}");
    }

    #[test]
    fn plans_have_the_passes_the_strategy_names() {
        // One thread per line.
        let p = plan(&instr("sum", F32, &[3, 20], F32, &[3], ints(&[1])));
        assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
        // One group per line.
        let p = plan(&instr("sum", F32, &[128, 1000], F32, &[128], ints(&[1])));
        assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
        assert_eq!(p.passes[0].local, vec![256]);
        // Slices: partials in scratch, then a fold of them.
        let p = plan(&instr("sum", F32, &[2, 100_000], F32, &[2], ints(&[1])));
        assert_eq!(p.passes.len(), 2);
        assert_eq!(p.scratch, vec![2 * 25 * 4]);
        // The kernel never uses atomics.
        for s in sources(&p) {
            assert!(!s.contains("atomic"));
        }
    }

    #[test]
    fn group_size_obeys_the_device_limits() {
        // A device with 64 threads and 1 KiB of local memory.
        let small = Limits {
            max_wg: 64,
            local_mem: 1024,
        };
        let i = instr("sum", F64, &[2, 100_000], F64, &[2], ints(&[1]));
        let p = generate(&i, &Caps::all(), &small).unwrap();
        for s in &p.passes {
            let wg = s.local.first().copied().unwrap_or(1);
            assert!(wg <= 64 && wg * 8 <= 1024, "{wg}");
        }
    }

    #[test]
    fn an_empty_axis_fills_with_the_identity_and_arg_ops_refuse() {
        let p = plan(&instr("reduce_max", F32, &[3, 0], F32, &[3], ints(&[1])));
        assert_eq!(p.passes.len(), 1);
        assert!(p.passes[0].source.contains("-INFINITY"));
        let a = vec![Attr::Atom("low".to_string()), Attr::Int(1)];
        let err = generate(
            &instr("argmax", F32, &[3, 0], S64, &[3], a),
            &Caps::all(),
            &Limits::apple(),
        );
        assert!(err.unwrap_err().contains("empty axis"));
    }
}
