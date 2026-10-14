//! Data-movement kernels: `broadcast`, `transpose`, `reverse`, `slice`,
//! `put_slice`, `pad`, `concatenate`, `stack`, `iota` and `eye`.
//!
//! Each output element comes from one input element, or from a constant. The
//! kernel takes the flat output id, splits it into coordinates, and maps them
//! to an input index with strides, offsets and clamps. Shapes, strides and
//! offsets reach the kernel as `long` arguments, so the source depends on the
//! rank and the types only, not on the sizes. Values move as whole buffer
//! elements (`float2` for `c64`, `ushort` for `bf16`), so one template serves
//! every dtype. Every op writes a fresh buffer; none works in place.

use super::complex;
use super::elementwise::BF16_HELPERS;
use super::index::{self, coords, dims, strides};
use super::types::{type_info, DType, Half, TypeInfo};
use super::{element_count, Attr, Caps, Instr, KernelArg, KernelSpec};

const OPS: [&str; 10] = [
    "broadcast",
    "transpose",
    "reverse",
    "slice",
    "put_slice",
    "pad",
    "concatenate",
    "stack",
    "iota",
    "eye",
];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

// ── Kernel assembly ─────────────────────────────────────

/// Collects the parameter list and the matching argument list in one order.
struct Kernel<'a> {
    instr: &'a Instr,
    params: Vec<String>,
    args: Vec<KernelArg>,
}

impl<'a> Kernel<'a> {
    fn new(instr: &'a Instr) -> Self {
        Self {
            instr,
            params: Vec::new(),
            args: Vec::new(),
        }
    }

    /// Declare operand `k`: a buffer, or a scalar argument for a constant.
    fn operand(&mut self, k: usize) -> Result<(), String> {
        let o = self
            .instr
            .operands
            .get(k)
            .ok_or_else(|| format!("{} is missing operand {}", self.instr.op, k))?;
        let storage = type_info(o.dtype)?.storage;
        match &o.constant {
            Some(bytes) => {
                self.params.push(format!("const {} in{}", storage, k));
                self.args.push(KernelArg::Scalar(bytes.clone()));
            }
            None => {
                self.params
                    .push(format!("__global const {}* in{}", storage, k));
                self.args.push(KernelArg::Operand(k));
            }
        }
        Ok(())
    }

    fn output(&mut self) -> Result<(), String> {
        let storage = type_info(self.instr.out)?.storage;
        self.params.push(format!("__global {}* out", storage));
        self.args.push(KernelArg::Output);
        Ok(())
    }

    /// Declare the scalar argument `name` with value `v`.
    fn long(&mut self, name: String, v: i64) {
        self.params.push(format!("const long {}", name));
        self.args.push(KernelArg::Scalar(v.to_ne_bytes().to_vec()));
    }

    /// Declare `prefix0`, `prefix1`, ... one per value.
    fn longs(&mut self, prefix: &str, values: &[i64]) {
        for (d, v) in values.iter().enumerate() {
            self.long(format!("{}{}", prefix, d), *v);
        }
    }

    /// Operand `k` as a C expression: the scalar itself for a constant,
    /// else the element at `idx`.
    fn read(&self, k: usize, idx: &str) -> String {
        if self.instr.operands[k].constant.is_some() {
            format!("in{}", k)
        } else {
            format!("in{}[{}]", k, idx)
        }
    }

    fn finish(
        self,
        caps: &Caps,
        entry: String,
        helpers: &str,
        body: &str,
    ) -> Result<KernelSpec, String> {
        let pragmas = pragmas(self.instr, caps)?;
        let source = format!(
            "{pragmas}{helpers}__kernel void {entry}({params}) {{\n{body}}}\n",
            params = self.params.join(", ")
        );
        Ok(KernelSpec {
            source,
            entry,
            build_options: String::new(),
            global: vec![element_count(&self.instr.out_shape).max(1)],
            local: Vec::new(),
            args: self.args,
        })
    }
}

/// The extension pragmas for every type the kernel touches, or an error
/// when the device lacks one.
fn pragmas(instr: &Instr, caps: &Caps) -> Result<String, String> {
    let mut dtypes: Vec<DType> = instr.operands.iter().map(|o| o.dtype).collect();
    dtypes.push(instr.out);
    let mut out = String::new();
    for dtype in dtypes {
        if let Some(ext) = type_info(dtype)?.ext {
            if !caps.has(ext) {
                return Err(format!(
                    "missing extension {}: the device cannot run {}",
                    ext.name(),
                    dtype
                ));
            }
            let pragma = format!("#pragma OPENCL EXTENSION {} : enable\n", ext.name());
            if !out.contains(&pragma) {
                out.push_str(&pragma);
            }
        }
    }
    Ok(out)
}

// ── Index helpers ───────────────────────────────────────

fn check_rank(instr: &Instr) -> Result<usize, String> {
    let widest = instr
        .operands
        .iter()
        .map(|o| o.shape.len())
        .chain(std::iter::once(instr.out_shape.len()))
        .max()
        .unwrap_or(0);
    index::check_rank(&instr.op, widest)?;
    Ok(instr.out_shape.len())
}

fn ints(attrs: &[Attr], what: &str) -> Result<Vec<i64>, String> {
    attrs
        .iter()
        .map(|a| match a {
            Attr::Int(v) => Ok(*v),
            _ => Err(format!("{} attrs must be integers", what)),
        })
        .collect()
}

fn same_type(instr: &Instr, ks: &[usize]) -> Result<(), String> {
    for &k in ks {
        let o = instr
            .operands
            .get(k)
            .ok_or_else(|| format!("{} is missing operand {}", instr.op, k))?;
        if o.dtype != instr.out {
            return Err(format!(
                "{}: operand {} is {}, the output {}",
                instr.op, k, o.dtype, instr.out
            ));
        }
    }
    Ok(())
}

fn expect_operands(instr: &Instr, n: usize) -> Result<(), String> {
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

/// Names of the operand types, for the entry name.
fn type_names(instr: &Instr) -> String {
    let mut names: Vec<String> = instr.operands.iter().map(|o| o.dtype.name()).collect();
    names.push(instr.out.name());
    names.join("_")
}

// ── Strided gather: broadcast, transpose, reverse ───────

/// `out[i] = in0[off + sum(c_d * s_d)]`. A stride is 0 on a broadcast axis and
/// negative on a reversed one.
fn strided(instr: &Instr, caps: &Caps, off: i64, s: &[i64]) -> Result<KernelSpec, String> {
    let rank = instr.out_shape.len();
    let mut k = Kernel::new(instr);
    k.operand(0)?;
    k.output()?;
    k.long("off".to_string(), off);
    k.longs("o", &dims(&instr.out_shape));
    k.longs("s", s);
    let mut body = coords(rank);
    let terms: String = (0..rank).map(|d| format!(" + c{d} * s{d}")).collect();
    body.push_str(&format!("  long j = off{};\n", terms));
    body.push_str(&format!("  out[i] = {};\n", k.read(0, "j")));
    let entry = format!("mv_{}_r{}_{}", instr.op, rank, type_names(instr));
    k.finish(caps, entry, "", &body)
}

fn broadcast(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 1)?;
    same_type(instr, &[0])?;
    let input = &instr.operands[0];
    // A constant fills the output; no index math.
    if input.constant.is_some() {
        let mut k = Kernel::new(instr);
        k.operand(0)?;
        k.output()?;
        let body = format!(
            "  size_t i = get_global_id(0);\n  out[i] = {};\n",
            k.read(0, "0")
        );
        let entry = format!("mv_broadcast_fill_{}", instr.out.name());
        return k.finish(caps, entry, "", &body);
    }
    let axes = ints(&instr.attrs, "broadcast axes")?;
    let out = &instr.out_shape;
    if axes.len() != input.shape.len() {
        return Err(format!(
            "broadcast: {} axes for an operand of rank {}",
            axes.len(),
            input.shape.len()
        ));
    }
    let in_strides = strides(&input.shape);
    let mut s = vec![0i64; out.len()];
    for (j, &axis) in axes.iter().enumerate() {
        let a = usize::try_from(axis)
            .ok()
            .filter(|&a| a < out.len())
            .ok_or_else(|| format!("broadcast: axis {} is out of range", axis))?;
        let dim = input.shape[j];
        if dim != 1 && dim != out[a] {
            return Err(format!(
                "broadcast: dimension {} of size {} does not fit output axis {} of size {}",
                j, dim, a, out[a]
            ));
        }
        if dim != 1 {
            s[a] = in_strides[j];
        }
    }
    strided(instr, caps, 0, &s)
}

fn transpose(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 1)?;
    same_type(instr, &[0])?;
    let axes = ints(&instr.attrs, "transpose axes")?;
    let input = &instr.operands[0].shape;
    if axes.len() != input.len() || input.len() != instr.out_shape.len() {
        return Err("transpose: axes must name every dimension".to_string());
    }
    let in_strides = strides(input);
    let mut s = Vec::with_capacity(axes.len());
    let mut seen = vec![false; axes.len()];
    for (d, &axis) in axes.iter().enumerate() {
        let a = usize::try_from(axis)
            .ok()
            .filter(|&a| a < axes.len() && !seen[a])
            .ok_or_else(|| format!("transpose: axes {:?} are not a permutation", axes))?;
        seen[a] = true;
        if instr.out_shape[d] != input[a] {
            return Err("transpose: the output shape does not follow the axes".to_string());
        }
        s.push(in_strides[a]);
    }
    strided(instr, caps, 0, &s)
}

fn reverse(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 1)?;
    same_type(instr, &[0])?;
    let axes = ints(&instr.attrs, "reverse axes")?;
    let input = &instr.operands[0].shape;
    if *input != instr.out_shape {
        return Err("reverse: the output shape must equal the input shape".to_string());
    }
    let mut s = strides(input);
    let mut off = 0i64;
    let mut done = vec![false; input.len()];
    for &axis in &axes {
        let a = usize::try_from(axis)
            .ok()
            .filter(|&a| a < input.len())
            .ok_or_else(|| format!("reverse: axis {} is out of range", axis))?;
        if !done[a] {
            done[a] = true;
            off += (input[a] as i64 - 1).max(0) * s[a];
            s[a] = -s[a];
        }
    }
    strided(instr, caps, off, &s)
}

// ── slice and put_slice ─────────────────────────────────

/// Statements that read start operand `k` for axis `d` into `s{d}`, clamped
/// to `[0, h{d}]`. A `u64` start compares as unsigned.
fn clamp_start(kernel: &Kernel, k: usize, d: usize) -> Result<String, String> {
    let dtype = kernel.instr.operands[k].dtype;
    let info = type_info(dtype)?;
    if info.is_float() || info.is_complex() {
        return Err(format!(
            "{}: start indices must be integers",
            kernel.instr.op
        ));
    }
    let v = kernel.read(k, "0");
    Ok(if !info.is_signed() && dtype.bits == 64 {
        format!("  ulong u{d} = {v};\n  long s{d} = u{d} > (ulong)h{d} ? h{d} : (long)u{d};\n")
    } else {
        format!(
            "  long s{d} = (long)({v});\n  s{d} = s{d} < 0 ? 0 : (s{d} > h{d} ? h{d} : s{d});\n"
        )
    })
}

/// Operands: the tensor, then one start per axis. Attrs: the input span of
/// each axis, then the stride of each axis.
fn slice(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    let input = &instr.operands.first().ok_or("slice needs a tensor")?.shape;
    let rank = input.len();
    expect_operands(instr, 1 + rank)?;
    same_type(instr, &[0])?;
    let a = ints(&instr.attrs, "slice")?;
    if a.len() != 2 * rank || instr.out_shape.len() != rank {
        return Err("slice: attrs must hold a length and a stride per axis".to_string());
    }
    let (lens, steps) = a.split_at(rank);
    let in_st = strides(input);
    let (mut hi, mut step) = (Vec::new(), Vec::new());
    for d in 0..rank {
        let (len, st) = (lens[d], steps[d]);
        if len < 0 || len > input[d] as i64 || st < 1 {
            return Err(format!(
                "slice: bad length {} or stride {} on axis {}",
                len, st, d
            ));
        }
        let want = if len == 0 { 0 } else { (len - 1) / st + 1 };
        if instr.out_shape[d] as i64 != want {
            return Err(format!("slice: axis {} should give {} elements", d, want));
        }
        hi.push(input[d] as i64 - len);
        step.push(st * in_st[d]);
    }
    let mut k = Kernel::new(instr);
    for n in 0..=rank {
        k.operand(n)?;
    }
    k.output()?;
    k.longs("o", &dims(&instr.out_shape));
    k.longs("h", &hi);
    k.longs("ss", &step);
    k.longs("is", &in_st);
    let mut body = coords(rank);
    for d in 0..rank {
        body.push_str(&clamp_start(&k, 1 + d, d)?);
    }
    let terms: String = (0..rank)
        .map(|d| format!(" + s{d} * is{d} + c{d} * ss{d}"))
        .collect();
    body.push_str(&format!("  long j = 0{};\n", terms));
    body.push_str(&format!("  out[i] = {};\n", k.read(0, "j")));
    let entry = format!("mv_slice_r{}_{}", rank, type_names(instr));
    k.finish(caps, entry, "", &body)
}

/// Operands: the tensor, the update, then one start per axis.
fn put_slice(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    let rank = instr.out_shape.len();
    expect_operands(instr, 2 + rank)?;
    same_type(instr, &[0, 1])?;
    let (target, update) = (&instr.operands[0].shape, &instr.operands[1].shape);
    if *target != instr.out_shape || update.len() != rank {
        return Err("put_slice: the update must have the tensor's rank".to_string());
    }
    if (0..rank).any(|d| update[d] > target[d]) {
        return Err("put_slice: the update is larger than the tensor".to_string());
    }
    let hi: Vec<i64> = (0..rank).map(|d| (target[d] - update[d]) as i64).collect();
    let mut k = Kernel::new(instr);
    for n in 0..2 + rank {
        k.operand(n)?;
    }
    k.output()?;
    k.longs("o", &dims(&instr.out_shape));
    k.longs("h", &hi);
    k.longs("l", &dims(update));
    k.longs("us", &strides(update));
    let mut body = coords(rank);
    for d in 0..rank {
        body.push_str(&clamp_start(&k, 2 + d, d)?);
    }
    body.push_str("  int inside = 1;\n  long k = 0;\n");
    for d in 0..rank {
        body.push_str(&format!(
            "  long p{d} = c{d} - s{d};\n  inside = inside && p{d} >= 0 && p{d} < l{d};\n  k += p{d} * us{d};\n"
        ));
    }
    body.push_str(&format!(
        "  out[i] = inside ? {} : {};\n",
        k.read(1, "k"),
        k.read(0, "i")
    ));
    let entry = format!("mv_put_slice_r{}_{}", rank, type_names(instr));
    k.finish(caps, entry, "", &body)
}

// ── pad ─────────────────────────────────────────────────

/// Operands: the tensor and the pad value. Attrs: low, high and interior
/// padding for each axis. Low and high may be negative.
fn pad(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 2)?;
    same_type(instr, &[0, 1])?;
    let input = &instr.operands[0].shape;
    let rank = input.len();
    if !instr.operands[1].shape.is_empty() || instr.out_shape.len() != rank {
        return Err("pad: the value must be a scalar and the rank must not change".to_string());
    }
    let a = ints(&instr.attrs, "pad config")?;
    if a.len() != 3 * rank {
        return Err("pad: the config needs low, high and interior per axis".to_string());
    }
    let in_st = strides(input);
    let (mut lo, mut step, mut ext) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..rank {
        let (l, h, i) = (a[3 * d], a[3 * d + 1], a[3 * d + 2]);
        let n = input[d] as i64;
        if i < 0 {
            return Err(format!("pad: interior padding {} is negative", i));
        }
        let want = l + h + n + (n - 1).max(0) * i;
        if instr.out_shape[d] as i64 != want {
            return Err(format!("pad: axis {} should have {} elements", d, want));
        }
        lo.push(l);
        step.push(i + 1);
        ext.push(if n == 0 { 0 } else { (n - 1) * (i + 1) + 1 });
    }
    let mut k = Kernel::new(instr);
    k.operand(0)?;
    k.operand(1)?;
    k.output()?;
    k.longs("o", &dims(&instr.out_shape));
    k.longs("lo", &lo);
    k.longs("st", &step);
    k.longs("ex", &ext);
    k.longs("is", &in_st);
    let mut body = coords(rank);
    body.push_str("  int fill = 0;\n  long j = 0;\n");
    for d in 0..rank {
        // A negative low pad shifts p up; a negative high pad shortens `ex`.
        body.push_str(&format!(
            "  {{\n    long p = c{d} - lo{d};\n    if (p < 0 || p >= ex{d} || p % st{d} != 0) fill = 1;\n    else j += (p / st{d}) * is{d};\n  }}\n"
        ));
    }
    body.push_str(&format!(
        "  out[i] = fill ? {} : {};\n",
        k.read(1, "0"),
        k.read(0, "j")
    ));
    let entry = format!("mv_pad_r{}_{}", rank, type_names(instr));
    k.finish(caps, entry, "", &body)
}

// ── concatenate and stack ───────────────────────────────

/// One kernel reads every input. Each input covers a run of the axis, so the
/// output id picks the input by comparing the axis coordinate with the run
/// ends. `stack` is the same kernel with a run of 1 per input.
fn join(instr: &Instr, caps: &Caps, stack: bool) -> Result<KernelSpec, String> {
    let n = instr.operands.len();
    if n == 0 {
        return Err(format!("{} needs at least one operand", instr.op));
    }
    let axis = match instr.attrs.as_slice() {
        [Attr::Int(a)] => usize::try_from(*a).map_err(|_| "axis is negative".to_string())?,
        _ => return Err(format!("{} takes one axis attr", instr.op)),
    };
    let out = &instr.out_shape;
    if axis >= out.len() {
        return Err(format!("{}: axis {} is out of range", instr.op, axis));
    }
    same_type(instr, &(0..n).collect::<Vec<_>>())?;
    let mut lens = Vec::with_capacity(n);
    for (idx, o) in instr.operands.iter().enumerate() {
        let mut want = out.clone();
        if stack {
            want.remove(axis);
            lens.push(1i64);
        } else {
            let len = *o
                .shape
                .get(axis)
                .ok_or_else(|| format!("concatenate: operand {} has too few axes", idx))?;
            want[axis] = len;
            lens.push(len as i64);
        }
        if o.shape != want {
            return Err(format!(
                "{}: operand {} has shape {:?}",
                instr.op, idx, o.shape
            ));
        }
    }
    let total: i64 = lens.iter().sum();
    if total != out[axis] as i64 {
        return Err(format!(
            "{}: the runs add to {}, not {}",
            instr.op, total, out[axis]
        ));
    }
    let inner: i64 = out[axis + 1..].iter().map(|&d| d as i64).product();
    let mut k = Kernel::new(instr);
    for idx in 0..n {
        k.operand(idx)?;
    }
    k.output()?;
    k.long("inner".to_string(), inner);
    k.long("total".to_string(), total);
    let mut begin = 0i64;
    let (mut begins, mut ends) = (Vec::new(), Vec::new());
    for len in &lens {
        begins.push(begin);
        begin += len;
        ends.push(begin);
    }
    k.longs("b", &begins);
    k.longs("e", &ends);
    k.longs("l", &lens);
    let mut body = String::from(
        "  size_t i = get_global_id(0);\n  long id = (long)i;\n  long rr = id % inner;\n  long a = (id / inner) % total;\n  long oo = id / (inner * total);\n",
    );
    for idx in 0..n {
        let src = k.read(idx, &format!("(oo * l{idx} + (a - b{idx})) * inner + rr"));
        let lead = match (idx, idx + 1 == n) {
            (0, true) => "  ".to_string(),
            (0, false) => format!("  if (a < e{idx}) "),
            (_, true) => "  else ".to_string(),
            _ => format!("  else if (a < e{idx}) "),
        };
        body.push_str(&format!("{lead}out[i] = {src};\n"));
    }
    let entry = format!("mv_{}_n{}_{}", instr.op, n, instr.out.name());
    k.finish(caps, entry, "", &body)
}

// ── iota and eye ────────────────────────────────────────

/// The integer expression `v` as a value of the output type.
fn from_long(info: &TypeInfo, v: &str) -> String {
    if info.is_complex() {
        return format!(
            "({})(({})({}), ({})0)",
            info.math, info.component, v, info.component
        );
    }
    match info.half {
        Some(Half::Bf16) => format!("f32_to_bf16((float)({}))", v),
        None if info.storage == "half" => format!("(half)({})", v),
        None if info.is_float() => format!("({})({})", info.math, v),
        None => format!("({})({})", info.storage, v),
    }
}

fn helpers_for(info: &TypeInfo) -> &'static str {
    if info.half == Some(Half::Bf16) {
        BF16_HELPERS
    } else {
        ""
    }
}

/// Attrs: the axis, or the atom `nil` for the flat index.
fn iota(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 0)?;
    let out = &instr.out_shape;
    let info = type_info(instr.out)?;
    let mut k = Kernel::new(instr);
    k.output()?;
    let value = match instr.attrs.as_slice() {
        [Attr::Atom(a)] if a == "nil" => "(long)i".to_string(),
        [Attr::Int(a)] => {
            let axis = usize::try_from(*a)
                .ok()
                .filter(|&a| a < out.len())
                .ok_or_else(|| format!("iota: axis {} is out of range", a))?;
            k.long(
                "inner".to_string(),
                out[axis + 1..].iter().map(|&d| d as i64).product(),
            );
            k.long("dim".to_string(), out[axis] as i64);
            "((long)i / inner) % dim".to_string()
        }
        _ => return Err("iota takes an axis or nil".to_string()),
    };
    let body = format!(
        "  size_t i = get_global_id(0);\n  out[i] = {};\n",
        from_long(&info, &value)
    );
    let flat = if instr.attrs == [Attr::Atom("nil".to_string())] {
        "flat"
    } else {
        "axis"
    };
    let entry = format!("mv_iota_{}_{}", flat, instr.out.name());
    k.finish(caps, entry, helpers_for(&info), &body)
}

/// A batch of identity matrices: the last two axes are rows and columns.
fn eye(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    expect_operands(instr, 0)?;
    let out = &instr.out_shape;
    if out.len() < 2 {
        return Err("eye needs a shape of rank 2 or more".to_string());
    }
    let info = type_info(instr.out)?;
    let (one, zero) = if info.is_complex() {
        (complex::one(&info), complex::zero(&info))
    } else {
        (from_long(&info, "1"), from_long(&info, "0"))
    };
    let mut k = Kernel::new(instr);
    k.output()?;
    k.long("n".to_string(), out[out.len() - 2] as i64);
    k.long("m".to_string(), out[out.len() - 1] as i64);
    let body = format!(
        "  size_t i = get_global_id(0);\n  long col = (long)i % m;\n  long row = ((long)i / m) % n;\n  out[i] = row == col ? {one} : {zero};\n"
    );
    let entry = format!("mv_eye_{}", instr.out.name());
    k.finish(caps, entry, helpers_for(&info), &body)
}

// ── Entry point ─────────────────────────────────────────

/// Generate the kernel for a movement instruction.
pub fn generate(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    check_rank(instr)?;
    match instr.op.as_str() {
        "broadcast" => broadcast(instr, caps),
        "transpose" => transpose(instr, caps),
        "reverse" => reverse(instr, caps),
        "slice" => slice(instr, caps),
        "put_slice" => put_slice(instr, caps),
        "pad" => pad(instr, caps),
        "concatenate" => join(instr, caps, false),
        "stack" => join(instr, caps, true),
        "iota" => iota(instr, caps),
        "eye" => eye(instr, caps),
        other => Err(format!("no movement codegen for op {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{encode_const, ConstVal, Kind};
    use super::super::Operand;
    use super::*;

    const S32: DType = DType::new(Kind::S, 32);
    const S64: DType = DType::new(Kind::S, 64);
    const U64: DType = DType::new(Kind::U, 64);
    const F32: DType = DType::new(Kind::F, 32);
    const F64: DType = DType::new(Kind::F, 64);
    const C64: DType = DType::new(Kind::C, 64);
    const C128: DType = DType::new(Kind::C, 128);
    const BF16: DType = DType::new(Kind::Bf, 16);

    fn buf(dtype: DType, shape: &[usize]) -> Operand {
        Operand {
            dtype,
            shape: shape.to_vec(),
            constant: None,
        }
    }

    fn konst(dtype: DType, v: i128) -> Operand {
        Operand {
            dtype,
            shape: vec![],
            constant: Some(encode_const(dtype, ConstVal::Int(v)).unwrap()),
        }
    }

    fn ints(v: &[i64]) -> Vec<Attr> {
        v.iter().map(|&i| Attr::Int(i)).collect()
    }

    fn instr(
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

    fn gen(i: &Instr) -> KernelSpec {
        generate(i, &Caps::all()).unwrap_or_else(|e| panic!("{}: {}", i.op, e))
    }

    /// The scalar arguments as integers.
    fn longs(spec: &KernelSpec) -> Vec<i64> {
        spec.args
            .iter()
            .filter_map(|a| match a {
                KernelArg::Scalar(b) if b.len() == 8 => {
                    Some(i64::from_ne_bytes(b[..].try_into().unwrap()))
                }
                _ => None,
            })
            .collect()
    }

    // ── broadcast, transpose, reverse ──

    #[test]
    fn broadcast_zeroes_the_stride_of_an_expanded_axis() {
        // {3, 1} to {2, 3, 4}: input axes 1 and 2.
        let i = instr(
            "broadcast",
            vec![buf(F32, &[3, 1])],
            F32,
            &[2, 3, 4],
            ints(&[1, 2]),
        );
        let spec = gen(&i);
        assert_eq!(spec.entry, "mv_broadcast_r3_f32_f32");
        assert_eq!(spec.global, vec![24]);
        // off, o0..o2, s0..s2
        assert_eq!(longs(&spec), vec![0, 2, 3, 4, 0, 1, 0]);
        assert!(spec.source.contains("out[i] = in0[j];"), "{}", spec.source);
    }

    #[test]
    fn a_constant_broadcast_fills_with_no_index_math() {
        let i = instr("broadcast", vec![konst(S32, 7)], S32, &[5, 5], vec![]);
        let spec = gen(&i);
        assert_eq!(spec.entry, "mv_broadcast_fill_s32");
        assert!(spec.source.contains("out[i] = in0;"), "{}", spec.source);
        assert!(spec.source.contains("const int in0"), "{}", spec.source);
        assert_eq!(spec.args[0], KernelArg::Scalar(7i32.to_ne_bytes().to_vec()));
    }

    #[test]
    fn transpose_uses_the_input_strides_in_axis_order() {
        let i = instr(
            "transpose",
            vec![buf(F32, &[2, 3, 4])],
            F32,
            &[4, 2, 3],
            ints(&[2, 0, 1]),
        );
        // off, o (4 2 3), s: input strides 12 4 1 picked by axes 2 0 1
        assert_eq!(longs(&gen(&i)), vec![0, 4, 2, 3, 1, 12, 4]);
    }

    #[test]
    fn reverse_negates_the_stride_and_starts_at_the_far_end() {
        let i = instr("reverse", vec![buf(F32, &[2, 5])], F32, &[2, 5], ints(&[1]));
        // off = 4, o = 2 5, s = 5 -1
        assert_eq!(longs(&gen(&i)), vec![4, 2, 5, 5, -1]);
    }

    #[test]
    fn bad_axes_return_errors() {
        let t = |axes: &[i64]| {
            instr(
                "transpose",
                vec![buf(F32, &[2, 3])],
                F32,
                &[3, 2],
                ints(axes),
            )
        };
        assert!(generate(&t(&[0, 0]), &Caps::all()).is_err());
        assert!(generate(&t(&[0]), &Caps::all()).is_err());
        let r = instr("reverse", vec![buf(F32, &[2])], F32, &[2], ints(&[3]));
        assert!(generate(&r, &Caps::all()).is_err());
        let b = instr("broadcast", vec![buf(F32, &[3])], F32, &[2, 4], ints(&[1]));
        assert!(generate(&b, &Caps::all()).is_err());
    }

    // ── slice and put_slice ──

    #[test]
    fn slice_clamps_the_start_to_zero_and_dim_minus_length() {
        let i = instr(
            "slice",
            vec![buf(F32, &[5, 6]), konst(S64, -3), buf(S32, &[])],
            F32,
            &[2, 3],
            ints(&[2, 3, 1, 1]),
        );
        let spec = gen(&i);
        // The clamp reads `s < 0 ? 0 : (s > h ? h : s)` with h = dim - length.
        assert!(
            spec.source
                .contains("s0 = s0 < 0 ? 0 : (s0 > h0 ? h0 : s0);"),
            "{}",
            spec.source
        );
        assert!(
            spec.source.contains("long s1 = (long)(in2[0]);"),
            "{}",
            spec.source
        );
        // o 2 3, h 3 3, ss 6 1, is 6 1; the constant start is a scalar argument.
        assert_eq!(longs(&spec), vec![-3, 2, 3, 3, 3, 6, 1, 6, 1]);
    }

    #[test]
    fn slice_stride_scales_the_output_step_and_checks_the_output_size() {
        let i = instr(
            "slice",
            vec![buf(F32, &[10]), konst(S64, 1)],
            F32,
            &[3],
            ints(&[7, 3]),
        );
        // length 7, stride 3 gives (7 - 1) / 3 + 1 = 3 elements; h = 3, ss = 3
        let spec = gen(&i);
        assert_eq!(longs(&spec)[1..], [3, 3, 3, 1]);
        let bad = instr(
            "slice",
            vec![buf(F32, &[10]), konst(S64, 1)],
            F32,
            &[4],
            ints(&[7, 3]),
        );
        assert!(generate(&bad, &Caps::all()).is_err());
    }

    #[test]
    fn a_u64_start_compares_as_unsigned() {
        let i = instr(
            "slice",
            vec![buf(F32, &[4]), buf(U64, &[])],
            F32,
            &[2],
            ints(&[2, 1]),
        );
        let spec = gen(&i);
        assert!(
            spec.source.contains("ulong u0 = in1[0];"),
            "{}",
            spec.source
        );
        assert!(
            spec.source.contains("u0 > (ulong)h0 ? h0 : (long)u0"),
            "{}",
            spec.source
        );
    }

    #[test]
    fn a_float_start_is_an_error() {
        let i = instr(
            "slice",
            vec![buf(F32, &[4]), buf(F32, &[])],
            F32,
            &[2],
            ints(&[2, 1]),
        );
        assert!(generate(&i, &Caps::all()).is_err());
    }

    #[test]
    fn put_slice_clamps_and_reads_the_update_inside_the_window() {
        let i = instr(
            "put_slice",
            vec![buf(F32, &[5]), buf(F32, &[2]), buf(S32, &[])],
            F32,
            &[5],
            vec![],
        );
        let spec = gen(&i);
        assert!(
            spec.source
                .contains("s0 = s0 < 0 ? 0 : (s0 > h0 ? h0 : s0);"),
            "{}",
            spec.source
        );
        assert!(
            spec.source.contains("out[i] = inside ? in1[k] : in0[i];"),
            "{}",
            spec.source
        );
        // o 5, h 3, l 2, us 1
        assert_eq!(longs(&spec), vec![5, 3, 2, 1]);
        // The output is a new buffer: no argument aliases the output to an input.
        assert_eq!(
            spec.args
                .iter()
                .filter(|a| **a == KernelArg::Output)
                .count(),
            1
        );
    }

    // ── pad ──

    #[test]
    fn pad_has_a_branch_for_negative_low_and_high_padding() {
        // {4}: low -1, high -2, interior 1 gives -1 - 2 + 4 + 3 = 4 elements.
        let i = instr(
            "pad",
            vec![buf(F32, &[4]), konst(F32, 0)],
            F32,
            &[4],
            ints(&[-1, -2, 1]),
        );
        let spec = gen(&i);
        assert!(
            spec.source
                .contains("if (p < 0 || p >= ex0 || p % st0 != 0) fill = 1;"),
            "{}",
            spec.source
        );
        // o 4, lo -1, st 2, ex 7, is 1
        assert_eq!(longs(&spec), vec![4, -1, 2, 7, 1]);
        assert!(
            spec.source.contains("out[i] = fill ? in1 : in0[j];"),
            "{}",
            spec.source
        );
    }

    #[test]
    fn pad_checks_the_output_shape() {
        let i = instr(
            "pad",
            vec![buf(F32, &[4]), konst(F32, 0)],
            F32,
            &[5],
            ints(&[1, 1, 0]),
        );
        assert!(generate(&i, &Caps::all()).is_err());
    }

    // ── concatenate and stack ──

    #[test]
    fn concatenate_chains_one_branch_per_input_with_run_ends() {
        let i = instr(
            "concatenate",
            vec![buf(F32, &[2, 3]), buf(F32, &[2, 1]), buf(F32, &[2, 2])],
            F32,
            &[2, 6],
            ints(&[1]),
        );
        let spec = gen(&i);
        assert_eq!(spec.entry, "mv_concatenate_n3_f32");
        assert_eq!(spec.global, vec![12]);
        // inner 1, total 6, b 0 3 4, e 3 4 6, l 3 1 2
        assert_eq!(longs(&spec), vec![1, 6, 0, 3, 4, 3, 4, 6, 3, 1, 2]);
        assert!(
            spec.source.contains("if (a < e0) out[i] = in0["),
            "{}",
            spec.source
        );
        assert!(
            spec.source.contains("else if (a < e1) out[i] = in1["),
            "{}",
            spec.source
        );
        assert!(
            spec.source.contains("else out[i] = in2["),
            "{}",
            spec.source
        );
    }

    #[test]
    fn stack_is_a_concatenate_of_runs_of_one() {
        let i = instr(
            "stack",
            vec![buf(F32, &[3]), buf(F32, &[3])],
            F32,
            &[3, 2],
            ints(&[1]),
        );
        let spec = gen(&i);
        // inner 1, total 2, b 0 1, e 1 2, l 1 1
        assert_eq!(longs(&spec), vec![1, 2, 0, 1, 1, 2, 1, 1]);
    }

    #[test]
    fn a_stack_of_scalar_constants_reads_them_by_value() {
        let i = instr(
            "stack",
            vec![konst(S32, 1), konst(S32, 2)],
            S32,
            &[2],
            ints(&[0]),
        );
        let spec = gen(&i);
        assert!(spec.source.contains("out[i] = in0;"), "{}", spec.source);
        assert!(
            spec.source.contains("else out[i] = in1;"),
            "{}",
            spec.source
        );
    }

    #[test]
    fn concatenate_rejects_a_mismatched_run() {
        let i = instr(
            "concatenate",
            vec![buf(F32, &[2]), buf(F32, &[2])],
            F32,
            &[5],
            ints(&[0]),
        );
        assert!(generate(&i, &Caps::all()).is_err());
    }

    #[test]
    fn a_large_concatenate_has_source_linear_in_its_inputs() {
        let n = 64;
        let i = instr(
            "concatenate",
            (0..n).map(|_| buf(F32, &[1, 2])).collect(),
            F32,
            &[1, 2 * n],
            ints(&[1]),
        );
        let spec = gen(&i);
        assert!(spec.source.len() < 200 * n, "{} bytes", spec.source.len());
        assert_eq!(spec.args.len(), n + 1 + 2 + 3 * n);
    }

    // ── iota and eye ──

    #[test]
    fn iota_with_an_axis_divides_by_the_inner_size() {
        let i = instr("iota", vec![], S32, &[2, 3, 4], ints(&[1]));
        let spec = gen(&i);
        assert!(
            spec.source.contains("((long)i / inner) % dim"),
            "{}",
            spec.source
        );
        assert_eq!(longs(&spec), vec![4, 3]);
    }

    #[test]
    fn iota_without_an_axis_is_the_flat_index() {
        let i = instr("iota", vec![], F32, &[2, 3], vec![Attr::Atom("nil".into())]);
        let spec = gen(&i);
        assert!(
            spec.source.contains("out[i] = (float)((long)i);"),
            "{}",
            spec.source
        );
    }

    #[test]
    fn iota_and_eye_build_values_in_the_output_type() {
        let c = gen(&instr("iota", vec![], C64, &[4], ints(&[0])));
        assert!(c.source.contains("(float2)((float)("), "{}", c.source);
        let b = gen(&instr("iota", vec![], BF16, &[4], ints(&[0])));
        assert!(b.source.contains("f32_to_bf16"), "{}", b.source);
        let e = gen(&instr("eye", vec![], C128, &[3, 3], vec![]));
        assert!(e.source.contains("(double2)(1.0, 0.0)"), "{}", e.source);
        assert!(
            e.source.contains("#pragma OPENCL EXTENSION cl_khr_fp64"),
            "{}",
            e.source
        );
    }

    #[test]
    fn eye_takes_the_last_two_axes_as_rows_and_columns() {
        let i = instr("eye", vec![], F32, &[2, 3, 4], vec![]);
        let spec = gen(&i);
        assert_eq!(longs(&spec), vec![3, 4]);
        assert!(
            spec.source.contains("row == col ? (float)(1) : (float)(0)"),
            "{}",
            spec.source
        );
        assert!(generate(&instr("eye", vec![], F32, &[3], vec![]), &Caps::all()).is_err());
    }

    // ── Types, sizes, errors ──

    #[test]
    fn element_width_comes_from_the_type_table() {
        for (dtype, storage) in [
            (C64, "float2"),
            (C128, "double2"),
            (BF16, "ushort"),
            (S64, "long"),
        ] {
            let i = instr(
                "transpose",
                vec![buf(dtype, &[2, 3])],
                dtype,
                &[3, 2],
                ints(&[1, 0]),
            );
            let spec = gen(&i);
            assert!(
                spec.source
                    .contains(&format!("__global const {}* in0", storage)),
                "{}",
                spec.source
            );
            // The same template: no complex preamble.
            assert!(!spec.source.contains("cf_"), "{}", spec.source);
        }
    }

    #[test]
    fn the_source_does_not_depend_on_the_shape() {
        let a = gen(&instr(
            "transpose",
            vec![buf(F32, &[2, 3])],
            F32,
            &[3, 2],
            ints(&[1, 0]),
        ));
        let b = gen(&instr(
            "transpose",
            vec![buf(F32, &[40, 7])],
            F32,
            &[7, 40],
            ints(&[1, 0]),
        ));
        assert_eq!(a.source, b.source);
    }

    #[test]
    fn f64_and_c128_need_the_extension() {
        let no64 = Caps {
            fp64: false,
            fp16: true,
        };
        for dtype in [F64, C128] {
            let i = instr("reverse", vec![buf(dtype, &[2])], dtype, &[2], ints(&[0]));
            let err = generate(&i, &no64).unwrap_err();
            assert!(err.contains("cl_khr_fp64"), "{}", err);
        }
    }

    #[test]
    fn a_rank_above_the_cap_is_an_error_and_the_cap_itself_works() {
        let shape = vec![1usize; index::MAX_RANK + 1];
        let axes: Vec<i64> = (0..=index::MAX_RANK as i64).collect();
        let i = instr(
            "transpose",
            vec![buf(F32, &shape)],
            F32,
            &shape,
            ints(&axes),
        );
        let err = generate(&i, &Caps::all()).unwrap_err();
        assert!(err.contains("rank 9"), "{}", err);
        let shape = vec![1usize; index::MAX_RANK];
        let axes: Vec<i64> = (0..index::MAX_RANK as i64).collect();
        let i = instr(
            "transpose",
            vec![buf(F32, &shape)],
            F32,
            &shape,
            ints(&axes),
        );
        assert!(generate(&i, &Caps::all()).is_ok());
    }

    #[test]
    fn a_zero_size_output_still_gets_a_valid_spec() {
        let i = instr("broadcast", vec![buf(F32, &[0])], F32, &[0], ints(&[0]));
        let spec = gen(&i);
        // The runtime skips the launch when the output has no bytes.
        assert_eq!(spec.global, vec![1]);
    }

    #[test]
    fn entry_names_carry_the_op_rank_and_types() {
        let e = |op: &str, o: Vec<Operand>, shape: &[usize], attrs: Vec<Attr>| {
            gen(&instr(op, o, F32, shape, attrs)).entry
        };
        assert_eq!(
            e("reverse", vec![buf(F32, &[3])], &[3], ints(&[0])),
            "mv_reverse_r1_f32_f32"
        );
        assert_eq!(
            e(
                "pad",
                vec![buf(F32, &[2]), konst(F32, 0)],
                &[3],
                ints(&[1, 0, 0])
            ),
            "mv_pad_r1_f32_f32_f32"
        );
    }

    #[test]
    fn every_listed_op_is_handled() {
        for op in OPS {
            assert!(handles(op));
        }
        assert!(!handles("add"));
    }
}
