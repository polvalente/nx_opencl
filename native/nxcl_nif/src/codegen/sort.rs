//! `sort` and `argsort` along any axis, ascending or descending.
//!
//! A bitonic network over a line padded to the next power of two `P`. Every
//! compare uses a total order, so the result is stable: the key decides, then
//! the original index, and a padding slot (index at or past `N`) sorts last.
//! A NaN sorts last in ascending order and first in descending order, as in
//! `Nx.BinaryBackend`. The key and its index move together.
//!
//! - `P` at most the local capacity `M`: one launch sorts a line in local memory.
//! - Longer lines: pass 1 sorts blocks of `M` in local memory into a scratch
//!   copy. For each merge size above `M`, one launch per compare distance of
//!   `M` or more works on the scratch in global memory, and one launch finishes
//!   the distances below `M` in local memory. A last launch copies the first
//!   `N` slots of each line to the output.
//!
//! `M` comes from the device's local memory size and key width, capped at 4096.
//! No atomics.

use super::index::{self, LINE};
use super::movement::{ints, pragmas, Kernel};
use super::types::{bf16_helpers, load_as, math_bytes, math_of, store_as, type_info, TypeInfo};
use super::{
    checked_product, entry_name, expect_operands, Attr, Caps, Instr, KernelArg, KernelSpec, Limits,
    Plan,
};

const OPS: [&str; 2] = ["sort", "argsort"];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

const MAX_LOCAL: usize = 4096;

/// Local capacity (a power of two) for keys of `key_bytes`, plus a 4-byte index.
fn local_capacity(limits: &Limits, key_bytes: usize) -> usize {
    let mut m = 1;
    while m * 2 <= MAX_LOCAL && m * 2 * (key_bytes + 4) <= limits.local_mem {
        m *= 2;
    }
    m
}

fn padded(n: usize) -> Result<usize, String> {
    n.max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| format!("a sort axis of {} elements is too long", n))
}

/// The largest padded line: the kernels carry indices as `int`.
const MAX_PADDED: usize = i32::MAX as usize;

/// Launches a sort of one line of `n` elements takes with local capacity `m`.
#[cfg(test)]
fn pass_count(n: usize, m: usize) -> usize {
    let p = padded(n).unwrap();
    let m = m.min(p);
    if p <= m {
        return 1;
    }
    let mut count = 1;
    let mut kk = 2 * m;
    while kk <= p {
        // Distances `kk/2` down to `m` are global; the rest run in one local launch.
        count += (kk / 2 / m).trailing_zeros() as usize + 1 + 1;
        kk *= 2;
    }
    count + 1
}

struct Ctx<'a> {
    instr: &'a Instr,
    in_info: TypeInfo,
    out_info: TypeInfo,
    key: TypeInfo,
    head: String,
    /// Local capacity actually used: `min(P, capacity)`.
    m: usize,
    /// Work items per group.
    t: usize,
    p: usize,
    n: usize,
    inner: usize,
    lines: usize,
    arg: bool,
}

impl Ctx<'_> {
    fn entry(&self, pass: &str) -> String {
        entry_name(
            "so",
            &[
                if self.arg { "argsort" } else { "sort" },
                &self.in_info.dtype.name(),
                &self.out_info.dtype.name(),
                pass,
                &format!("m{}", self.m),
            ],
        )
    }

    /// A pass of `groups` groups of `t` items.
    fn spec(
        &self,
        pass: &str,
        mut k: Kernel,
        body: String,
        groups: usize,
    ) -> Result<KernelSpec, String> {
        let global = checked_product("the sort launch size", &[groups, self.t])?;
        let body = format!("{}{body}", k.guard_groups(groups)?);
        Ok(k.into_spec(
            &self.head,
            self.entry(pass),
            &body,
            vec![global],
            vec![self.t],
        ))
    }

    /// A pass of one item per element of `items`, rounded up to whole groups.
    fn spec_items(
        &self,
        pass: &str,
        mut k: Kernel,
        body: String,
        items: usize,
    ) -> Result<KernelSpec, String> {
        let groups = items.div_ceil(self.t);
        let global = checked_product("the sort launch size", &[groups, self.t])?;
        // The launch rounds up to whole groups, so the extra items stop here.
        let body = format!("{}{body}", k.guard(items)?);
        Ok(k.into_spec(
            &self.head,
            self.entry(pass),
            &body,
            vec![global],
            vec![self.t],
        ))
    }

    /// The output element for a sorted pair: `key` for `sort`, `idx` for `argsort`.
    fn result(&self, key: &str, idx: &str) -> String {
        if self.arg {
            store_as(&self.out_info, idx)
        } else {
            store_as(&self.out_info, key)
        }
    }
}

/// The compare function. `before(a, ia, b, ib, n)` is true when `(a, ia)`
/// goes before `(b, ib)` in the final order.
fn before_fn(key: &TypeInfo, desc: bool) -> String {
    let t = key.math;
    let (lt, gt) = if desc { (">", "<") } else { ("<", ">") };
    let nan = if key.is_float() {
        let pick = if desc { "na" } else { "nb" };
        format!(
            "  int na = isnan(a), nb = isnan(b);\n  if (na || nb) return (na && nb) ? ia < ib : {pick};\n"
        )
    } else {
        String::new()
    };
    format!(
        "static int before({t} a, int ia, {t} b, int ib, long n) {{\n  if (ia >= n) return ib >= n ? ia < ib : 0;\n  if (ib >= n) return 1;\n{nan}  if (a {lt} b) return 1;\n  if (a {gt} b) return 0;\n  return ia < ib;\n}}\n"
    )
}

/// One compare-exchange over the pairs `t` of a local or global array.
/// `lo`, `hi`, `asc` are in scope; `key` and `idx` name the arrays.
fn exchange(key: &str, idx: &str, kt: &str, n: &str) -> String {
    format!(
        "      int sw = asc ? before({key}[hi], {idx}[hi], {key}[lo], {idx}[lo], {n}) : before({key}[lo], {idx}[lo], {key}[hi], {idx}[hi], {n});\n      if (sw) {{ {kt} tk = {key}[lo]; {key}[lo] = {key}[hi]; {key}[hi] = tk; int ti = {idx}[lo]; {idx}[lo] = {idx}[hi]; {idx}[hi] = ti; }}\n"
    )
}

const PAIR: &str =
    "      long lo = ((t & ~(j - 1)) << 1) | (t & (j - 1));\n      long hi = lo + j;\n";

fn geometry(ctx: &Ctx, k: &mut Kernel) {
    k.long("N".to_string(), ctx.n as i64);
    k.long("inner".to_string(), ctx.inner as i64);
}

/// Pass 1: sort blocks of `M` in local memory.
fn pass_block(ctx: &Ctx) -> Result<KernelSpec, String> {
    let (m, t, kt) = (ctx.m, ctx.t, ctx.key.math);
    let single = ctx.p <= m;
    let nb = ctx.p / m;
    let mut k = Kernel::new(ctx.instr);
    k.operand(0)?;
    k.output()?;
    if !single {
        k.params.push(format!("__global {kt}* skeys"));
        k.args.push(KernelArg::Scratch(0));
        k.params.push("__global int* sidx".to_string());
        k.args.push(KernelArg::Scratch(1));
    }
    geometry(ctx, &mut k);
    k.long("P".to_string(), ctx.p as i64);
    let load = load_as(&ctx.in_info, &k.read(0, "base + kpos * inner"), &ctx.key)?;
    let write = if single {
        format!(
            "    long kpos = gbase + e;\n    if (kpos < N) out[base + kpos * inner] = {};\n",
            ctx.result("sk[e]", "si[e]")
        )
    } else {
        "    skeys[line * P + gbase + e] = sk[e];\n    sidx[line * P + gbase + e] = si[e];\n"
            .to_string()
    };
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long grp = (long)get_group_id(0);
  long line = grp / {nb};
  long blk = grp % {nb};
{LINE}  long gbase = blk * {m};
  __local {kt} sk[{m}];
  __local int si[{m}];
  for (long e = lid; e < {m}; e += {t}) {{
    long kpos = gbase + e;
    {kt} v = ({kt})0;
    if (kpos < N) v = {load};
    sk[e] = v;
    si[e] = (int)kpos;
  }}
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long kk = 2; kk <= {m}; kk <<= 1) {{
    for (long j = kk >> 1; j > 0; j >>= 1) {{
      for (long t = lid; t < {half}; t += {t}) {{
{PAIR}      int asc = ((gbase + lo) & kk) == 0;
{ex}      }}
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
  }}
  for (long e = lid; e < {m}; e += {t}) {{
{write}  }}
",
        half = m / 2,
        ex = exchange("sk", "si", kt, "N"),
    );
    ctx.spec(
        "block",
        k,
        body,
        checked_product("the sort block count", &[ctx.lines, nb])?,
    )
}

/// One global compare distance `j` of merge size `kk`.
fn pass_global(ctx: &Ctx, kk: usize, j: usize) -> Result<KernelSpec, String> {
    let kt = ctx.key.math;
    let mut k = Kernel::new(ctx.instr);
    k.params.push(format!("__global {kt}* skeys"));
    k.args.push(KernelArg::Scratch(0));
    k.params.push("__global int* sidx".to_string());
    k.args.push(KernelArg::Scratch(1));
    k.long("N".to_string(), ctx.n as i64);
    k.long("P".to_string(), ctx.p as i64);
    k.long("kk".to_string(), kk as i64);
    k.long("j".to_string(), j as i64);
    let half = ctx.p / 2;
    // One work item per pair; the group size does not matter, so the launch
    // rounds up and the guard drops the extra items.
    let body = format!(
        "  long g = (long)get_global_id(0);
  long line = g / {half};
  long t = g % {half};
  __global {kt}* key = skeys + line * P;
  __global int* idx = sidx + line * P;
{PAIR_G}  int asc = (lo & kk) == 0;
  {{
{ex}  }}
",
        ex = exchange("key", "idx", kt, "N"),
        PAIR_G = PAIR.replace("      ", "  "),
    );
    let items = checked_product("the sort pair count", &[ctx.lines, half])?;
    ctx.spec_items("global", k, body, items)
}

/// The distances below `M` of merge size `kk`, in local memory.
fn pass_merge(ctx: &Ctx, kk: usize) -> Result<KernelSpec, String> {
    let (m, t, kt) = (ctx.m, ctx.t, ctx.key.math);
    let nb = ctx.p / m;
    let mut k = Kernel::new(ctx.instr);
    k.params.push(format!("__global {kt}* skeys"));
    k.args.push(KernelArg::Scratch(0));
    k.params.push("__global int* sidx".to_string());
    k.args.push(KernelArg::Scratch(1));
    k.long("N".to_string(), ctx.n as i64);
    k.long("P".to_string(), ctx.p as i64);
    k.long("kk".to_string(), kk as i64);
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long grp = (long)get_group_id(0);
  long line = grp / {nb};
  long blk = grp % {nb};
  long gbase = blk * {m};
  __global {kt}* gk = skeys + line * P + gbase;
  __global int* gi = sidx + line * P + gbase;
  __local {kt} sk[{m}];
  __local int si[{m}];
  for (long e = lid; e < {m}; e += {t}) {{ sk[e] = gk[e]; si[e] = gi[e]; }}
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long j = {m} >> 1; j > 0; j >>= 1) {{
    for (long t = lid; t < {half}; t += {t}) {{
{PAIR}      int asc = ((gbase + lo) & kk) == 0;
{ex}      }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  for (long e = lid; e < {m}; e += {t}) {{ gk[e] = sk[e]; gi[e] = si[e]; }}
",
        half = m / 2,
        ex = exchange("sk", "si", kt, "N"),
    );
    ctx.spec(
        "merge",
        k,
        body,
        checked_product("the sort block count", &[ctx.lines, nb])?,
    )
}

/// Last pass: the first `N` slots of each line go to the output.
fn pass_out(ctx: &Ctx) -> Result<KernelSpec, String> {
    let kt = ctx.key.math;
    let mut k = Kernel::new(ctx.instr);
    k.output()?;
    k.params.push(format!("__global const {kt}* skeys"));
    k.args.push(KernelArg::Scratch(0));
    k.params.push("__global const int* sidx".to_string());
    k.args.push(KernelArg::Scratch(1));
    geometry(ctx, &mut k);
    k.long("P".to_string(), ctx.p as i64);
    let body = format!(
        "  long g = (long)get_global_id(0);
  long line = g / N;
  long kpos = g % N;
{LINE}  out[base + kpos * inner] = {r};
",
        r = ctx.result("skeys[line * P + kpos]", "sidx[line * P + kpos]"),
    );
    let items = checked_product("the sort output size", &[ctx.lines, ctx.n])?;
    ctx.spec_items("out", k, body, items)
}

pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    expect_operands(instr, 1)?;
    let input = &instr.operands[0];
    let rank = input.shape.len();
    index::check_rank(&instr.op, rank)?;
    let desc = match instr.attrs.get(1) {
        Some(Attr::Atom(d)) if d == "desc" => true,
        Some(Attr::Atom(d)) if d == "asc" => false,
        _ => return Err(format!("{}: direction must be asc or desc", instr.op)),
    };
    let axis_attr = ints(&instr.attrs[..1], "sort axis")?[0];
    if instr.out_shape != input.shape {
        return Err(format!(
            "{}: the output shape must equal the input shape",
            instr.op
        ));
    }
    // A scalar is one line of one element.
    let (n, inner, outer) = if rank == 0 {
        (1, 1, 1)
    } else {
        let axis = usize::try_from(axis_attr)
            .ok()
            .filter(|&a| a < rank)
            .ok_or_else(|| format!("{}: axis {} is out of range", instr.op, axis_attr))?;
        (
            input.shape[axis],
            checked_product("the sort inner size", &input.shape[axis + 1..])?,
            checked_product("the sort outer size", &input.shape[..axis])?,
        )
    };
    let lines = checked_product("the sort line count", &[outer, inner])?;
    let in_info = type_info(input.dtype)?;
    if in_info.is_complex() {
        return Err(format!("{} does not accept complex values", instr.op));
    }
    let out_info = type_info(instr.out)?;
    let key = math_of(input.dtype)?;
    let arg = instr.op == "argsort";
    let p = padded(n)?;
    if p > MAX_PADDED {
        return Err(format!(
            "{}: an axis of {} elements is too long: the sort indexes with 32 bits",
            instr.op, n
        ));
    }
    let cap = local_capacity(limits, math_bytes(&key));
    if cap < 2 {
        return Err(format!(
            "{}: the device has too little local memory to sort",
            instr.op
        ));
    }
    let m = p.min(cap).max(2);
    let p = p.max(2);
    let items = limits.group_size(1);
    let head = format!(
        "{}{}{}",
        pragmas(&instr.dtypes(), caps)?,
        bf16_helpers(&[&in_info, &out_info]),
        before_fn(&key, desc)
    );
    let ctx = Ctx {
        instr,
        in_info,
        out_info,
        key,
        head,
        m,
        t: items.min(m / 2).max(1),
        p,
        n,
        inner,
        lines,
        arg,
    };
    if p <= m {
        return Ok(Plan::single(pass_block(&ctx)?));
    }
    let mut passes = vec![pass_block(&ctx)?];
    let mut kk = 2 * m;
    while kk <= p {
        let mut j = kk / 2;
        while j >= m {
            passes.push(pass_global(&ctx, kk, j)?);
            j /= 2;
        }
        passes.push(pass_merge(&ctx, kk)?);
        kk *= 2;
    }
    passes.push(pass_out(&ctx)?);
    let scratch = vec![
        checked_product("the sort key buffer", &[ctx.lines, p, math_bytes(&ctx.key)])?,
        checked_product("the sort index buffer", &[ctx.lines, p, 4])?,
    ];
    Ok(Plan { passes, scratch })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{self, plan};
    use super::super::types::Kind;
    use super::super::{Attr, DType};
    use super::*;

    const S32: DType = DType::new(Kind::S, 32);
    const F32: DType = DType::new(Kind::F, 32);
    const F64: DType = DType::new(Kind::F, 64);
    const U8: DType = DType::new(Kind::U, 8);
    const C64: DType = DType::new(Kind::C, 64);

    fn instr(op: &str, dtype: DType, shape: &[usize], axis: i64, dir: &str) -> Instr {
        let out = if op == "argsort" { S32 } else { dtype };
        test_support::instr(
            op,
            vec![test_support::buf(dtype, shape)],
            out,
            shape,
            vec![Attr::Int(axis), Attr::Atom(dir.to_string())],
        )
    }

    #[test]
    fn local_capacity_follows_local_memory_and_key_width() {
        let apple = Limits::apple();
        // 4-byte keys and 4-byte indices: 8 bytes each, 4096 fit in 32 KiB.
        assert_eq!(local_capacity(&apple, 4), 4096);
        assert_eq!(local_capacity(&apple, 8), 2048);
        assert_eq!(local_capacity(&apple, 1), 4096);
        let small = Limits::new(64, 1024);
        assert_eq!(local_capacity(&small, 4), 128);
    }

    #[test]
    fn lines_pad_to_a_power_of_two() {
        assert_eq!(padded(0).unwrap(), 1);
        assert_eq!(padded(1).unwrap(), 1);
        assert_eq!(padded(63).unwrap(), 64);
        assert_eq!(padded(65).unwrap(), 128);
        assert_eq!(padded(4097).unwrap(), 8192);
        assert_eq!(padded(9000).unwrap(), 16384);
        assert!(padded(usize::MAX).unwrap_err().contains("too long"));
    }

    #[test]
    fn a_line_that_fits_local_memory_sorts_in_one_launch() {
        for n in [1, 2, 63, 65, 4096] {
            let p = plan(&instr("sort", F32, &[3, n], 1, "asc"));
            assert_eq!((p.passes.len(), p.scratch.len()), (1, 0), "n = {n}");
        }
    }

    #[test]
    fn longer_lines_take_the_passes_pass_count_names() {
        let m = local_capacity(&Limits::apple(), 4);
        for n in [4097, 8192, 9000, 1 << 16, 1 << 20] {
            let p = plan(&instr("argsort", F32, &[n], 0, "desc"));
            assert_eq!(p.passes.len(), pass_count(n, m), "n = {n}");
            // Keys and indices for the padded line.
            let padded = padded(n).unwrap();
            assert_eq!(p.scratch, vec![padded * 4, padded * 4], "n = {n}");
        }
        // 8192 elements: block sort, one global exchange, one merge, copy out.
        assert_eq!(pass_count(8192, 4096), 4);
        assert_eq!(pass_count(4096, 4096), 1);
    }

    #[test]
    fn wider_keys_lower_the_capacity_and_add_passes() {
        let a = plan(&instr("sort", F32, &[4096], 0, "asc"));
        let b = plan(&instr("sort", F64, &[4096], 0, "asc"));
        assert_eq!(a.passes.len(), 1);
        assert!(b.passes.len() > 1);
    }

    #[test]
    fn the_order_is_total_and_stable() {
        let f = plan(&instr("argsort", F32, &[100], 0, "asc"));
        let src = &f.passes[0].source;
        // A tie on the key falls back to the index, so equal keys keep their order.
        assert!(src.contains("return ia < ib;"), "{src}");
        assert!(src.contains("isnan"));
        // Integers have no NaN.
        let i = plan(&instr("argsort", U8, &[100], 0, "asc"));
        assert!(!i.passes[0].source.contains("isnan"));
        // The direction is part of the source.
        let desc = plan(&instr("argsort", F32, &[100], 0, "desc"));
        assert_ne!(f.passes[0].source, desc.passes[0].source);
        // No atomics anywhere.
        let big = plan(&instr("sort", S32, &[1 << 16], 0, "asc"));
        assert!(big.passes.iter().all(|p| !p.source.contains("atomic")));
    }

    #[test]
    fn complex_values_a_bad_direction_and_a_bad_axis_are_errors() {
        let caps = Caps::all();
        let lim = Limits::apple();
        assert!(generate(&instr("sort", C64, &[8], 0, "asc"), &caps, &lim).is_err());
        let err = generate(&instr("sort", F32, &[8], 0, "up"), &caps, &lim).unwrap_err();
        assert!(err.contains("asc or desc"), "{err}");
        let err = generate(&instr("sort", F32, &[8], 3, "asc"), &caps, &lim).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn an_empty_tensor_has_a_positive_global_size() {
        // The executor never launches this, and the spec stays well formed.
        for shape in [[0usize].as_slice(), &[3, 0], &[0, 9000]] {
            let p = plan(&instr("sort", F32, shape, shape.len() as i64 - 1, "asc"));
            assert!(
                p.passes.iter().all(|s| s.global.iter().all(|&g| g > 0)),
                "{shape:?}"
            );
            // The one item the launch would run must find nothing to do.
            for s in &p.passes {
                assert!(
                    test_support::starts_guarded(&s.source),
                    "{shape:?} {}",
                    s.entry
                );
            }
        }
    }

    #[test]
    fn an_axis_that_a_32_bit_index_cannot_reach_is_an_error() {
        let err = generate(
            &instr("argsort", F32, &[1usize << 31], 0, "asc"),
            &Caps::all(),
            &Limits::apple(),
        )
        .unwrap_err();
        assert!(
            err.contains("32 bits") && err.contains("2147483648"),
            "{err}"
        );
        // An axis too long to pad at all is an error too.
        assert!(padded(usize::MAX).is_err());
    }

    #[test]
    fn sizes_that_overflow_are_errors() {
        let big = 1usize << 40;
        let err = generate(
            &instr("sort", F32, &[big, big, 2], 2, "asc"),
            &Caps::all(),
            &Limits::apple(),
        )
        .unwrap_err();
        assert!(err.contains("is too large"), "{err}");
    }
}
