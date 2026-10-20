//! Cumulative ops: `cumulative_sum`, `cumulative_product`, `cumulative_min` and
//! `cumulative_max` along one axis, forward or reversed.
//!
//! A line is one run along the axis. Kernels index it with the axis size `N` and
//! the inner stride, so any axis works with no transpose. A work-efficient
//! (Blelloch) scan runs in local memory over one block of `WG` elements.
//!
//! - A line of at most `WG` elements takes one launch.
//! - A longer line splits into `B = ceil(N / WG)` blocks. Pass 1 scans each block
//!   and writes its total. Pass 2 scans each line's totals, one group per line,
//!   `WG` totals at a time with a running carry. Pass 3 adds each block's offset
//!   to its elements.
//!
//! The accumulator has the output type, so integers wrap as the reference does.
//! No atomics. A float scan adds in a different order from a left fold, so
//! results differ in the last bits.

use super::index;
use super::movement::{ints, pragmas, Kernel};
use super::reduce::{combiner, Canon, CombKind, Combiner};
use super::types::{bf16_helpers, load_as, math_bytes, math_of, store_as, type_info, TypeInfo};
use super::{checked_product, entry_name, Caps, Instr, KernelArg, KernelSpec, Limits, Plan};

const OPS: [&str; 4] = [
    "cumulative_sum",
    "cumulative_product",
    "cumulative_min",
    "cumulative_max",
];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

fn blocks(n: usize, wg: usize) -> usize {
    n.div_ceil(wg).max(1)
}

struct Ctx<'a> {
    instr: &'a Instr,
    in_info: TypeInfo,
    out_info: TypeInfo,
    acc: TypeInfo,
    comb: Combiner,
    head: String,
    wg: usize,
    n: usize,
    inner: usize,
    lines: usize,
}

impl Ctx<'_> {
    fn entry(&self, pass: &str) -> String {
        entry_name(
            "sc",
            &[
                &self.comb.tag,
                &self.in_info.dtype.name(),
                &self.out_info.dtype.name(),
                pass,
            ],
        )
    }

    fn id(&self) -> &str {
        self.comb.identity.as_deref().unwrap_or("0")
    }

    /// A pass of `lines * blocks_per_line` groups of `wg` items.
    fn spec(
        &self,
        pass: &str,
        k: Kernel,
        body: String,
        groups: usize,
    ) -> Result<KernelSpec, String> {
        let global = checked_product("the scan launch size", &[groups, self.wg])?;
        Ok(k.into_spec(
            &self.head,
            self.entry(pass),
            &body,
            vec![global],
            vec![self.wg],
        ))
    }

    fn geometry(&self, k: &mut Kernel, b: usize, rev: bool) {
        k.long("N".to_string(), self.n as i64);
        k.long("inner".to_string(), self.inner as i64);
        k.long("B".to_string(), b as i64);
        k.long("rev".to_string(), rev as i64);
    }
}

/// Source that scans `sh[0..WG]` into an exclusive prefix; the sum of the
/// whole array goes to `total`. Needs `lid`, `sh`, `total`, and `ident`.
fn blelloch(wg: usize, id: &str) -> String {
    format!(
        "  barrier(CLK_LOCAL_MEM_FENCE);
  for (long d = 1; d < {wg}; d <<= 1) {{
    long i = (lid + 1) * 2 * d - 1;
    if (i < {wg}) sh[i] = comb(sh[i], sh[i - d]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  if (lid == 0) {{ total = sh[{wg} - 1]; sh[{wg} - 1] = {id}; }}
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long d = {wg} / 2; d >= 1; d >>= 1) {{
    long i = (lid + 1) * 2 * d - 1;
    if (i < {wg}) {{ ACC t = sh[i - d]; sh[i - d] = sh[i]; sh[i] = comb(t, sh[i]); }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
"
    )
}

/// Source: `base` and the position of logical element `k`.
const LINE: &str = "  long outer = line / inner;
  long inn = line % inner;
  long base = outer * N * inner + inn;
";

fn pass_block(ctx: &Ctx, b: usize, rev: bool, totals: bool) -> Result<KernelSpec, String> {
    let (wg, t) = (ctx.wg, ctx.acc.math);
    let mut k = Kernel::new(ctx.instr);
    k.operand(0)?;
    k.output()?;
    if totals {
        k.params.push(format!("__global {t}* totals"));
        k.args.push(KernelArg::Scratch(0));
    }
    ctx.geometry(&mut k, b, rev);
    let load = load_as(&ctx.in_info, &k.read(0, "pos"), &ctx.acc)?;
    let write_total = if totals {
        "  if (lid == 0) totals[grp] = total;\n"
    } else {
        ""
    };
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long grp = (long)get_group_id(0);
  long line = grp / B;
  long blk = grp % B;
{LINE}  long k = blk * {wg} + lid;
  {t} v = {id};
  long pos = 0;
  if (k < N) {{ pos = base + (rev ? N - 1 - k : k) * inner; v = {load}; }}
  __local {t} sh[{wg}];
  __local {t} total;
  sh[lid] = v;
{scan}  {t} r = comb(v, sh[lid]);
  if (k < N) out[pos] = {store};
{write_total}",
        id = ctx.id(),
        scan = blelloch(wg, ctx.id()).replace("ACC", t),
        store = store_as(&ctx.out_info, "r"),
    );
    ctx.spec(
        if totals { "p1t" } else { "p1" },
        k,
        body,
        checked_product("the scan block count", &[ctx.lines, b])?,
    )
}

fn pass_totals(ctx: &Ctx, b: usize) -> Result<KernelSpec, String> {
    let (wg, t) = (ctx.wg, ctx.acc.math);
    let mut k = Kernel::new(ctx.instr);
    k.params.push(format!("__global {t}* totals"));
    k.args.push(KernelArg::Scratch(0));
    k.long("B".to_string(), b as i64);
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long line = (long)get_group_id(0);
  __local {t} sh[{wg}];
  __local {t} total;
  __local {t} carry;
  if (lid == 0) carry = {id};
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long t0 = 0; t0 < B; t0 += {wg}) {{
    long idx = t0 + lid;
    sh[lid] = idx < B ? totals[line * B + idx] : {id};
{scan}    if (idx < B) totals[line * B + idx] = comb(sh[lid], carry);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) carry = comb(total, carry);
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
",
        id = ctx.id(),
        scan = blelloch(wg, ctx.id()).replace("ACC", t),
    );
    ctx.spec("p2", k, body, ctx.lines)
}

fn pass_add(ctx: &Ctx, b: usize, rev: bool) -> Result<KernelSpec, String> {
    let (wg, t) = (ctx.wg, ctx.acc.math);
    let mut k = Kernel::new(ctx.instr);
    k.output()?;
    k.params.push(format!("__global const {t}* offs"));
    k.args.push(KernelArg::Scratch(0));
    ctx.geometry(&mut k, b, rev);
    let cur = load_as(&ctx.out_info, "out[pos]", &ctx.acc)?;
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long grp = (long)get_group_id(0);
  long line = grp / B;
  long blk = grp % B;
  if (blk == 0) return;
{LINE}  long k = blk * {wg} + lid;
  if (k >= N) return;
  long pos = base + (rev ? N - 1 - k : k) * inner;
  {t} r = comb({cur}, offs[grp]);
  out[pos] = {store};
",
        store = store_as(&ctx.out_info, "r"),
    );
    ctx.spec(
        "p3",
        k,
        body,
        checked_product("the scan block count", &[ctx.lines, b])?,
    )
}

pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    if instr.operands.len() != 1 {
        return Err(format!(
            "{} takes 1 operand, got {}",
            instr.op,
            instr.operands.len()
        ));
    }
    let input = &instr.operands[0];
    index::check_rank(&instr.op, input.shape.len())?;
    let a = ints(&instr.attrs, "scan")?;
    if a.len() != 2 {
        return Err(format!("{}: attrs must be [axis, reverse]", instr.op));
    }
    let axis = usize::try_from(a[0])
        .ok()
        .filter(|&x| x < input.shape.len())
        .ok_or_else(|| format!("{}: axis {} is out of range", instr.op, a[0]))?;
    let rev = a[1] != 0;
    if input.shape != instr.out_shape {
        return Err(format!(
            "{}: the output shape must equal the input shape",
            instr.op
        ));
    }
    let n = input.shape[axis];
    let inner = checked_product("the scan inner size", &input.shape[axis + 1..])?;
    let outer = checked_product("the scan outer size", &input.shape[..axis])?;
    let lines = checked_product("the scan line count", &[outer, inner])?;
    let acc_dtype = math_of(instr.out)?.dtype;
    let canon =
        Canon::from_op(&instr.op).ok_or_else(|| format!("{}: not a cumulative op", instr.op))?;
    let comb = combiner(&CombKind::Canon(canon), acc_dtype)?;
    let acc = comb.acc;
    let in_info = type_info(input.dtype)?;
    let out_info = type_info(instr.out)?;
    let pragma_text = pragmas(&instr.dtypes(), caps)?;
    // The scan array, the total and the carry.
    let wg = limits.group_size(math_bytes(&acc) + 1);
    let head = format!(
        "{pragma_text}{bf}{code}",
        bf = bf16_helpers(&[&in_info, &out_info]),
        code = comb.code.render()
    );
    let ctx = Ctx {
        instr,
        in_info,
        out_info,
        acc,
        comb,
        head,
        wg,
        n,
        inner: inner.max(1),
        lines,
    };
    let b = blocks(n, wg);
    if b == 1 {
        return Ok(Plan::single(pass_block(&ctx, 1, rev, false)?));
    }
    Ok(Plan {
        passes: vec![
            pass_block(&ctx, b, rev, true)?,
            pass_totals(&ctx, b)?,
            pass_add(&ctx, b, rev)?,
        ],
        scratch: vec![checked_product(
            "the scan totals buffer",
            &[ctx.lines, b, math_bytes(&acc)],
        )?],
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::plan;
    use super::super::types::Kind;
    use super::super::{Attr, DType, Operand};
    use super::*;

    const S32: DType = DType::new(Kind::S, 32);
    const F32: DType = DType::new(Kind::F, 32);
    const C64: DType = DType::new(Kind::C, 64);

    fn instr(op: &str, dtype: DType, shape: &[usize], axis: i64, reverse: i64) -> Instr {
        Instr {
            op: op.to_string(),
            operands: vec![Operand {
                dtype,
                shape: shape.to_vec(),
                constant: None,
            }],
            out: dtype,
            out_shape: shape.to_vec(),
            attrs: vec![Attr::Int(axis), Attr::Int(reverse)],
        }
    }

    #[test]
    fn block_counts_round_up() {
        assert_eq!(blocks(0, 256), 1);
        assert_eq!(blocks(1, 256), 1);
        assert_eq!(blocks(256, 256), 1);
        assert_eq!(blocks(257, 256), 2);
        assert_eq!(blocks(4097, 256), 17);
        assert_eq!(blocks(1 << 20, 256), 4096);
    }

    #[test]
    fn a_line_of_one_group_takes_one_launch() {
        let p = plan(&instr("cumulative_sum", F32, &[3, 256], 1, 0));
        assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
        assert_eq!(p.passes[0].local, vec![256]);
        // 256 groups' worth of lines: still one launch each.
        let p = plan(&instr("cumulative_max", S32, &[1000, 7], 1, 1));
        assert_eq!(p.passes.len(), 1);
    }

    #[test]
    fn a_longer_line_scans_blocks_then_totals_then_adds_offsets() {
        let p = plan(&instr("cumulative_sum", F32, &[3, 257], 1, 0));
        assert_eq!(p.passes.len(), 3);
        // One total per block per line.
        assert_eq!(p.scratch, vec![3 * 2 * 4]);

        let p = plan(&instr("cumulative_sum", S32, &[1 << 20], 0, 0));
        assert_eq!(p.passes.len(), 3);
        assert_eq!(p.scratch, vec![4096 * 4]);
        // The totals pass is one group per line.
        assert_eq!(p.passes[1].global, vec![256]);
    }

    #[test]
    fn reverse_is_a_kernel_argument_and_nothing_uses_atomics() {
        let fwd = plan(&instr("cumulative_sum", F32, &[2, 600], 1, 0));
        let rev = plan(&instr("cumulative_sum", F32, &[2, 600], 1, 1));
        // One source serves both directions; the flag is a scalar argument.
        assert_eq!(fwd.passes[0].source, rev.passes[0].source);
        assert_ne!(fwd.passes[0].args, rev.passes[0].args);
        for pass in fwd.passes.iter().chain(rev.passes.iter()) {
            assert!(!pass.source.contains("atomic"));
        }
    }

    #[test]
    fn the_group_size_obeys_the_device_limits() {
        let small = Limits::new(32, 512);
        let p = generate(
            &instr("cumulative_sum", F32, &[600], 0, 0),
            &Caps::all(),
            &small,
        )
        .unwrap();
        for s in &p.passes {
            assert!(s.local.first().copied().unwrap_or(1) <= 32);
        }
        assert!(p.passes.len() > 1);
    }

    #[test]
    fn min_and_max_reject_complex_and_a_bad_axis_is_an_error() {
        let caps = Caps::all();
        let lim = Limits::apple();
        assert!(generate(&instr("cumulative_max", C64, &[8], 0, 0), &caps, &lim).is_err());
        assert!(generate(&instr("cumulative_sum", C64, &[8], 0, 0), &caps, &lim).is_ok());
        let err = generate(&instr("cumulative_sum", F32, &[8], 1, 0), &caps, &lim).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn an_empty_tensor_has_a_positive_global_size() {
        // The executor never launches this, and the spec stays well formed.
        for shape in [[0usize].as_slice(), &[3, 0], &[0, 300]] {
            let p = plan(&instr(
                "cumulative_sum",
                F32,
                shape,
                shape.len() as i64 - 1,
                0,
            ));
            assert!(
                p.passes.iter().all(|s| s.global.iter().all(|&g| g > 0)),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn sizes_that_overflow_are_errors() {
        let big = 1usize << 40;
        let err = generate(
            &instr("cumulative_sum", F32, &[big, big, 2], 2, 0),
            &Caps::all(),
            &Limits::apple(),
        )
        .unwrap_err();
        assert!(err.contains("is too large"), "{err}");
    }
}
