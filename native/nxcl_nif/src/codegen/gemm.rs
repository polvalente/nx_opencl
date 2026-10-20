//! `batched_gemm`: `C[b] = A[b] x B[b]` for row-major `A: [B, M, K]` and
//! `B: [B, K, N]`. Every `dot` and the conv expansion (`conv.rs`, groups as the
//! batch) use it.
//!
//! The main kernel is register tiled. A work group computes a `BM x BN` block of
//! one batch entry; each thread owns a `TM x TN` block, strided by the group
//! shape so neighbouring threads touch neighbouring columns. The group walks `K`
//! in steps of `BK`, stages the `A` and `B` tiles in local memory, and
//! bounds-checks every load and store. Out-of-range tile cells load as zero.
//!
//! The tile comes from a table keyed on device limits (work group size, local
//! memory), not a device name; a tile that does not fit is skipped for the next
//! smaller. The tile is in the build options, so it is part of the cache key.
//!
//! `N == 1` (a work group folds each row of `A` against the vector with a local
//! tree) and `M == 1` (one thread per output column) have their own kernels.
//! `K == 0` or an empty output writes zeros and runs no GEMM kernel.
//!
//! Floats accumulate in the math type (`f32` for `f16` and `bf16`). Integers wrap
//! in an unsigned type of at least 32 bits, so `s32` stays `s32`. Complex values
//! use the helpers of `complex.rs`.

use super::complex;
use super::movement::pragmas;
use super::types::{
    bf16_helpers, c_type_bytes, load_as, math_of, store_as, type_info, Class, DType, TypeInfo,
};
use super::{
    checked_product, expect_operands, long_arg, Attr, Caps, Instr, KernelArg, KernelSpec, Limits,
    Plan,
};

pub fn handles(op: &str) -> bool {
    op == "batched_gemm"
}

/// One work-group tile. A group has `bn / tn` by `bm / tm` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub tm: usize,
    pub tn: usize,
}

impl Tile {
    const fn new(bm: usize, bn: usize, bk: usize, tm: usize, tn: usize) -> Self {
        Self { bm, bn, bk, tm, tn }
    }

    pub fn threads(&self) -> usize {
        (self.bm / self.tm) * (self.bn / self.tn)
    }

    /// Bytes of local memory the two staged tiles take for `elem` byte values.
    pub fn local_bytes(&self, elem: usize) -> usize {
        (self.bm + self.bn) * self.bk * elem
    }

    fn valid(&self) -> bool {
        self.bm > 0
            && self.bn > 0
            && self.bk > 0
            && self.tm > 0
            && self.tn > 0
            && self.bm.is_multiple_of(self.tm)
            && self.bn.is_multiple_of(self.tn)
    }

    /// True when a group of this tile runs on a device with `limits`: the
    /// group is `bn / tn` by `bm / tm` items, and each side must fit its own
    /// per-dimension limit as well as the total.
    pub fn fits(&self, limits: &Limits, elem: usize) -> bool {
        self.valid()
            && self.threads() <= limits.max_wg
            && self.bn / self.tn <= limits.max_item_sizes[0]
            && self.bm / self.tm <= limits.max_item_sizes[1]
            && self.local_bytes(elem) <= limits.local_mem
    }

    /// The build options that carry the tile into the kernel source.
    pub fn options(&self) -> String {
        format!(
            "-DBM={} -DBN={} -DBK={} -DTM={} -DTN={}",
            self.bm, self.bn, self.bk, self.tm, self.tn
        )
    }
}

/// Tiles from largest to smallest. The last one needs one thread and two
/// values of local memory.
const TABLE: [Tile; 8] = [
    Tile::new(128, 128, 8, 8, 8),
    Tile::new(64, 64, 16, 4, 4),
    Tile::new(64, 32, 16, 4, 2),
    Tile::new(32, 32, 16, 2, 2),
    Tile::new(16, 16, 8, 2, 2),
    Tile::new(8, 8, 8, 1, 1),
    Tile::new(4, 4, 4, 1, 1),
    Tile::new(1, 1, 1, 1, 1),
];

/// A device that runs 1024 or more work items per group runs them one after
/// another (a CPU), so large groups and large private tiles only add loop
/// overhead there. It takes these tiles instead.
const WIDE_GROUP_TABLE: [Tile; 5] = [
    Tile::new(32, 32, 16, 2, 2),
    Tile::new(16, 16, 8, 2, 2),
    Tile::new(8, 8, 8, 1, 1),
    Tile::new(4, 4, 4, 1, 1),
    Tile::new(1, 1, 1, 1, 1),
];
const WIDE_GROUP: usize = 1024;

/// The tile should leave at least this many work groups (batch included), so
/// the device has groups to spread over its cores.
const TARGET_GROUPS: usize = 96;

/// Work groups a tile makes for `batch` products of `m x n`.
fn groups(t: &Tile, batch: usize, m: usize, n: usize) -> usize {
    batch
        .max(1)
        .saturating_mul(m.div_ceil(t.bm))
        .saturating_mul(n.div_ceil(t.bn))
}

/// The tile for `batch` products of `m x n` values of `elem` bytes: the first
/// table tile that fits the limits, is not much larger than the problem, and leaves at least
/// `TARGET_GROUPS` groups; if none leaves that many, the fitting tile with the
/// most groups. Errors only when even the smallest tile does not fit.
fn select_tile(
    limits: &Limits,
    elem: usize,
    (batch, m, n): (usize, usize, usize),
) -> Result<Tile, String> {
    let table: &[Tile] = if limits.max_wg >= WIDE_GROUP {
        &WIDE_GROUP_TABLE
    } else {
        &TABLE
    };
    let fitting: Vec<&Tile> = table.iter().filter(|t| t.fits(limits, elem)).collect();
    let last = fitting.len().saturating_sub(1);
    let roomy = |(i, t): &(usize, &&Tile)| {
        let small_problem =
            t.bm >= m.max(1).saturating_mul(4) || t.bn >= n.max(1).saturating_mul(4);
        *i == last || !small_problem
    };
    let candidates: Vec<&Tile> = fitting
        .iter()
        .enumerate()
        .filter(roomy)
        .map(|(_, t)| *t)
        .collect();
    candidates
        .iter()
        .find(|t| groups(t, batch, m, n) >= TARGET_GROUPS)
        .or_else(|| candidates.iter().max_by_key(|t| groups(t, batch, m, n)))
        .map(|t| **t)
        .ok_or_else(|| {
            format!(
                "no GEMM tile fits {} threads and {} bytes of local memory",
                limits.max_wg, limits.local_mem
            )
        })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Gemm {
    pub dtype: DType,
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub(super) struct Binding {
    pub a: KernelArg,
    pub b: KernelArg,
    pub out: KernelArg,
}

struct Code {
    info: TypeInfo,
    t: &'static str,
    head: String,
    load: String,
    mac: String,
    add: String,
    store: String,
}

fn code(dtype: DType) -> Result<Code, String> {
    let info = type_info(dtype)?;
    let acc = math_of(dtype)?;
    let (t, load, mac, add) = match info.class {
        Class::Complex => {
            let (mul, add) = (
                complex::fn_name(&info, "mul"),
                complex::fn_name(&info, "add"),
            );
            (
                acc.math,
                "x".to_string(),
                format!("{add}(acc, {mul}(a, b))"),
                format!("{add}(a, b)"),
            )
        }
        Class::Signed | Class::Unsigned => (
            info.wide,
            format!("({})(x)", info.wide),
            "acc + a * b".to_string(),
            "a + b".to_string(),
        ),
        Class::Float => (
            acc.math,
            load_as(&info, "x", &acc)?,
            "acc + a * b".to_string(),
            "a + b".to_string(),
        ),
    };
    let head = format!(
        "{}{}",
        bf16_helpers(&[&info]),
        complex::preamble(&info, &["add", "mul"])
    );
    let store = store_as(&info, "v");
    Ok(Code {
        info,
        t,
        head,
        load,
        mac,
        add,
        store,
    })
}

fn helpers(c: &Code) -> String {
    format!(
        "{head}\
static inline {t} gemm_ld({s} x) {{ return {load}; }}\n\
static inline {t} gemm_mac({t} acc, {t} a, {t} b) {{ return {mac}; }}\n\
static inline {t} gemm_add({t} a, {t} b) {{ return {add}; }}\n\
static inline {s} gemm_st({t} v) {{ return {store}; }}\n",
        head = c.head,
        t = c.t,
        s = c.info.storage,
        load = c.load,
        mac = c.mac,
        add = c.add,
        store = c.store,
    )
}

const TILED: &str = r#"
#define NTX (BN / TN)
#define NTY (BM / TM)
__kernel void $ENTRY(__global const $S* A, __global const $S* B, __global $S* C,
                     const long M, const long N, const long K) {
  __local $T As[BK][BM];
  __local $T Bs[BK][BN];
  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int tid = ty * NTX + tx;
  const long batch = get_group_id(2);
  const long row0 = (long)get_group_id(1) * BM;
  const long col0 = (long)get_group_id(0) * BN;
  __global const $S* Ab = A + batch * M * K;
  __global const $S* Bb = B + batch * K * N;
  __global $S* Cb = C + batch * M * N;
  $T acc[TM][TN];
  for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++) acc[i][j] = ($T)(0);
  for (long k0 = 0; k0 < K; k0 += BK) {
    for (int e = tid; e < BM * BK; e += NTX * NTY) {
      int r = e / BK, kk = e % BK;
      long gr = row0 + r, gk = k0 + kk;
      As[kk][r] = (gr < M && gk < K) ? gemm_ld(Ab[gr * K + gk]) : ($T)(0);
    }
    for (int e = tid; e < BK * BN; e += NTX * NTY) {
      int kk = e / BN, c = e % BN;
      long gk = k0 + kk, gc = col0 + c;
      Bs[kk][c] = (gk < K && gc < N) ? gemm_ld(Bb[gk * N + gc]) : ($T)(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < BK; kk++) {
      $T a[TM];
      $T b[TN];
      for (int i = 0; i < TM; i++) a[i] = As[kk][ty + i * NTY];
      for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx + j * NTX];
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = gemm_mac(acc[i][j], a[i], b[j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < TM; i++) {
    long gr = row0 + ty + i * NTY;
    for (int j = 0; j < TN; j++) {
      long gc = col0 + tx + j * NTX;
      if (gr < M && gc < N) Cb[gr * N + gc] = gemm_st(acc[i][j]);
    }
  }
}
"#;

/// `N == 1`: one work group per output row folds `A[row] . x` with a local tree.
const MATVEC: &str = r#"
__kernel void $ENTRY(__global const $S* A, __global const $S* B, __global $S* C,
                     const long M, const long N, const long K) {
  __local $T part[WG];
  const long m = get_group_id(0);
  const long b = get_group_id(1);
  const int t = get_local_id(0);
  __global const $S* a = A + (b * M + m) * K;
  __global const $S* x = B + b * K;
  $T acc = ($T)(0);
  for (long k = t; k < K; k += WG) acc = gemm_mac(acc, gemm_ld(a[k]), gemm_ld(x[k]));
  part[t] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG / 2; s > 0; s >>= 1) {
    if (t < s) part[t] = gemm_add(part[t], part[t + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) C[b * M + m] = gemm_st(part[0]);
}
"#;

/// `M == 1`: one thread per output column.
const VECMAT: &str = r#"
__kernel void $ENTRY(__global const $S* A, __global const $S* B, __global $S* C,
                     const long M, const long N, const long K) {
  const long n = get_global_id(0);
  const long b = get_global_id(1);
  if (n >= N) return;
  __global const $S* a = A + b * K;
  __global const $S* x = B + b * K * N + n;
  $T acc = ($T)(0);
  for (long k = 0; k < K; k++) acc = gemm_mac(acc, gemm_ld(a[k]), gemm_ld(x[k * N]));
  C[b * N + n] = gemm_st(acc);
}
"#;

const FILL: &str = r#"
__kernel void $ENTRY(__global $S* C, const long total) {
  const long i = get_global_id(0);
  if (i < total) C[i] = ($S)(0);
}
"#;

fn fill_template(tpl: &str, entry: &str, c: &Code) -> String {
    tpl.replace("$ENTRY", entry)
        .replace("$S", c.info.storage)
        .replace("$T", c.t)
}

fn size_arg(v: usize) -> Result<KernelArg, String> {
    Ok(KernelArg::Scalar(long_arg(v)?))
}

/// The kernel for one product, reading and writing the buffers in `bind`.
pub(super) fn kernel(
    g: &Gemm,
    bind: &Binding,
    caps: &Caps,
    limits: &Limits,
) -> Result<KernelSpec, String> {
    let c = code(g.dtype)?;
    let pragma = pragmas(&[g.dtype], caps)?;
    let name = g.dtype.name();
    let elem = c_type_bytes(c.t)?;
    let head = format!("{pragma}{}", helpers(&c));
    let sizes = [size_arg(g.m)?, size_arg(g.n)?, size_arg(g.k)?];
    let mut args = vec![bind.a.clone(), bind.b.clone(), bind.out.clone()];
    args.extend(sizes);

    let total = checked_product("the gemm output size", &[g.batch, g.m, g.n])?;
    let (source, entry, options, global, local) = if g.k == 0 || total == 0 {
        let entry = format!("gemm_zero_{name}");
        let args = vec![bind.out.clone(), size_arg(total)?];
        return Ok(KernelSpec {
            source: format!("{pragma}{}", fill_template(FILL, &entry, &c)),
            entry,
            build_options: String::new(),
            global: vec![total.max(1)],
            local: Vec::new(),
            args,
        });
    } else if g.n == 1 {
        let wg = limits
            .group_size(elem)
            .min(g.k.checked_next_power_of_two().unwrap_or(usize::MAX))
            .max(1);
        let entry = format!("gemm_matvec_{name}");
        (
            fill_template(MATVEC, &entry, &c),
            entry,
            format!("-DWG={wg}"),
            vec![
                checked_product("the gemm launch size", &[g.m, wg])?,
                g.batch,
            ],
            vec![wg, 1],
        )
    } else if g.m == 1 {
        let entry = format!("gemm_vecmat_{name}");
        (
            fill_template(VECMAT, &entry, &c),
            entry,
            String::new(),
            vec![g.n, g.batch],
            Vec::new(),
        )
    } else {
        let tile = select_tile(limits, elem, (g.batch, g.m, g.n))?;
        let entry = format!("gemm_tiled_{name}");
        (
            fill_template(TILED, &entry, &c),
            entry,
            tile.options(),
            vec![
                checked_product(
                    "the gemm launch width",
                    &[g.n.div_ceil(tile.bn), tile.bn / tile.tn],
                )?,
                checked_product(
                    "the gemm launch height",
                    &[g.m.div_ceil(tile.bm), tile.bm / tile.tm],
                )?,
                g.batch,
            ],
            vec![tile.bn / tile.tn, tile.bm / tile.tm, 1],
        )
    };
    Ok(KernelSpec {
        source: format!("{head}{source}"),
        entry,
        build_options: options,
        global,
        local,
        args,
    })
}

/// The kernel for a `batched_gemm` instruction: operands `[B,M,K]` and `[B,K,N]`,
/// output `[B,M,N]`, attrs `[B, M, N, K]`.
pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    expect_operands(instr, 2)?;
    let size = |i: usize, what: &str| -> Result<usize, String> {
        match instr.attrs.get(i) {
            Some(Attr::Int(v)) => usize::try_from(*v).ok(),
            _ => None,
        }
        .ok_or_else(|| format!("batched_gemm attr {} ({}) must be a size", i, what))
    };
    let g = Gemm {
        dtype: instr.out,
        batch: size(0, "batch")?,
        m: size(1, "m")?,
        n: size(2, "n")?,
        k: size(3, "k")?,
    };
    let (a, b) = (&instr.operands[0], &instr.operands[1]);
    if a.dtype != g.dtype || b.dtype != g.dtype {
        return Err(format!(
            "batched_gemm operands must have the output type {}",
            g.dtype
        ));
    }
    if a.constant.is_some() || b.constant.is_some() {
        return Err("batched_gemm operands must be buffers".to_string());
    }
    if a.shape != [g.batch, g.m, g.k]
        || b.shape != [g.batch, g.k, g.n]
        || instr.out_shape != [g.batch, g.m, g.n]
    {
        return Err(format!(
            "batched_gemm shapes {:?} x {:?} -> {:?} do not match [B,M,K] x [B,K,N] -> [B,M,N] for {:?}",
            a.shape,
            b.shape,
            instr.out_shape,
            [g.batch, g.m, g.n, g.k]
        ));
    }
    let bind = Binding {
        a: KernelArg::Operand(0),
        b: KernelArg::Operand(1),
        out: KernelArg::Output,
    };
    kernel(&g, &bind, caps, limits).map(Plan::single)
}

#[cfg(test)]
#[path = "tests/gemm.rs"]
mod tests;
