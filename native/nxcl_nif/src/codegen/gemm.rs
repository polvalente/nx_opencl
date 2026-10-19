//! `batched_gemm`: `C[b] = A[b] x B[b]` for row-major `A: [B, M, K]` and
//! `B: [B, K, N]`. Elixir turns every `dot` into this form; the conv expansion
//! in `conv.rs` calls it with groups as the batch axis.
//!
//! The main kernel is register tiled. A work group computes a `BM x BN` block
//! of one batch entry. Each thread owns a `TM x TN` block of it, strided by the
//! group shape so that neighbouring threads touch neighbouring columns. The
//! group walks `K` in steps of `BK`, stages the `A` and `B` tiles in local
//! memory, and bounds-checks every load and store, so `M` and `N` need not be
//! multiples of the tile. Out-of-range tile cells load as zero.
//!
//! The tile comes from a table that is keyed on device limits (work group size
//! and local memory), not on a device name. A tile that does not fit the limits
//! is skipped for the next smaller one. `NXCL_GEMM_TILE=bm,bn,bk,tm,tn`
//! replaces the table when the tile is valid and fits. The tile goes into the
//! build options, so it is part of the kernel cache key.
//!
//! Two shapes get their own kernels: `N == 1` (a work group folds each row of
//! `A` against the vector with a local tree) and `M == 1` (one thread per
//! output column). `K == 0`, or an empty output, writes zeros with a fill
//! kernel and runs no GEMM kernel.
//!
//! Accumulation: floats accumulate in the math type (`f32` for `f16` and
//! `bf16`); integers wrap in an unsigned type of at least 32 bits, so `s32`
//! stays `s32`; complex values use the helpers of `complex.rs`.

use super::complex;
use super::movement::pragmas;
use super::types::{
    bf16_helpers, c_type_bytes, load_as, math_of, store_as, type_info, Class, DType, TypeInfo,
};
use super::{checked_product, long_arg, Attr, Caps, Instr, KernelArg, KernelSpec, Limits, Plan};

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

/// Parses `bm,bn,bk,tm,tn`.
fn parse_tile(text: &str) -> Result<Tile, String> {
    let parts: Vec<usize> = text
        .split(',')
        .map(|p| p.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("NXCL_GEMM_TILE must be bm,bn,bk,tm,tn; got {:?}", text))?;
    if parts.len() != 5 {
        return Err(format!(
            "NXCL_GEMM_TILE must hold 5 numbers (bm,bn,bk,tm,tn); got {}",
            parts.len()
        ));
    }
    let tile = Tile::new(parts[0], parts[1], parts[2], parts[3], parts[4]);
    if !tile.valid() {
        return Err(format!(
            "NXCL_GEMM_TILE {:?}: sizes must be positive and bm, bn divisible by tm, tn",
            text
        ));
    }
    Ok(tile)
}

/// Work groups a tile makes for `batch` products of `m x n`.
fn groups(t: &Tile, batch: usize, m: usize, n: usize) -> usize {
    batch
        .max(1)
        .saturating_mul(m.div_ceil(t.bm))
        .saturating_mul(n.div_ceil(t.bn))
}

/// The tile for `batch` products of `m x n` values of `elem` bytes. `forced`
/// (the `NXCL_GEMM_TILE` value) wins when it fits. Else the first table tile
/// that fits the limits, is not much larger than the problem, and leaves at least
/// `TARGET_GROUPS` groups; if none leaves that many, the fitting tile with the
/// most groups. Errors only when even the smallest tile does not fit.
fn select_tile(
    limits: &Limits,
    elem: usize,
    (batch, m, n): (usize, usize, usize),
    forced: Option<Tile>,
) -> Result<Tile, String> {
    if let Some(t) = forced.filter(|t| t.fits(limits, elem)) {
        return Ok(t);
    }
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

/// The tile `NXCL_GEMM_TILE` asks for, if the variable is set. A value that
/// does not parse is an error, not a silent fall back to the table.
pub(super) fn forced_tile() -> Result<Option<Tile>, String> {
    tile_from(std::env::var("NXCL_GEMM_TILE"))
}

fn tile_from(value: Result<String, std::env::VarError>) -> Result<Option<Tile>, String> {
    match value {
        Ok(text) => parse_tile(&text).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("NXCL_GEMM_TILE is not valid text".to_string())
        }
    }
}

/// The sizes of one product.
#[derive(Clone, Copy, Debug)]
pub(super) struct Gemm {
    pub dtype: DType,
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Which buffers a GEMM kernel reads and writes.
#[derive(Clone, Debug)]
pub(super) struct Binding {
    pub a: KernelArg,
    pub b: KernelArg,
    pub out: KernelArg,
}

/// Types and helper source for one dtype.
struct Code {
    info: TypeInfo,
    /// C type a value is computed and staged in.
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

/// The shared helpers: load, multiply-add, add, store.
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
/// `tile` is the forced tile, if any; the caller passes `None` to use the table.
pub(super) fn kernel(
    g: &Gemm,
    bind: &Binding,
    caps: &Caps,
    limits: &Limits,
    tile: Option<Tile>,
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
        let tile = select_tile(limits, elem, (g.batch, g.m, g.n), tile)?;
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
    if instr.operands.len() != 2 {
        return Err(format!(
            "batched_gemm takes 2 operands, got {}",
            instr.operands.len()
        ));
    }
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
    kernel(&g, &bind, caps, limits, forced_tile()?).map(Plan::single)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::buf;
    use super::super::types::Kind;
    use super::*;

    const F32: DType = DType::new(Kind::F, 32);
    const S32: DType = DType::new(Kind::S, 32);
    const C64: DType = DType::new(Kind::C, 64);

    fn limits(max_wg: usize, local_mem: usize) -> Limits {
        Limits::new(max_wg, local_mem)
    }

    fn gemm(dtype: DType, batch: usize, m: usize, n: usize, k: usize) -> Instr {
        let op = |shape: [usize; 3]| buf(dtype, &shape);
        Instr {
            op: "batched_gemm".to_string(),
            operands: vec![op([batch, m, k]), op([batch, k, n])],
            out: dtype,
            out_shape: vec![batch, m, n],
            attrs: [batch, m, n, k]
                .iter()
                .map(|&v| Attr::Int(v as i64))
                .collect(),
        }
    }

    fn spec(i: &Instr, l: &Limits) -> KernelSpec {
        let plan = kernel(
            &Gemm {
                dtype: i.out,
                batch: i.out_shape[0],
                m: i.out_shape[1],
                n: i.out_shape[2],
                k: i.operands[0].shape[2],
            },
            &Binding {
                a: KernelArg::Operand(0),
                b: KernelArg::Operand(1),
                out: KernelArg::Output,
            },
            &Caps::all(),
            l,
            None,
        );
        plan.unwrap()
    }

    #[test]
    fn a_big_problem_gets_the_largest_tile_and_a_medium_one_a_smaller_tile() {
        let l = limits(256, 48 * 1024);
        assert_eq!(select_tile(&l, 4, (1, 2048, 2048), None).unwrap(), TABLE[0]);
        // 512 x 512 makes 16 groups of 128 x 128 and 64 of 64 x 64: fewer than
        // the target, so the 64 x 32 tile (128 groups) wins.
        assert_eq!(select_tile(&l, 4, (1, 512, 512), None).unwrap(), TABLE[2]);
        // A batch adds groups, so a large tile serves small products.
        assert_eq!(select_tile(&l, 4, (128, 128, 128), None).unwrap(), TABLE[0]);
    }

    #[test]
    fn a_device_with_wide_groups_takes_the_cpu_table() {
        let t = select_tile(&limits(4096, 64 * 1024), 4, (1, 2048, 2048), None).unwrap();
        assert_eq!(t, WIDE_GROUP_TABLE[0]);
        let small = select_tile(&limits(4096, 64 * 1024), 4, (1, 7, 7), None).unwrap();
        assert!(small.bm <= 16);
    }

    #[test]
    fn small_local_memory_picks_a_smaller_tile() {
        // The largest tile needs 8 KiB for f32; 2 KiB rules it out.
        let t = select_tile(&limits(256, 2 * 1024), 4, (1, 2048, 2048), None).unwrap();
        assert!(t.bm < 128 && t.local_bytes(4) <= 2 * 1024);
        // Wide values take twice the room, so the choice shrinks again.
        let w = select_tile(&limits(256, 2 * 1024), 16, (1, 2048, 2048), None).unwrap();
        assert!(w.local_bytes(16) <= 2 * 1024 && w.bm <= t.bm);
    }

    #[test]
    fn a_small_group_limit_picks_a_tile_with_few_threads() {
        let t = select_tile(&limits(64, 32 * 1024), 4, (1, 512, 512), None).unwrap();
        assert!(t.threads() <= 64);
        let one = select_tile(&limits(1, 32 * 1024), 4, (1, 512, 512), None).unwrap();
        assert_eq!(one.threads(), 1);
    }

    #[test]
    fn a_tiny_problem_does_not_take_the_largest_tile() {
        let t = select_tile(&limits(256, 32 * 1024), 4, (1, 7, 7), None).unwrap();
        assert!(t.bm <= 16 && t.bn <= 16, "{:?}", t);
    }

    #[test]
    fn every_tile_in_the_table_is_valid() {
        for t in TABLE {
            assert!(t.valid(), "{:?}", t);
        }
    }

    #[test]
    fn nothing_fits_without_local_memory() {
        assert!(select_tile(&limits(256, 4), 4, (1, 64, 64), None).is_err());
    }

    #[test]
    fn the_override_wins_when_it_fits() {
        let forced = parse_tile("32,64,8,2,4").unwrap();
        let t = select_tile(&limits(256, 32 * 1024), 4, (1, 512, 512), Some(forced)).unwrap();
        assert_eq!(t, forced);
    }

    #[test]
    fn an_override_that_does_not_fit_falls_back_to_the_table() {
        let forced = parse_tile("128,128,32,4,4").unwrap();
        let l = limits(256, 8 * 1024);
        assert!(!forced.fits(&l, 4));
        let t = select_tile(&l, 4, (1, 512, 512), Some(forced)).unwrap();
        assert!(t != forced && t.fits(&l, 4));
    }

    #[test]
    fn the_override_parser_rejects_bad_text() {
        assert!(parse_tile("64,64,16,4").is_err());
        assert!(parse_tile("64,64,16,4,x").is_err());
        assert!(parse_tile("64,64,16,3,4").is_err());
        assert!(parse_tile("0,64,16,4,4").is_err());
        assert_eq!(
            parse_tile(" 8, 8, 4, 1, 1 ").unwrap(),
            Tile::new(8, 8, 4, 1, 1)
        );
    }

    #[test]
    fn the_tile_is_in_the_build_options() {
        let s = spec(&gemm(F32, 1, 2048, 2048, 64), &limits(256, 32 * 1024));
        assert_eq!(s.build_options, TABLE[0].options());
        assert!(s.build_options.contains("-DBM=128"));
        let small = spec(&gemm(F32, 1, 2048, 2048, 64), &limits(256, 2 * 1024));
        assert_ne!(small.build_options, s.build_options);
    }

    #[test]
    fn the_tiled_kernel_bounds_checks_and_sizes_the_launch() {
        let s = spec(&gemm(F32, 128, 70, 33, 20), &limits(256, 32 * 1024));
        assert!(s.entry.starts_with("gemm_tiled_"));
        assert!(s.source.contains("gr < M && gk < K") && s.source.contains("gc < N"));
        // The batch of 128 leaves enough groups for the 128 by 128 tile, with 16 by 16
        // threads: 33 columns and 70 rows each need one group, and the batch is the third axis.
        assert_eq!(s.build_options, TABLE[0].options());
        assert_eq!(s.global, vec![16, 16, 128]);
        assert_eq!(s.local, vec![16, 16, 1]);
        assert_eq!(s.args.len(), 6);
    }

    #[test]
    fn n_equal_one_uses_the_matrix_vector_kernel() {
        let s = spec(&gemm(F32, 2, 9, 1, 100), &limits(256, 32 * 1024));
        assert!(s.entry.starts_with("gemm_matvec_"));
        assert_eq!(s.build_options, "-DWG=128");
        assert_eq!(s.global, vec![9 * 128, 2]);
        assert_eq!(s.local, vec![128, 1]);
        // A vector dot a vector takes the same path.
        assert!(spec(&gemm(F32, 1, 1, 1, 5), &limits(256, 32 * 1024))
            .entry
            .starts_with("gemm_matvec_"));
    }

    #[test]
    fn m_equal_one_uses_the_vector_matrix_kernel() {
        let s = spec(&gemm(F32, 2, 1, 40, 30), &limits(256, 32 * 1024));
        assert!(s.entry.starts_with("gemm_vecmat_"));
        assert_eq!(s.global, vec![40, 2]);
        assert!(s.local.is_empty());
    }

    #[test]
    fn k_equal_zero_fills_zeros_without_a_gemm_kernel() {
        let s = spec(&gemm(F32, 2, 3, 4, 0), &limits(256, 32 * 1024));
        assert!(s.entry.starts_with("gemm_zero_"));
        assert!(!s.source.contains("gemm_mac"));
        assert_eq!(s.global, vec![24]);
        assert_eq!(s.args.len(), 2);
    }

    #[test]
    fn integers_accumulate_in_a_wide_unsigned_type() {
        let s = spec(&gemm(S32, 1, 8, 8, 8), &limits(256, 32 * 1024));
        assert!(s.source.contains("static inline uint gemm_mac(uint acc"));
        assert!(s.source.contains("static inline int gemm_st(uint v)"));
        let s8 = spec(
            &gemm(DType::new(Kind::S, 8), 1, 8, 8, 8),
            &limits(256, 32 * 1024),
        );
        assert!(s8.source.contains("uint gemm_ld(char x)"));
    }

    #[test]
    fn bf16_and_f16_accumulate_in_f32() {
        let bf = spec(
            &gemm(DType::new(Kind::Bf, 16), 1, 8, 8, 8),
            &limits(256, 32 * 1024),
        );
        assert!(bf.source.contains("float gemm_mac(float acc"));
        assert!(bf.source.contains("bf16_to_f32"));
        let h = spec(
            &gemm(DType::new(Kind::F, 16), 1, 8, 8, 8),
            &limits(256, 32 * 1024),
        );
        assert!(h.source.contains("float gemm_mac(float acc"));
        assert!(h.source.contains("cl_khr_fp16"));
    }

    #[test]
    fn complex_uses_the_complex_helpers() {
        let s = spec(&gemm(C64, 1, 8, 8, 8), &limits(256, 32 * 1024));
        assert!(s.source.contains("cf_add(acc, cf_mul(a, b))"));
        assert!(s.source.contains("float2 cf_mul(float2 a, float2 b)"));
        assert!(s.source.starts_with("#pragma OPENCL FP_CONTRACT OFF"));
    }

    #[test]
    fn wide_values_take_a_smaller_tile_when_local_memory_is_tight() {
        let c128 = DType::new(Kind::C, 128);
        let big = spec(&gemm(c128, 1, 2048, 2048, 64), &limits(256, 32 * 1024));
        let s = spec(&gemm(c128, 1, 2048, 2048, 64), &limits(256, 16 * 1024));
        assert_ne!(s.build_options, big.build_options);
        assert!(s.source.contains("cl_khr_fp64"));
    }

    #[test]
    fn the_instruction_checks_types_and_shapes() {
        let ok = gemm(F32, 1, 4, 5, 6);
        assert!(generate(&ok, &Caps::all(), &Limits::apple()).is_ok());
        let mut bad = gemm(F32, 1, 4, 5, 6);
        bad.operands[1].shape = vec![1, 7, 5];
        assert!(generate(&bad, &Caps::all(), &Limits::apple())
            .unwrap_err()
            .contains("do not match"));
        let mut mixed = gemm(F32, 1, 4, 5, 6);
        mixed.operands[0].dtype = S32;
        assert!(generate(&mixed, &Caps::all(), &Limits::apple())
            .unwrap_err()
            .contains("output type"));
    }

    #[test]
    fn f64_needs_the_extension() {
        let f64t = DType::new(Kind::F, 64);
        let none = Caps {
            fp64: false,
            fp16: false,
            int64_atomics: true,
        };
        assert!(generate(&gemm(f64t, 1, 4, 4, 4), &none, &Limits::apple())
            .unwrap_err()
            .contains("cl_khr_fp64"));
    }

    #[test]
    fn a_bad_tile_variable_is_an_error_and_an_unset_one_is_not() {
        use std::env::VarError;
        assert_eq!(tile_from(Err(VarError::NotPresent)), Ok(None));
        assert_eq!(
            tile_from(Ok("8,8,4,1,1".to_string())),
            Ok(Some(Tile::new(8, 8, 4, 1, 1)))
        );
        let err = tile_from(Ok("8,8,4,1".to_string())).unwrap_err();
        assert!(err.contains("NXCL_GEMM_TILE"), "{err}");
        assert!(tile_from(Ok("8,8,4,3,3".to_string())).is_err());
        assert!(tile_from(Err(VarError::NotUnicode("x".into()))).is_err());
    }

    #[test]
    fn each_side_of_the_group_obeys_its_own_work_item_limit() {
        // 256 items in total, but at most 4 along each of the first two axes.
        let mut l = limits(256, 32 * 1024);
        l.max_item_sizes = [4, 4, 1];
        let t = select_tile(&l, 4, (1, 512, 512), None).unwrap();
        assert!(t.bn / t.tn <= 4 && t.bm / t.tm <= 4, "{:?}", t);
        // The 128 x 128 tile has 16 x 16 items and no longer fits.
        assert!(!TABLE[0].fits(&l, 4));
        // A forced tile obeys the limit as well.
        let forced = parse_tile("128,128,8,8,8").unwrap();
        assert!(!forced.fits(&l, 4));
    }

    #[test]
    fn sizes_that_overflow_are_errors() {
        let big = 1usize << 30;
        let mut i = gemm(F32, 1, 1, 1, 1);
        i.operands = vec![buf(F32, &[big, big, 1]), buf(F32, &[big, 1, big])];
        i.out_shape = vec![big, big, big];
        i.attrs = vec![
            Attr::Int(big as i64),
            Attr::Int(big as i64),
            Attr::Int(big as i64),
            Attr::Int(1),
        ];
        let err = generate(&i, &Caps::all(), &limits(256, 32 * 1024)).unwrap_err();
        assert!(err.contains("the gemm output size is too large"), "{err}");
        // A negative size is not a size.
        let mut i = gemm(F32, 1, 2, 2, 2);
        i.attrs[1] = Attr::Int(-1);
        assert!(generate(&i, &Caps::all(), &limits(256, 32 * 1024))
            .unwrap_err()
            .contains("must be a size"));
    }

    #[test]
    fn an_unknown_c_type_has_no_size() {
        assert_eq!(c_type_bytes("float2"), Ok(8));
        assert_eq!(c_type_bytes("double2"), Ok(16));
        assert!(c_type_bytes("quad").unwrap_err().contains("quad"));
    }
}
