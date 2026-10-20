//! `fft` and `ifft` along one axis of a `c64` or `c128` tensor. The lowering
//! pads or crops the axis first, so input and output shapes match. `ifft`
//! divides by the length, as Nx does.
//!
//! - A power-of-two length `N = 2^L` takes `L` Stockham radix-2 passes, `N / 2`
//!   butterflies per line, one work item each. The input, the scratch buffer and
//!   the output take turns as source and target; the last pass writes the
//!   output. The result is in natural order.
//! - Any other length uses Bluestein's method: a cyclic convolution of length
//!   `M` (the next power of two at least `2N - 1`) done with two power-of-two
//!   transforms. Pre-multiply by the chirp and pad, forward transform, multiply
//!   by the transform of the chirp filter, inverse transform, post-multiply.
//! - `ifft` is `conj(fft(conj(x))) / N`: a sign flag on the twiddles and a final scale.
//!
//! The host builds twiddles, the chirp and the filter transform in `f64` and
//! rounds once to the element width. The device never calls `sin` or `cos`, so
//! accuracy holds at long lengths and without `fp64`.

use super::complex;
use super::index;
use super::movement::{assemble, pragmas};
use super::types::{type_info, TypeInfo};
use super::{
    checked_product, expect_operands, long_arg, Attr, Caps, Instr, KernelArg, KernelSpec, Plan,
};
use std::f64::consts::PI;
use std::sync::Arc;

const OPS: [&str; 2] = ["fft", "ifft"];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

type C64 = (f64, f64);

/// In-place forward FFT of a power-of-two length, in `f64`.
fn fft_f64(x: &mut [C64]) {
    let n = x.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (-2.0 * PI * k as f64 / len as f64).sin_cos();
                let (ar, ai) = x[start + k];
                let (br, bi) = x[start + k + len / 2];
                let (tr, ti) = (br * c - bi * s, br * s + bi * c);
                x[start + k] = (ar + tr, ai + ti);
                x[start + k + len / 2] = (ar - tr, ai - ti);
            }
        }
        len <<= 1;
    }
}

/// `exp(-2 pi i j / n)` for `j` in `0..n/2`.
fn twiddles(n: usize) -> Vec<C64> {
    (0..n / 2)
        .map(|j| {
            let (s, c) = (-2.0 * PI * j as f64 / n as f64).sin_cos();
            (c, s)
        })
        .collect()
}

/// `exp(-i pi k^2 / n)`. The square is reduced mod `2n` in integers first.
fn chirp(n: usize) -> Vec<C64> {
    (0..n)
        .map(|k| {
            let r = (k as u128 * k as u128 % (2 * n as u128)) as f64;
            let (s, c) = (-PI * r / n as f64).sin_cos();
            (c, s)
        })
        .collect()
}

/// The forward transform, length `m`, of the filter `conj(chirp)` laid out
/// cyclically: entry `j` and entry `m - j` hold `conj(chirp[j])`.
fn filter_transform(n: usize, m: usize) -> Vec<C64> {
    let mut b = vec![(0.0, 0.0); m];
    for (j, (re, im)) in chirp(n).into_iter().enumerate() {
        b[j] = (re, -im);
        if j > 0 {
            b[m - j] = (re, -im);
        }
    }
    fft_f64(&mut b);
    b
}

/// The table as bytes of the element type: two `f32` or two `f64` per entry.
/// The passes that read it share the one allocation.
fn table_bytes(info: &TypeInfo, values: &[C64]) -> Arc<[u8]> {
    let mut out = Vec::with_capacity(values.len() * info.bytes);
    for &(re, im) in values {
        if info.component == "double" {
            out.extend_from_slice(&re.to_ne_bytes());
            out.extend_from_slice(&im.to_ne_bytes());
        } else {
            out.extend_from_slice(&(re as f32).to_ne_bytes());
            out.extend_from_slice(&(im as f32).to_ne_bytes());
        }
    }
    out.into()
}

fn scale_bytes(info: &TypeInfo, scale: f64) -> Vec<u8> {
    if info.component == "double" {
        scale.to_ne_bytes().to_vec()
    } else {
        (scale as f32).to_ne_bytes().to_vec()
    }
}

#[derive(Clone)]
struct Ctx<'a> {
    instr: &'a Instr,
    info: TypeInfo,
    head: String,
    /// Elements of one line, lines, and the axis stride (the product of the
    /// axes after it).
    n: usize,
    lines: usize,
    inner: usize,
    inverse: bool,
}

impl Ctx<'_> {
    fn spec(
        &self,
        entry: &str,
        params: &str,
        body: &str,
        global: usize,
        args: Vec<KernelArg>,
    ) -> KernelSpec {
        let entry = format!("{}_{}", entry, self.instr.out.name());
        KernelSpec {
            source: assemble(&self.head, &entry, params, body),
            entry,
            build_options: String::new(),
            global: vec![global.max(1)],
            local: Vec::new(),
            args,
        }
    }

    fn long(v: usize) -> Result<KernelArg, String> {
        Ok(KernelArg::Scalar(long_arg(v)?))
    }

    fn scale_arg(&self, scale: f64) -> KernelArg {
        KernelArg::Scalar(scale_bytes(&self.info, scale))
    }

    fn st(&self) -> &'static str {
        self.info.storage
    }

    fn ct(&self) -> &'static str {
        self.info.component
    }

    fn scaled(&self, v: &str) -> String {
        format!("({})(({v}).x * scale, ({v}).y * scale)", self.st(), v = v)
    }

    /// One Stockham pass. Reads `src` and the twiddle table, writes `dst`.
    /// `len` is the line length of both and `inner` the axis stride. `n` is the
    /// length of the sub-transform this pass splits and `s` the stride between
    /// its elements: `n` halves and `s` doubles from one pass to the next.
    fn stage(
        &self,
        (src, dst, table): (KernelArg, KernelArg, &Arc<[u8]>),
        (len, inner): (usize, usize),
        (n, s): (usize, usize),
        scale: f64,
    ) -> Result<KernelSpec, String> {
        let (st, ct) = (self.st(), self.ct());
        let items = checked_product("the fft butterfly count", &[self.lines, len / 2])?;
        let params = format!(
            "__global const {st}* src, __global {st}* dst, __global const {st}* tw, \
             const long half_len, const long len, const long inner, const long n, \
             const long s, const long inv, const {ct} scale, const long total"
        );
        let body = format!(
            "  long gid = (long)get_global_id(0);
  if (gid >= total) return;
  long line = gid / half_len;
  long t = gid % half_len;
  long p = t / s;
  long q = t % s;
  long base = (line / inner) * len * inner + (line % inner);
  long m = n / 2;
  {st} a = src[base + (q + s * p) * inner];
  {st} b = src[base + (q + s * (p + m)) * inner];
  {st} w = tw[p * s];
  if (inv) w = {conj}(w);
  {st} u = {add}(a, b);
  {st} v = {mul}({sub}(a, b), w);
  dst[base + (q + s * (2 * p)) * inner] = {su};
  dst[base + (q + s * (2 * p + 1)) * inner] = {sv};
",
            conj = complex::fn_name(&self.info, "conj"),
            add = complex::fn_name(&self.info, "add"),
            sub = complex::fn_name(&self.info, "sub"),
            mul = complex::fn_name(&self.info, "mul"),
            su = self.scaled("u"),
            sv = self.scaled("v"),
        );
        let args = vec![
            src,
            dst,
            KernelArg::Table(table.clone()),
            Self::long(len / 2)?,
            Self::long(len)?,
            Self::long(inner)?,
            Self::long(n)?,
            Self::long(s)?,
            Self::long(self.inverse as usize)?,
            self.scale_arg(scale),
            Self::long(items)?,
        ];
        Ok(self.spec("fft_stage", &params, &body, items, args))
    }

    /// The `L` passes of a transform of lines of length `len = 2^L` with axis
    /// stride `inner` (`shape` is `(len, inner)`). The first pass reads `first`,
    /// the passes ping-pong with `other`, and the last writes `last`. The last
    /// pass multiplies by `scale`.
    fn run(
        &self,
        bufs: (KernelArg, KernelArg, KernelArg),
        shape: (usize, usize),
        scale: f64,
        table: &Arc<[u8]>,
    ) -> Result<Vec<KernelSpec>, String> {
        let (first, other, last) = bufs;
        let (len, inner) = shape;
        let stages = len.trailing_zeros() as usize;
        let mut passes = Vec::with_capacity(stages);
        let (mut n, mut s) = (len, 1);
        for k in 0..stages {
            let src = if k == 0 {
                first.clone()
            } else if (stages - 1 - k).is_multiple_of(2) {
                other.clone()
            } else {
                last.clone()
            };
            let dst = if (stages - 1 - k).is_multiple_of(2) {
                last.clone()
            } else {
                other.clone()
            };
            let sc = if k + 1 == stages { scale } else { 1.0 };
            passes.push(self.stage((src, dst, table), (len, inner), (n, s), sc)?);
            n /= 2;
            s *= 2;
        }
        Ok(passes)
    }

    fn copy(&self, scale: f64) -> Result<KernelSpec, String> {
        let st = self.st();
        let params = format!(
            "__global const {st}* src, __global {st}* dst, const {} scale, const long total",
            self.ct()
        );
        let body = format!(
            "  long gid = (long)get_global_id(0);
  if (gid >= total) return;
  dst[gid] = {};
",
            self.scaled("src[gid]")
        );
        let total = checked_product("the fft size", &self.instr.out_shape)?;
        let args = vec![
            KernelArg::Operand(0),
            KernelArg::Output,
            self.scale_arg(scale),
            Self::long(total)?,
        ];
        Ok(self.spec("fft_copy", &params, &body, total, args))
    }
}

// Memory per transform. Every pass reads one twiddle table, and `KernelArg::Table`
// shares it by `Arc`, so the table exists once on the host and once on the device,
// however many passes there are. Sizes are in bytes, `E` is the element size (8
// for `c64`, 16 for `c128`) and `N` the axis length.
//
// - Power of two: the table has `N / 2` entries, `N * E / 2` bytes. While the
//   host rounds it from `f64` it holds a second `N * 8` bytes. The device holds
//   the table plus an input, an output and a scratch buffer of `N * E` bytes per
//   line. At `N = 2^26`: host peak 768 MiB (`c64`) or 1 GiB (`c128`), and the
//   table on the device is 256 MiB or 512 MiB.
// - Bluestein: `M` is the next power of two at least `2N - 1`. Three tables
//   hold `N`, `M` and `M / 2` entries, at most `2 * M * E` bytes. The forward
//   and the inverse transform share the twiddle table. The device also holds two
//   scratch buffers of `M * E` bytes per line. At `N = 2^22` (`M = 2^23`) the
//   tables are 64 + 128 + 64 MiB for `c128` and half that for `c64`. The `f64`
//   filter transform adds `M * 16` bytes on the host while it is built.

/// The longest axis a power-of-two transform takes.
const MAX_POWER_OF_TWO: usize = 1 << 26;

/// The longest axis Bluestein's method takes.
const MAX_BLUESTEIN: usize = 1 << 22;

/// The length `M` Bluestein convolves over: the next power of two at least
/// `2n - 1`. `n` is at most `MAX_BLUESTEIN`.
fn bluestein_length(n: usize) -> usize {
    (2 * n - 1).next_power_of_two()
}

fn power_plan(ctx: &Ctx) -> Result<Plan, String> {
    let n = ctx.n;
    let scale = if ctx.inverse { 1.0 / n as f64 } else { 1.0 };
    if n == 1 {
        return Ok(Plan::single(ctx.copy(scale)?));
    }
    let table = table_bytes(&ctx.info, &twiddles(n));
    let bufs = (
        KernelArg::Operand(0),
        KernelArg::Scratch(0),
        KernelArg::Output,
    );
    let passes = ctx.run(bufs, (n, ctx.inner), scale, &table)?;
    let scratch = if passes.len() > 1 {
        let elements = checked_product("the fft size", &ctx.instr.out_shape)?;
        vec![checked_product(
            "the fft scratch buffer",
            &[elements, ctx.info.bytes],
        )?]
    } else {
        Vec::new()
    };
    Ok(Plan { passes, scratch })
}

fn bluestein_plan(ctx: &Ctx) -> Result<Plan, String> {
    let (n, lines, inner) = (ctx.n, ctx.lines, ctx.inner);
    let m = bluestein_length(n);
    let (st, ct) = (ctx.st(), ctx.ct());
    let info = &ctx.info;
    let chirp_table = table_bytes(info, &chirp(n));
    let filter_table = table_bytes(info, &filter_transform(n, m));
    let tw_table = table_bytes(info, &twiddles(m));
    let conj = complex::fn_name(info, "conj");
    let mul = complex::fn_name(info, "mul");
    let zero = complex::zero(info);
    let inv = ctx.inverse as usize;
    let lines_m = checked_product("the fft padded size", &[lines, m])?;
    let lines_n = checked_product("the fft size", &[lines, n])?;

    // Pre: scale by the chirp, zero-pad to `m`, into line-major scratch 0.
    let pre = {
        let params = format!(
            "__global const {st}* src, __global {st}* dst, __global const {st}* chirp, \
             const long n, const long m, const long inner, const long inv, const long total"
        );
        let body = format!(
            "  long gid = (long)get_global_id(0);
  if (gid >= total) return;
  long line = gid / m;
  long k = gid % m;
  long base = (line / inner) * n * inner + (line % inner);
  {st} v = {zero};
  if (k < n) {{
    {st} x = src[base + k * inner];
    if (inv) x = {conj}(x);
    v = {mul}(x, chirp[k]);
  }}
  dst[gid] = v;
"
        );
        let args = vec![
            KernelArg::Operand(0),
            KernelArg::Scratch(0),
            KernelArg::Table(chirp_table.clone()),
            Ctx::long(n)?,
            Ctx::long(m)?,
            Ctx::long(inner)?,
            Ctx::long(inv)?,
            Ctx::long(lines_m)?,
        ];
        ctx.spec("bl_pre", &params, &body, lines_m, args)
    };

    // Forward transform of length `m`, ping-ponging the two scratch buffers.
    let forward_ctx = Ctx {
        inverse: false,
        ..ctx.clone()
    };
    let (forward, at) = m_transform(&forward_ctx, m, 0, 1.0, &tw_table)?;

    // Multiply by the transform of the filter, in place in scratch 0.
    let filter = {
        let params =
            format!("__global {st}* buf, __global const {st}* flt, const long m, const long total");
        let body = format!(
            "  long gid = (long)get_global_id(0);
  if (gid >= total) return;
  buf[gid] = {mul}(buf[gid], flt[gid % m]);
"
        );
        let args = vec![
            KernelArg::Scratch(at),
            KernelArg::Table(filter_table),
            Ctx::long(m)?,
            Ctx::long(lines_m)?,
        ];
        ctx.spec("bl_filter", &params, &body, lines_m, args)
    };

    // Inverse transform of length `m`, with the `1 / m` scale in its last pass.
    let backward_ctx = Ctx {
        inverse: true,
        ..ctx.clone()
    };
    let (backward, at) = m_transform(&backward_ctx, m, at, 1.0 / m as f64, &tw_table)?;

    // Post: scale by the chirp, drop the padding, undo the conjugation.
    let post = {
        let params = format!(
            "__global const {st}* src, __global {st}* dst, __global const {st}* chirp, \
             const long n, const long m, const long inner, const long inv, const {ct} scale, \
             const long total"
        );
        let body = format!(
            "  long gid = (long)get_global_id(0);
  if (gid >= total) return;
  long line = gid / n;
  long k = gid % n;
  long base = (line / inner) * n * inner + (line % inner);
  {st} r = {mul}(src[line * m + k], chirp[k]);
  if (inv) r = {conj}(r);
  dst[base + k * inner] = {scaled};
",
            scaled = ctx.scaled("r")
        );
        let scale = if ctx.inverse { 1.0 / n as f64 } else { 1.0 };
        let args = vec![
            KernelArg::Scratch(at),
            KernelArg::Output,
            KernelArg::Table(chirp_table),
            Ctx::long(n)?,
            Ctx::long(m)?,
            Ctx::long(inner)?,
            Ctx::long(inv)?,
            ctx.scale_arg(scale),
            Ctx::long(lines_n)?,
        ];
        ctx.spec("bl_post", &params, &body, lines_n, args)
    };

    let mut passes = vec![pre];
    passes.extend(forward);
    passes.push(filter);
    passes.extend(backward);
    passes.push(post);
    let one = checked_product("the fft scratch buffer", &[lines_m, info.bytes])?;
    Ok(Plan {
        passes,
        scratch: vec![one, one],
    })
}

/// Passes of a length-`m` transform that starts in scratch `from`. The result
/// lands in the buffer the second return value names. The last buffer is chosen
/// so the first pass never writes the buffer it reads.
fn m_transform(
    ctx: &Ctx,
    m: usize,
    from: usize,
    scale: f64,
    table: &Arc<[u8]>,
) -> Result<(Vec<KernelSpec>, usize), String> {
    let stages = m.trailing_zeros() as usize;
    let to = if stages.is_multiple_of(2) {
        from
    } else {
        1 - from
    };
    let bufs = (
        KernelArg::Scratch(from),
        KernelArg::Scratch(1 - to),
        KernelArg::Scratch(to),
    );
    Ok((ctx.run(bufs, (m, 1), scale, table)?, to))
}

/// Attrs: `[axis]`. The operand and the output are complex, of one type and shape.
pub fn generate(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    expect_operands(instr, 1)?;
    let input = &instr.operands[0];
    let info = type_info(instr.out)?;
    if !info.is_complex() || input.dtype != instr.out {
        return Err(format!(
            "{}: needs a complex operand of the output type, got {} for {}",
            instr.op, input.dtype, instr.out
        ));
    }
    if input.shape != instr.out_shape {
        return Err(format!(
            "{}: the output shape must equal the input shape",
            instr.op
        ));
    }
    index::check_rank(&instr.op, input.shape.len())?;
    let axis = match instr.attrs.as_slice() {
        [Attr::Int(a)] => usize::try_from(*a)
            .ok()
            .filter(|&a| a < input.shape.len())
            .ok_or_else(|| format!("{}: axis {} is out of range", instr.op, a))?,
        _ => return Err(format!("{}: attrs must be [axis]", instr.op)),
    };
    let n = input.shape[axis];
    if n == 0 {
        return Err(format!("{}: the axis has no elements", instr.op));
    }
    let power_of_two = n.is_power_of_two();
    let limit = if power_of_two {
        MAX_POWER_OF_TWO
    } else {
        MAX_BLUESTEIN
    };
    if n > limit {
        return Err(format!(
            "{}: an axis of {} elements is too long for a {} transform (limit {}); \
             the host builds its tables",
            instr.op,
            n,
            if power_of_two {
                "power-of-two"
            } else {
                "Bluestein"
            },
            limit
        ));
    }
    let inner = checked_product("the fft inner size", &input.shape[axis + 1..])?;
    let outer = checked_product("the fft outer size", &input.shape[..axis])?;
    let lines = checked_product("the fft line count", &[outer, inner])?;
    let head = format!(
        "{}{}",
        pragmas(&instr.dtypes(), caps)?,
        complex::preamble(&info, &["add", "sub", "mul", "conj"])
    );
    let ctx = Ctx {
        instr,
        info,
        head,
        n,
        lines,
        inner,
        inverse: instr.op == "ifft",
    };
    if power_of_two {
        power_plan(&ctx)
    } else {
        bluestein_plan(&ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{self, plan};
    use super::super::types::Kind;
    use super::super::DType;
    use super::*;

    const C64: DType = DType::new(Kind::C, 64);
    const C128: DType = DType::new(Kind::C, 128);
    const F32: DType = DType::new(Kind::F, 32);

    fn instr(op: &str, dtype: DType, shape: &[usize], axis: i64) -> Instr {
        test_support::instr(
            op,
            vec![test_support::buf(dtype, shape)],
            dtype,
            shape,
            vec![Attr::Int(axis)],
        )
    }

    fn dft(x: &[C64]) -> Vec<C64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(re, im), (j, &(a, b))| {
                        let ang = -2.0 * PI * ((j * k) % n) as f64 / n as f64;
                        let (s, c) = ang.sin_cos();
                        (re + a * c - b * s, im + a * s + b * c)
                    })
            })
            .collect()
    }

    #[test]
    fn the_host_fft_matches_the_definition() {
        let x: Vec<C64> = (0..16)
            .map(|i| ((i as f64).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let mut y = x.clone();
        fft_f64(&mut y);
        for (a, b) in y.iter().zip(dft(&x)) {
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
        }
    }

    #[test]
    fn bluestein_with_host_arithmetic_matches_the_definition() {
        // The same steps the passes run, in f64: chirp, filter, product, inverse.
        let n = 100;
        let m = bluestein_length(n);
        assert_eq!(m, 256);
        let x: Vec<C64> = (0..n)
            .map(|i| ((i as f64).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let w = chirp(n);
        let mut a = vec![(0.0, 0.0); m];
        for k in 0..n {
            a[k] = (
                x[k].0 * w[k].0 - x[k].1 * w[k].1,
                x[k].0 * w[k].1 + x[k].1 * w[k].0,
            );
        }
        fft_f64(&mut a);
        let b = filter_transform(n, m);
        let mut c: Vec<C64> = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0))
            .collect();
        // Inverse through conj(fft(conj(.))) / m.
        for v in c.iter_mut() {
            v.1 = -v.1;
        }
        fft_f64(&mut c);
        let got: Vec<C64> = (0..n)
            .map(|k| {
                let (re, im) = (c[k].0 / m as f64, -c[k].1 / m as f64);
                (re * w[k].0 - im * w[k].1, re * w[k].1 + im * w[k].0)
            })
            .collect();
        for (g, e) in got.iter().zip(dft(&x)) {
            assert!(
                (g.0 - e.0).abs() < 1e-9 && (g.1 - e.1).abs() < 1e-9,
                "{g:?} {e:?}"
            );
        }
    }

    #[test]
    fn a_power_of_two_takes_log_n_passes_and_one_scratch_buffer() {
        let p = plan(&instr("fft", C64, &[3, 8], 1));
        assert_eq!(p.passes.len(), 3);
        assert_eq!(p.scratch, vec![3 * 8 * 8]);
        // The last pass writes the output; the first reads the operand.
        assert_eq!(p.passes[0].args[0], KernelArg::Operand(0));
        assert_eq!(p.passes[2].args[1], KernelArg::Output);
        let p = plan(&instr("fft", C64, &[2], 0));
        assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
        let p = plan(&instr("ifft", C128, &[1], 0));
        assert_eq!(p.passes.len(), 1);
        // One source serves every pass.
        let p = plan(&instr("fft", C64, &[1024], 0));
        assert_eq!(p.passes.len(), 10);
        assert!(p.passes.iter().all(|s| s.source == p.passes[0].source));
    }

    #[test]
    fn other_lengths_use_bluestein_over_the_next_power_of_two() {
        assert_eq!(bluestein_length(3), 8);
        assert_eq!(bluestein_length(100), 256);
        assert_eq!(bluestein_length(4097), 16384);
        let p = plan(&instr("fft", C64, &[100], 0));
        // Pre, 8 forward, filter, 8 inverse, post.
        assert_eq!(p.passes.len(), 19);
        assert_eq!(p.scratch, vec![256 * 8, 256 * 8]);
    }

    #[test]
    fn the_device_source_has_no_trig_and_c64_tables_are_f32() {
        let p = plan(&instr("fft", C64, &[100, 4], 0));
        for s in &p.passes {
            assert!(!s.source.contains("sin(") && !s.source.contains("cos("));
        }
        let t = table_bytes(&type_info(C64).unwrap(), &twiddles(8));
        assert_eq!(t.len(), 4 * 8);
        let t = table_bytes(&type_info(C128).unwrap(), &twiddles(8));
        assert_eq!(t.len(), 4 * 16);
    }

    #[test]
    fn real_types_and_bad_attrs_are_errors() {
        let caps = Caps::all();
        assert!(generate(&instr("fft", F32, &[8], 0), &caps).is_err());
        assert!(generate(&instr("fft", C64, &[8], 1), &caps).is_err());
        let mut bad = instr("fft", C64, &[8], 0);
        bad.attrs.clear();
        assert!(generate(&bad, &caps).is_err());
        let no64 = Caps {
            fp64: false,
            ..Caps::all()
        };
        assert!(generate(&instr("fft", C128, &[8], 0), &no64).is_err());
    }

    #[test]
    fn an_axis_beyond_the_host_table_limit_is_an_error() {
        let err =
            generate(&instr("fft", C64, &[MAX_POWER_OF_TWO * 2], 0), &Caps::all()).unwrap_err();
        assert!(
            err.contains("too long") && err.contains("power-of-two"),
            "{err}"
        );
        let err = generate(&instr("fft", C64, &[MAX_BLUESTEIN + 1], 0), &Caps::all()).unwrap_err();
        assert!(
            err.contains("too long") && err.contains("Bluestein"),
            "{err}"
        );
    }

    #[test]
    fn sizes_that_overflow_are_errors() {
        let big = 1usize << 40;
        let err = generate(&instr("fft", C64, &[big, big, 4], 2), &Caps::all()).unwrap_err();
        assert!(err.contains("is too large"), "{err}");
        let err = generate(&instr("fft", C64, &[big, big, 100], 2), &Caps::all()).unwrap_err();
        assert!(err.contains("is too large"), "{err}");
    }

    #[test]
    fn an_empty_axis_is_an_error() {
        let err = generate(&instr("fft", C64, &[3, 0], 1), &Caps::all()).unwrap_err();
        assert!(err.contains("no elements"), "{err}");
    }

    #[test]
    fn the_passes_of_one_transform_share_each_table() {
        let tables = |p: &Plan| -> Vec<Arc<[u8]>> {
            p.passes
                .iter()
                .flat_map(|s| s.args.iter())
                .filter_map(|a| match a {
                    KernelArg::Table(t) => Some(t.clone()),
                    _ => None,
                })
                .collect()
        };
        // 2^10: ten passes, one twiddle table.
        let p = plan(&instr("fft", C64, &[1024], 0));
        let t = tables(&p);
        assert_eq!(t.len(), 10);
        assert!(t.iter().all(|x| Arc::ptr_eq(x, &t[0])));
        // 100 is Bluestein: the chirp serves the first and last pass, and the
        // twiddles serve both transforms. Three distinct allocations in all.
        let p = plan(&instr("fft", C64, &[100], 0));
        let t = tables(&p);
        let mut distinct: Vec<&Arc<[u8]>> = Vec::new();
        for x in &t {
            if !distinct.iter().any(|d| Arc::ptr_eq(d, x)) {
                distinct.push(x);
            }
        }
        assert_eq!(distinct.len(), 3);
        assert!(t.len() > 3);
    }
}
