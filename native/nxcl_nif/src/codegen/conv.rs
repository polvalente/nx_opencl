//! `conv`: a convolution as im2col followed by `batched_gemm`.
//!
//! The instruction reads an input `[N, C, spatial...]` and a kernel
//! `[O, I, kernel spatial...]` and writes `[O, N', output spatial...]`
//! (`N' = N / batch_groups`); Elixir permutes the layouts. Groups are the GEMM
//! batch axis. With `G` groups (`feature_group_size` or `batch_group_size`; Nx
//! allows only one above 1) and `Og = O / G`:
//!
//! * pass 1 gathers `cols[G][K][NP]` into scratch, with `K = I * prod(kernel
//!   spatial)` and `NP = N' * prod(output spatial)`. A cell is zero where the
//!   position is padding or between dilated input cells;
//! * pass 2 is `out[G][Og][NP] = kernel[G][Og][K] x cols[G][K][NP]`, reading the
//!   kernel in place.
//!
//! `expand` maps conv sizes to GEMM sizes and scratch size; `Expansion::source`
//! is the matching index map. Neither needs a device, so `cargo test` checks
//! them against a direct convolution.

use super::gemm::{self, Binding, Gemm};
use super::index::MAX_RANK;
use super::movement::{assemble, ints, pragmas};
use super::types::type_info;
use super::{checked_product, element_count, Caps, Instr, KernelArg, KernelSpec, Limits, Plan};

pub fn handles(op: &str) -> bool {
    op == "conv"
}

/// The sizes of a normalized convolution. Vectors have one entry per spatial axis.
#[derive(Clone, Debug, PartialEq)]
struct Conv {
    pub batch: usize,
    pub channels: usize,
    pub in_sp: Vec<usize>,
    pub filters: usize,
    pub kernel_channels: usize,
    pub kernel_sp: Vec<usize>,
    pub strides: Vec<usize>,
    pub pad: Vec<(i64, i64)>,
    pub kernel_dilation: Vec<usize>,
    pub input_dilation: Vec<usize>,
    pub feature_groups: usize,
    pub batch_groups: usize,
}

/// The GEMM a convolution turns into, and where the columns come from.
#[derive(Clone, Debug, PartialEq)]
struct Expansion {
    pub groups: usize,
    /// Output channels per group (`M` of the GEMM).
    pub og: usize,
    /// `I * prod(kernel spatial)` (`K` of the GEMM).
    pub k: usize,
    /// Output batch (`N / batch_groups`).
    pub n_out: usize,
    pub out_sp: Vec<usize>,
    /// `N' * prod(out_sp)` (`N` of the GEMM).
    pub np: usize,
    /// `prod(out_sp)`.
    pub out_positions: usize,
    /// Elements in the `cols` scratch buffer: `groups * k * np`.
    pub cols: usize,
    /// Input batch step per group: `N'` for batch groups, else 0.
    pub batch_step: usize,
    /// Input channel step per group: `I` for feature groups, else 0.
    pub chan_step: usize,
}

impl Expansion {
    pub fn cols_len(&self) -> usize {
        self.cols
    }

    pub fn out_shape(&self) -> Vec<usize> {
        let mut s = vec![self.groups * self.og, self.n_out];
        s.extend(&self.out_sp);
        s
    }

    pub fn gemm(&self, dtype: super::DType) -> Gemm {
        Gemm {
            dtype,
            batch: self.groups,
            m: self.og,
            n: self.np,
            k: self.k,
        }
    }

    /// The flat input index that column cell `(g, kk, np)` reads, or `None` where
    /// the cell is padding or lies between dilated input cells.
    #[cfg(test)]
    pub fn source(&self, c: &Conv, g: usize, kk: usize, np: usize) -> Option<usize> {
        let d = c.in_sp.len();
        let kp: usize = c.kernel_sp.iter().product();
        let p: usize = self.out_sp.iter().product();
        let (n, mut pos) = (np / p, np % p);
        let (ch, mut kpos) = (kk / kp, kk % kp);
        let mut opos = vec![0usize; d];
        let mut wpos = vec![0usize; d];
        for a in (0..d).rev() {
            opos[a] = pos % self.out_sp[a];
            pos /= self.out_sp[a];
            wpos[a] = kpos % c.kernel_sp[a];
            kpos /= c.kernel_sp[a];
        }
        let mut idx = 0usize;
        let sp_total: usize = c.in_sp.iter().product();
        for a in 0..d {
            let q = (opos[a] * c.strides[a] + wpos[a] * c.kernel_dilation[a]) as i64 - c.pad[a].0;
            let dil = c.input_dilation[a] as i64;
            if q < 0 || q % dil != 0 || q / dil >= c.in_sp[a] as i64 {
                return None;
            }
            idx = idx * c.in_sp[a] + (q / dil) as usize;
        }
        let nb = n + g * self.batch_step;
        let chn = g * self.chan_step + ch;
        Some((nb * c.channels + chn) * sp_total + idx)
    }
}

/// Checks the sizes and returns the expansion. `out_shape` is the
/// instruction's output shape, `[O, N', spatial...]`.
fn expand(c: &Conv, out_shape: &[usize]) -> Result<Expansion, String> {
    let d = c.in_sp.len();
    if d == 0 {
        return Err("conv needs at least one spatial axis".to_string());
    }
    let lens = [
        c.kernel_sp.len(),
        c.strides.len(),
        c.pad.len(),
        c.kernel_dilation.len(),
        c.input_dilation.len(),
    ];
    if lens.iter().any(|&l| l != d) {
        return Err(format!("conv: every option needs {} entries", d));
    }
    if c.strides
        .iter()
        .chain(&c.kernel_dilation)
        .chain(&c.input_dilation)
        .any(|&v| v < 1)
    {
        return Err("conv: strides and dilations must be positive".to_string());
    }
    let (fg, bg) = (c.feature_groups, c.batch_groups);
    if fg < 1 || bg < 1 || (fg > 1 && bg > 1) {
        return Err("conv: feature and batch groups cannot both exceed 1".to_string());
    }
    if c.channels != c.kernel_channels * fg {
        return Err(format!(
            "conv: {} input channels do not match {} kernel channels in {} groups",
            c.channels, c.kernel_channels, fg
        ));
    }
    if !c.filters.is_multiple_of(fg * bg) || !c.batch.is_multiple_of(bg) {
        return Err("conv: filters and batch must divide by the group count".to_string());
    }
    if bg > 1 && c.channels != c.kernel_channels {
        return Err("conv: batch groups need the kernel to cover every channel".to_string());
    }
    let groups = fg * bg;
    let n_out = c.batch / bg;
    // Extent of `size` cells spaced `step` apart: `(size - 1) * step + 1`, or 0
    // for no cells. The sizes come from the wire, so the product is checked.
    let spread = |what: &str, a: usize, size: usize, step: usize| -> Result<usize, String> {
        if size == 0 {
            return Ok(0);
        }
        (size - 1)
            .checked_mul(step)
            .and_then(|v| v.checked_add(1))
            .filter(|&v| i64::try_from(v).is_ok())
            .ok_or_else(|| format!("conv: the {} of axis {} is too large", what, a))
    };
    let mut out_sp = Vec::with_capacity(d);
    for a in 0..d {
        let dilated = spread("dilated input", a, c.in_sp[a], c.input_dilation[a])?;
        let extent = spread("dilated kernel", a, c.kernel_sp[a], c.kernel_dilation[a])?;
        if extent == 0 {
            return Err(format!("conv: the kernel has size 0 along axis {}", a));
        }
        // Both extents fit in an i64, and the sum of three i64 values fits in i128.
        let padded = dilated as i128 + c.pad[a].0 as i128 + c.pad[a].1 as i128;
        if padded < extent as i128 {
            return Err(format!("conv: the kernel does not fit axis {}", a));
        }
        let cells = (padded - extent as i128) / c.strides[a] as i128 + 1;
        out_sp.push(
            usize::try_from(cells)
                .map_err(|_| format!("conv: the output of axis {} is too large", a))?,
        );
    }
    let out_positions = checked_product("the conv output positions", &out_sp)?;
    let kernel_positions = checked_product("the conv kernel positions", &c.kernel_sp)?;
    let k = checked_product("the conv K", &[c.kernel_channels, kernel_positions])?;
    let np = checked_product("the conv columns", &[n_out, out_positions])?;
    let e = Expansion {
        groups,
        og: c.filters / groups,
        k,
        n_out,
        np,
        out_positions,
        cols: checked_product("the conv column buffer", &[groups, k, np])?,
        out_sp,
        batch_step: if bg > 1 { n_out } else { 0 },
        chan_step: if fg > 1 { c.kernel_channels } else { 0 },
    };
    if e.out_shape() != out_shape {
        return Err(format!(
            "conv: output shape {:?} does not match the expected {:?}",
            out_shape,
            e.out_shape()
        ));
    }
    Ok(e)
}

/// Reads the instruction attrs: `D, strides, (lo, hi) per axis, kernel dilation,
/// input dilation, feature groups, batch groups`.
fn parse(instr: &Instr) -> Result<Conv, String> {
    if instr.operands.len() != 2 {
        return Err(format!(
            "conv takes 2 operands, got {}",
            instr.operands.len()
        ));
    }
    let (x, w) = (&instr.operands[0], &instr.operands[1]);
    if x.constant.is_some() || w.constant.is_some() {
        return Err("conv operands must be buffers".to_string());
    }
    if x.dtype != instr.out || w.dtype != instr.out {
        return Err(format!(
            "conv operands must have the output type {}",
            instr.out
        ));
    }
    let a = ints(&instr.attrs, "conv")?;
    let first = *a.first().ok_or("conv: missing the spatial rank")?;
    let d = usize::try_from(first)
        .ok()
        .filter(|&d| (1..=MAX_RANK).contains(&d))
        .ok_or_else(|| {
            format!(
                "conv: the spatial rank D is {}, not between 1 and {}",
                first, MAX_RANK
            )
        })?;
    if a.len() != 1 + 5 * d + 2 || x.shape.len() != d + 2 || w.shape.len() != d + 2 {
        return Err(format!(
            "conv: expected {} attrs and rank {} operands for {} spatial axes",
            1 + 5 * d + 2,
            d + 2,
            d
        ));
    }
    // A negative stride, dilation or group count is an error here, so a later
    // check never reports it as a zero.
    let size = |v: i64, what: &str| -> Result<usize, String> {
        usize::try_from(v).map_err(|_| format!("conv: {} {} is negative", what, v))
    };
    let take = |from: usize, what: &str| -> Result<Vec<usize>, String> {
        a[1 + from * d..1 + (from + 1) * d]
            .iter()
            .map(|&v| size(v, what))
            .collect()
    };
    let pad_at = 1 + d;
    Ok(Conv {
        batch: x.shape[0],
        channels: x.shape[1],
        in_sp: x.shape[2..].to_vec(),
        filters: w.shape[0],
        kernel_channels: w.shape[1],
        kernel_sp: w.shape[2..].to_vec(),
        strides: take(0, "stride")?,
        pad: (0..d)
            .map(|i| (a[pad_at + 2 * i], a[pad_at + 2 * i + 1]))
            .collect(),
        kernel_dilation: take(3, "kernel dilation")?,
        input_dilation: take(4, "input dilation")?,
        feature_groups: size(a[1 + 5 * d], "feature group count")?,
        batch_groups: size(a[2 + 5 * d], "batch group count")?,
    })
}

/// The gather kernel. All sizes are `long` arguments, so the source depends on
/// the spatial rank and the type only.
fn im2col(instr: &Instr, c: &Conv, e: &Expansion, caps: &Caps) -> Result<KernelSpec, String> {
    let d = c.in_sp.len();
    let info = type_info(instr.out)?;
    let pragma = pragmas(&instr.dtypes(), caps)?;
    let entry = format!("im2col_{}d_{}", d, instr.out.name());

    let mut params = vec![
        format!("__global const {}* in", info.storage),
        format!("__global {}* cols", info.storage),
    ];
    let mut args = vec![KernelArg::Operand(0), KernelArg::Scratch(0)];
    let mut long = |name: String, v: i64| {
        params.push(format!("const long {name}"));
        args.push(KernelArg::Scalar(v.to_ne_bytes().to_vec()));
    };
    let sp_total = checked_product("the conv input positions", &c.in_sp)?;
    let kernel_positions = checked_product("the conv kernel positions", &c.kernel_sp)?;
    long("total".into(), e.cols_len() as i64);
    long("K".into(), e.k as i64);
    long("NP".into(), e.np as i64);
    long("P".into(), e.out_positions as i64);
    long("KP".into(), kernel_positions as i64);
    long("C".into(), c.channels as i64);
    long("ISP".into(), sp_total as i64);
    long("bstep".into(), e.batch_step as i64);
    long("cstep".into(), e.chan_step as i64);
    for a in 0..d {
        long(format!("o{a}"), e.out_sp[a] as i64);
        long(format!("w{a}"), c.kernel_sp[a] as i64);
        long(format!("n{a}"), c.in_sp[a] as i64);
        long(format!("st{a}"), c.strides[a] as i64);
        long(format!("kd{a}"), c.kernel_dilation[a] as i64);
        long(format!("id{a}"), c.input_dilation[a] as i64);
        long(format!("lo{a}"), c.pad[a].0);
    }

    let mut body = String::from(
        "  const long id = get_global_id(0);\n  if (id >= total) return;\n  \
         const long np = id % NP;\n  const long t = id / NP;\n  \
         const long kk = t % K;\n  const long g = t / K;\n  \
         const long n = np / P;\n  long pos = np % P;\n  \
         const long ch = kk / KP;\n  long kpos = kk % KP;\n  long idx = 0;\n  int ok = 1;\n",
    );
    for a in (0..d).rev() {
        body.push_str(&format!(
            "  const long op{a} = pos % o{a};\n  pos /= o{a};\n  const long wp{a} = kpos % w{a};\n  kpos /= w{a};\n"
        ));
    }
    for a in 0..d {
        body.push_str(&format!(
            "  {{\n    const long q = op{a} * st{a} + wp{a} * kd{a} - lo{a};\n    \
             if (q < 0 || q % id{a} != 0 || q / id{a} >= n{a}) ok = 0;\n    \
             idx = idx * n{a} + q / id{a};\n  }}\n"
        ));
    }
    body.push_str(&format!(
        "  const long src = ((n + g * bstep) * C + g * cstep + ch) * ISP + idx;\n  \
         cols[id] = ok ? in[src] : ({s})(0);\n",
        s = info.storage
    ));
    Ok(KernelSpec {
        source: assemble(&pragma, &entry, &params.join(", "), &body),
        entry,
        build_options: String::new(),
        global: vec![e.cols_len().max(1)],
        local: Vec::new(),
        args,
    })
}

pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    let c = parse(instr)?;
    let e = expand(&c, &instr.out_shape)?;
    let g = e.gemm(instr.out);
    let bytes = type_info(instr.out)?.bytes;
    let out_elems = element_count(&instr.out_shape);
    let cols_bytes = checked_product("the conv column buffer", &[e.cols_len(), bytes])?;
    // No columns to gather: the GEMM writes zeros (K == 0) or nothing.
    let empty = e.cols_len() == 0 || out_elems == 0;
    let bind = Binding {
        a: KernelArg::Operand(1),
        b: KernelArg::Scratch(0),
        out: KernelArg::Output,
    };
    let gemm_pass = gemm::kernel(&g, &bind, caps, limits, gemm::forced_tile()?)?;
    if empty {
        return Ok(Plan::single(gemm_pass));
    }
    Ok(Plan {
        passes: vec![im2col(instr, &c, &e, caps)?, gemm_pass],
        scratch: vec![cols_bytes],
    })
}

/// A `conv` instruction over `x` and `w` of the given dtype, for tests.
#[cfg(test)]
fn test_instr(c: &Conv, dtype: super::DType) -> Instr {
    use super::{Attr, Operand};
    let e = expand(c, &expect_shape(c)).unwrap();
    let shape = |lead: [usize; 2], sp: &[usize]| {
        let mut s = lead.to_vec();
        s.extend(sp);
        s
    };
    let op = |shape: Vec<usize>| Operand {
        dtype,
        shape,
        constant: None,
    };
    let mut attrs: Vec<i64> = vec![c.in_sp.len() as i64];
    attrs.extend(c.strides.iter().map(|&v| v as i64));
    attrs.extend(c.pad.iter().flat_map(|&(l, h)| [l, h]));
    attrs.extend(c.kernel_dilation.iter().map(|&v| v as i64));
    attrs.extend(c.input_dilation.iter().map(|&v| v as i64));
    attrs.push(c.feature_groups as i64);
    attrs.push(c.batch_groups as i64);
    Instr {
        op: "conv".to_string(),
        operands: vec![
            op(shape([c.batch, c.channels], &c.in_sp)),
            op(shape([c.filters, c.kernel_channels], &c.kernel_sp)),
        ],
        out: dtype,
        out_shape: e.out_shape(),
        attrs: attrs.into_iter().map(Attr::Int).collect(),
    }
}

/// The output shape `expand` would compute, for tests that build instructions.
#[cfg(test)]
fn expect_shape(c: &Conv) -> Vec<usize> {
    let d = c.in_sp.len();
    let mut s = vec![c.filters, c.batch / c.batch_groups];
    for a in 0..d {
        let dilated = (c.in_sp[a] - 1) * c.input_dilation[a] + 1;
        let extent = (c.kernel_sp[a] - 1) * c.kernel_dilation[a] + 1;
        let padded = dilated as i64 + c.pad[a].0 + c.pad[a].1;
        s.push(((padded - extent as i64) as usize) / c.strides[a] + 1);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::super::types::Kind;
    use super::super::DType;
    use super::*;

    const F32: DType = DType::new(Kind::F, 32);

    fn conv2d() -> Conv {
        Conv {
            batch: 2,
            channels: 3,
            in_sp: vec![5, 6],
            filters: 4,
            kernel_channels: 3,
            kernel_sp: vec![3, 2],
            strides: vec![1, 1],
            pad: vec![(0, 0), (0, 0)],
            kernel_dilation: vec![1, 1],
            input_dilation: vec![1, 1],
            feature_groups: 1,
            batch_groups: 1,
        }
    }

    /// A direct convolution with the reference semantics, on small integers.
    fn direct(c: &Conv, e: &Expansion, x: &[i64], w: &[i64]) -> Vec<i64> {
        let d = c.in_sp.len();
        let og = e.og;
        let p: usize = e.out_sp.iter().product();
        let kp: usize = c.kernel_sp.iter().product();
        let sp_total: usize = c.in_sp.iter().product();
        let mut out = vec![0i64; c.filters * e.n_out * p];
        for o in 0..c.filters {
            let g = o / og;
            for n in 0..e.n_out {
                for pos in 0..p {
                    let mut opos = vec![0usize; d];
                    let mut r = pos;
                    for a in (0..d).rev() {
                        opos[a] = r % e.out_sp[a];
                        r /= e.out_sp[a];
                    }
                    let mut sum = 0;
                    for ch in 0..c.kernel_channels {
                        for kpos in 0..kp {
                            let mut wpos = vec![0usize; d];
                            let mut r = kpos;
                            for a in (0..d).rev() {
                                wpos[a] = r % c.kernel_sp[a];
                                r /= c.kernel_sp[a];
                            }
                            let mut flat = 0usize;
                            let mut inside = true;
                            for a in 0..d {
                                let q = (opos[a] * c.strides[a] + wpos[a] * c.kernel_dilation[a])
                                    as i64
                                    - c.pad[a].0;
                                let dil = c.input_dilation[a] as i64;
                                if q < 0 || q % dil != 0 || q / dil >= c.in_sp[a] as i64 {
                                    inside = false;
                                    break;
                                }
                                flat = flat * c.in_sp[a] + (q / dil) as usize;
                            }
                            if inside {
                                let nb = n + g * e.batch_step;
                                let chn = g * e.chan_step + ch;
                                let xi = (nb * c.channels + chn) * sp_total + flat;
                                let wi = (o * c.kernel_channels + ch) * kp + kpos;
                                sum += x[xi] * w[wi];
                            }
                        }
                    }
                    out[(o * e.n_out + n) * p + pos] = sum;
                }
            }
        }
        out
    }

    /// im2col through `Expansion::source`, then the GEMM in plain Rust.
    fn via_gemm(c: &Conv, e: &Expansion, x: &[i64], w: &[i64]) -> Vec<i64> {
        let mut out = vec![0i64; e.groups * e.og * e.np];
        for g in 0..e.groups {
            for m in 0..e.og {
                for n in 0..e.np {
                    let mut s = 0;
                    for k in 0..e.k {
                        if let Some(i) = e.source(c, g, k, n) {
                            s += w[(g * e.og + m) * e.k + k] * x[i];
                        }
                    }
                    out[(g * e.og + m) * e.np + n] = s;
                }
            }
        }
        out
    }

    fn data(len: usize, seed: i64) -> Vec<i64> {
        (0..len as i64)
            .map(|i| (i * 7 + seed * 3) % 11 - 5)
            .collect()
    }

    fn check(c: &Conv) {
        let e = expand(c, &expect_shape(c)).unwrap();
        let xl = c.batch * c.channels * c.in_sp.iter().product::<usize>();
        let wl = c.filters * c.kernel_channels * c.kernel_sp.iter().product::<usize>();
        let (x, w) = (data(xl, 1), data(wl, 2));
        assert_eq!(direct(c, &e, &x, &w), via_gemm(c, &e, &x, &w), "{:?}", c);
    }

    #[test]
    fn the_expansion_sizes_follow_the_options() {
        let e = expand(&conv2d(), &[4, 2, 3, 5]).unwrap();
        assert_eq!((e.groups, e.og, e.k, e.np), (1, 4, 18, 2 * 3 * 5));
        assert_eq!(e.cols_len(), 18 * 30);
        assert_eq!(e.out_shape(), vec![4, 2, 3, 5]);
    }

    #[test]
    fn the_cols_buffer_size_in_bytes_is_in_the_plan() {
        let i = test_instr(&conv2d(), F32);
        let plan = generate(&i, &Caps::all(), &Limits::apple()).unwrap();
        assert_eq!(plan.scratch, vec![18 * 30 * 4]);
        assert_eq!(plan.passes.len(), 2);
        // The gather reads the input and writes the scratch buffer.
        assert_eq!(
            plan.passes[0].args[..2],
            [KernelArg::Operand(0), KernelArg::Scratch(0)]
        );
        // The GEMM reads the kernel buffer and the scratch buffer.
        assert_eq!(
            plan.passes[1].args[..3],
            [
                KernelArg::Operand(1),
                KernelArg::Scratch(0),
                KernelArg::Output
            ]
        );
    }

    #[test]
    fn padding_strides_and_dilations_match_a_direct_conv() {
        let mut c = conv2d();
        c.pad = vec![(1, 2), (2, 0)];
        check(&c);
        let mut c = conv2d();
        c.strides = vec![2, 3];
        check(&c);
        let mut c = conv2d();
        c.kernel_dilation = vec![2, 2];
        c.in_sp = vec![8, 7];
        check(&c);
        let mut c = conv2d();
        c.input_dilation = vec![2, 3];
        c.pad = vec![(1, 1), (0, 2)];
        check(&c);
        let mut c = conv2d();
        c.pad = vec![(-1, 0), (0, -1)];
        check(&c);
        let mut c = conv2d();
        c.strides = vec![2, 1];
        c.kernel_dilation = vec![2, 1];
        c.input_dilation = vec![2, 1];
        c.pad = vec![(2, 1), (1, 1)];
        check(&c);
    }

    #[test]
    fn one_and_three_spatial_axes() {
        let c1 = Conv {
            batch: 2,
            channels: 2,
            in_sp: vec![9],
            filters: 3,
            kernel_channels: 2,
            kernel_sp: vec![3],
            strides: vec![2],
            pad: vec![(1, 1)],
            kernel_dilation: vec![2],
            input_dilation: vec![1],
            feature_groups: 1,
            batch_groups: 1,
        };
        check(&c1);
        let c3 = Conv {
            batch: 1,
            channels: 2,
            in_sp: vec![4, 5, 3],
            filters: 2,
            kernel_channels: 2,
            kernel_sp: vec![2, 2, 2],
            strides: vec![1, 2, 1],
            pad: vec![(1, 0), (0, 1), (1, 1)],
            kernel_dilation: vec![1, 1, 2],
            input_dilation: vec![2, 1, 1],
            feature_groups: 1,
            batch_groups: 1,
        };
        check(&c3);
    }

    #[test]
    fn feature_groups_and_depthwise_match_a_direct_conv() {
        let mut c = conv2d();
        c.channels = 4;
        c.kernel_channels = 2;
        c.filters = 6;
        c.feature_groups = 2;
        check(&c);
        // Depthwise: one input channel per group, one filter per group.
        let mut dw = conv2d();
        dw.channels = 3;
        dw.kernel_channels = 1;
        dw.filters = 3;
        dw.feature_groups = 3;
        check(&dw);
        let e = expand(&dw, &expect_shape(&dw)).unwrap();
        assert_eq!((e.groups, e.og), (3, 1));
    }

    #[test]
    fn batch_groups_match_a_direct_conv() {
        let mut c = conv2d();
        c.batch = 4;
        c.kernel_channels = 3;
        c.filters = 6;
        c.batch_groups = 2;
        check(&c);
        let e = expand(&c, &expect_shape(&c)).unwrap();
        assert_eq!((e.groups, e.og, e.n_out, e.batch_step), (2, 3, 2, 2));
    }

    #[test]
    fn bad_options_are_errors() {
        let mut c = conv2d();
        c.feature_groups = 2;
        c.batch_groups = 2;
        assert!(expand(&c, &[4, 1, 3, 5]).is_err());
        let mut c = conv2d();
        c.kernel_channels = 2;
        assert!(expand(&c, &[4, 2, 3, 5]).unwrap_err().contains("channels"));
        assert!(expand(&conv2d(), &[4, 2, 3, 6])
            .unwrap_err()
            .contains("output shape"));
        let mut c = conv2d();
        c.kernel_sp = vec![9, 2];
        assert!(expand(&c, &[4, 2, 1, 5])
            .unwrap_err()
            .contains("does not fit"));
    }

    #[test]
    fn the_source_has_one_loop_of_index_math_per_spatial_axis() {
        for d in 1..=3usize {
            let c = Conv {
                batch: 1,
                channels: 1,
                in_sp: vec![4; d],
                filters: 1,
                kernel_channels: 1,
                kernel_sp: vec![2; d],
                strides: vec![1; d],
                pad: vec![(0, 0); d],
                kernel_dilation: vec![1; d],
                input_dilation: vec![1; d],
                feature_groups: 1,
                batch_groups: 1,
            };
            let i = test_instr(&c, F32);
            let plan = generate(&i, &Caps::all(), &Limits::apple()).unwrap();
            let src = &plan.passes[0].source;
            for a in 0..d {
                assert!(
                    src.contains(&format!("const long op{a} =")),
                    "{d}d axis {a}"
                );
                assert!(src.contains(&format!("const long lo{a}")), "{d}d axis {a}");
            }
            assert!(!src.contains(&format!("const long op{d} =")));
        }
    }

    #[test]
    fn the_source_does_not_depend_on_the_sizes() {
        let a = generate(&test_instr(&conv2d(), F32), &Caps::all(), &Limits::apple()).unwrap();
        let mut c = conv2d();
        c.in_sp = vec![9, 9];
        c.strides = vec![2, 2];
        let b = generate(&test_instr(&c, F32), &Caps::all(), &Limits::apple()).unwrap();
        assert_eq!(a.passes[0].source, b.passes[0].source);
    }

    #[test]
    fn complex_and_half_types_reach_the_gemm_helpers() {
        let c64 = DType::new(Kind::C, 64);
        let p = generate(&test_instr(&conv2d(), c64), &Caps::all(), &Limits::apple()).unwrap();
        assert!(p.passes[1].source.contains("cf_mul"));
        assert!(p.passes[0].source.contains("float2"));
        assert_eq!(p.scratch, vec![18 * 30 * 8]);
    }

    /// The conv2d instruction with attr `at` replaced by `v`.
    fn with_attr(at: usize, v: i64) -> Instr {
        let mut i = test_instr(&conv2d(), F32);
        i.attrs[at] = super::super::Attr::Int(v);
        i
    }

    fn conv_err(i: &Instr) -> String {
        generate(i, &Caps::all(), &Limits::apple()).unwrap_err()
    }

    #[test]
    fn a_bad_spatial_rank_names_d_and_the_limit() {
        for d in [-1, 0, 9, i64::MAX] {
            let err = conv_err(&with_attr(0, d));
            assert!(
                err.contains("D is") && err.contains("between 1 and 8"),
                "{d}: {err}"
            );
        }
    }

    #[test]
    fn negative_strides_dilations_and_groups_are_errors_that_say_so() {
        // Attr layout for two spatial axes: D, strides (1-2), pads (3-6),
        // kernel dilation (7-8), input dilation (9-10), groups (11, 12).
        for (at, what) in [
            (1, "stride"),
            (7, "kernel dilation"),
            (9, "input dilation"),
            (11, "feature group count"),
            (12, "batch group count"),
        ] {
            let err = conv_err(&with_attr(at, -2));
            assert!(
                err.contains(what) && err.contains("negative"),
                "{what}: {err}"
            );
        }
        // Zero is still a different error: it is not negative.
        let err = conv_err(&with_attr(1, 0));
        assert!(err.contains("must be positive"), "{err}");
    }

    #[test]
    fn products_that_overflow_are_errors_not_wraps() {
        let big = usize::MAX / 2 + 1;
        let mut c = conv2d();
        c.in_sp = vec![big, 3];
        c.input_dilation = vec![4, 1];
        let err = expand(&c, &[4, 2, 1, 1]).unwrap_err();
        assert!(err.contains("too large"), "{err}");
        let mut c = conv2d();
        c.kernel_sp = vec![3, big];
        c.kernel_dilation = vec![1, 3];
        assert!(expand(&c, &[4, 2, 1, 1]).unwrap_err().contains("too large"));
        // Padding at the limits of i64 cannot wrap the sum.
        let mut c = conv2d();
        c.pad = vec![(i64::MAX, i64::MAX), (0, 0)];
        assert!(expand(&c, &[4, 2, 1, 1]).is_err());
        c.pad = vec![(i64::MIN, i64::MIN), (0, 0)];
        assert!(expand(&c, &[4, 2, 1, 1])
            .unwrap_err()
            .contains("does not fit"));
        // Sizes whose column buffer overflows.
        let mut c = conv2d();
        c.in_sp = vec![1 << 31, 1 << 31];
        c.kernel_sp = vec![1, 1];
        c.batch = 1 << 20;
        let out = [4, 1 << 20, 1 << 31, 1 << 31];
        assert!(expand(&c, &out).unwrap_err().contains("too large"));
    }

    #[test]
    fn a_zero_sized_kernel_and_a_kernel_that_does_not_fit_have_their_own_messages() {
        let mut c = conv2d();
        c.kernel_sp = vec![0, 2];
        let err = expand(&c, &[4, 2, 1, 1]).unwrap_err();
        assert!(err.contains("size 0 along axis 0"), "{err}");
        let mut c = conv2d();
        c.kernel_sp = vec![9, 2];
        let err = expand(&c, &[4, 2, 1, 1]).unwrap_err();
        assert!(err.contains("does not fit axis 0"), "{err}");
    }
}
