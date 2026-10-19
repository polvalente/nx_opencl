//! Window ops: `window_sum`, `window_product`, `window_max`, `window_min`,
//! the generic `window_reduce`, and `window_scatter_max` / `window_scatter_min`
//! (the gradient of max and min pooling).
//!
//! One work item per output element; no atomics. A window cell outside the
//! input reads as the initial value, as `Nx` pads with it, and folds in
//! row-major order over the window, so a generic body that is not associative
//! gives the reference result. Window size, strides, dilations and padding are
//! `long` arguments, so the source depends on the rank and the types only.
//!
//! `window_scatter_*` is written as a gather, since there are no atomics: the
//! work item of input element `p` visits each window that covers `p`, finds that
//! window's selected cell by the reference rule (the later of equal cells wins),
//! and adds the window's source value when the selected cell is `p`.

use super::elementwise::{scalar_binary, ScalarCode};
use super::index::{self, coords, dims, strides};
use super::movement::{ints, pragmas, Kernel};
use super::reduce::{combiner, parse_steps, Canon, CombKind};
use super::types::{bf16_helpers, load_as, math_of, store_as, type_info};
use super::{element_count, Caps, Instr, Plan};

const OPS: [&str; 7] = [
    "window_sum",
    "window_product",
    "window_max",
    "window_min",
    "window_reduce",
    "window_scatter_max",
    "window_scatter_min",
];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

/// Window parameters, one entry per input axis.
struct Window {
    dims: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
    /// Low padding; the high side does not enter the index math.
    low: Vec<i64>,
}

/// Reads `dims, strides, dilations, (low, high) * rank` from `attrs`.
fn parse_window(attrs: &[super::Attr], rank: usize) -> Result<Window, String> {
    let a = ints(attrs, "window")?;
    if a.len() != 5 * rank {
        return Err(format!(
            "window: expected {} attrs for rank {}, got {}",
            5 * rank,
            rank,
            a.len()
        ));
    }
    let take = |from: usize| a[from * rank..(from + 1) * rank].to_vec();
    let w = Window {
        dims: take(0),
        strides: take(1),
        dilations: take(2),
        low: (0..rank).map(|d| a[3 * rank + 2 * d]).collect(),
    };
    if w.dims.iter().any(|&v| v < 1)
        || w.strides.iter().any(|&v| v < 1)
        || w.dilations.iter().any(|&v| v < 1)
    {
        return Err("window: sizes, strides and dilations must be positive".to_string());
    }
    Ok(w)
}

/// Opens one `for` loop per axis with `header(d)` and puts `inner` inside.
fn nest(rank: usize, header: impl Fn(usize) -> String, inner: &str) -> String {
    let mut s = String::new();
    for d in 0..rank {
        s.push_str(&"  ".repeat(d + 1));
        s.push_str(&header(d));
        s.push_str(" {\n");
    }
    s.push_str(inner);
    for d in (0..rank).rev() {
        s.push_str(&"  ".repeat(d + 1));
        s.push_str("}\n");
    }
    s
}

/// `a >= 0 && a < n` over every axis, for coordinates `p0..`.
fn inside(rank: usize) -> String {
    if rank == 0 {
        return "1".to_string();
    }
    (0..rank)
        .map(|d| format!("p{d} >= 0 && p{d} < n{d}"))
        .collect::<Vec<_>>()
        .join(" && ")
}

fn input_index(rank: usize) -> String {
    if rank == 0 {
        return "0".to_string();
    }
    (0..rank)
        .map(|d| format!("p{d} * is{d}"))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn finish(instr: &Instr, k: Kernel, head: &str, entry: String, body: &str) -> Plan {
    let global = vec![element_count(&instr.out_shape)];
    Plan::single(k.into_spec(head, entry, body, global, Vec::new()))
}

/// `window_sum`, `window_product`, `window_max`, `window_min` and `window_reduce`.
fn fold(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    let generic = instr.op == "window_reduce";
    let want = if generic { 2 } else { 1 };
    if instr.operands.len() != want {
        return Err(format!(
            "{} takes {} operands, got {}",
            instr.op,
            want,
            instr.operands.len()
        ));
    }
    let input = &instr.operands[0];
    let rank = input.shape.len();
    index::check_rank(&instr.op, rank)?;
    if instr.out_shape.len() != rank {
        return Err(format!(
            "{}: the output must have the input's rank",
            instr.op
        ));
    }
    let (kind, window_attrs, tag) = if generic {
        let (steps, used) = parse_steps(&instr.attrs)?;
        let tag = combiner(&CombKind::Steps(steps.clone()), math_of(instr.out)?.dtype)?.tag;
        (CombKind::Steps(steps), &instr.attrs[used..], tag)
    } else {
        let canon = Canon::from_op(&instr.op)
            .ok_or_else(|| format!("{}: not a window reduction", instr.op))?;
        (CombKind::Canon(canon), &instr.attrs[..], canon_tag(canon))
    };
    let w = parse_window(window_attrs, rank)?;
    let acc = math_of(instr.out)?;
    let comb = combiner(&kind, acc.dtype)?;
    let in_info = type_info(input.dtype)?;
    let out_info = type_info(instr.out)?;
    let head = format!(
        "{}{}{}",
        pragmas(&instr.dtypes(), caps)?,
        bf16_helpers(&[&in_info, &out_info]),
        comb.code.render()
    );

    let mut k = Kernel::new(instr);
    k.operand(0)?;
    if generic {
        k.operand(1)?;
    }
    k.output()?;
    k.longs("o", &dims(&instr.out_shape));
    k.longs("n", &dims(&input.shape));
    k.longs("is", &strides(&input.shape));
    k.longs("st", &w.strides);
    k.longs("dl", &w.dilations);
    k.longs("lo", &w.low);
    k.longs("wd", &w.dims);

    // The initial value pads the window; a monoid uses its identity.
    let init = if generic {
        let init_info = type_info(instr.operands[1].dtype)?;
        load_as(&init_info, &k.read(1, "0"), &acc)?
    } else {
        comb.identity
            .clone()
            .ok_or("window op without an identity")?
    };
    let load = load_as(&in_info, &k.read(0, "idx"), &acc)?;
    let t = acc.math;
    let positions: String = (0..rank)
        .map(|d| {
            format!(
                "{}long p{d} = c{d} * st{d} + w{d} * dl{d} - lo{d};\n",
                "  ".repeat(rank + 1)
            )
        })
        .collect();
    let inner = format!(
        "{positions}{pad}{t} v = init;\n{pad}if ({inb}) {{ long idx = {idx}; v = {load}; }}\n{pad}acc = comb(v, acc);\n",
        pad = "  ".repeat(rank + 1),
        inb = inside(rank),
        idx = input_index(rank)
    );
    let loops = nest(
        rank,
        |d| format!("for (long w{d} = 0; w{d} < wd{d}; w{d}++)"),
        &inner,
    );
    let body = format!(
        "{coords}  {t} init = {init};\n  {t} acc = init;\n{loops}  out[i] = {store};\n",
        coords = coords(rank),
        store = store_as(&out_info, "acc")
    );
    let entry = format!(
        "wn_{}_{}_{}_r{}",
        tag,
        in_info.dtype.name(),
        out_info.dtype.name(),
        rank
    );
    Ok(finish(instr, k, &head, entry, &body))
}

fn canon_tag(canon: Canon) -> String {
    match canon {
        Canon::Sum => "sum",
        Canon::Product => "prod",
        Canon::Max => "max",
        Canon::Min => "min",
    }
    .to_string()
}

/// `window_scatter_max` and `window_scatter_min`.
fn scatter(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    if instr.operands.len() != 3 {
        return Err(format!(
            "{} takes 3 operands, got {}",
            instr.op,
            instr.operands.len()
        ));
    }
    let (input, source) = (&instr.operands[0], &instr.operands[1]);
    let rank = input.shape.len();
    index::check_rank(&instr.op, rank)?;
    if source.shape.len() != rank || instr.out_shape != input.shape {
        return Err(format!(
            "{}: source and output must match the input's rank and shape",
            instr.op
        ));
    }
    let w = parse_window(&instr.attrs, rank)?;
    let max = instr.op == "window_scatter_max";
    let acc = math_of(instr.out)?;
    let cmp_t = math_of(input.dtype)?;
    if cmp_t.is_complex() || acc.is_complex() {
        return Err(format!("{} does not accept complex values", instr.op));
    }
    let in_info = type_info(input.dtype)?;
    let src_info = type_info(source.dtype)?;
    let init_info = type_info(instr.operands[2].dtype)?;
    let out_info = type_info(instr.out)?;
    let mut code = ScalarCode::default();
    code.merge(scalar_binary("add", acc.dtype, "add_fn")?);
    let head = format!(
        "{}{}{}static long fdiv(long a, long b) {{ return a >= 0 ? a / b : -((-a + b - 1) / b); }}\nstatic long cdiv(long a, long b) {{ return -fdiv(-a, b); }}\n",
        pragmas(&instr.dtypes(), caps)?,
        bf16_helpers(&[&in_info, &src_info, &init_info, &out_info]),
        code.render()
    );

    let mut k = Kernel::new(instr);
    k.operand(0)?;
    k.operand(1)?;
    k.operand(2)?;
    k.output()?;
    k.longs("o", &dims(&input.shape));
    k.longs("is", &strides(&input.shape));
    k.longs("nw", &dims(&source.shape));
    k.longs("ss", &strides(&source.shape));
    k.longs("st", &w.strides);
    k.longs("lo", &w.low);
    k.longs("wd", &w.dims);

    let (ct, at) = (cmp_t.math, acc.math);
    let init_acc = load_as(&init_info, &k.read(2, "0"), &acc)?;
    let init_cmp = load_as(&init_info, &k.read(2, "0"), &cmp_t)?;
    let sym = if max { ">=" } else { "<=" };

    // Window index ranges that cover coordinate `c{d}`.
    let ranges: String = (0..rank)
        .map(|d| {
            format!(
                "  long a{d} = max(0L, cdiv(c{d} + lo{d} - wd{d} + 1, st{d}));\n  long b{d} = min(nw{d} - 1, fdiv(c{d} + lo{d}, st{d}));\n"
            )
        })
        .collect();

    // Inside one window `w`: pick the selected cell.
    let cell_pad = "  ".repeat(2 * rank + 2);
    let cell_pos: String = (0..rank)
        .map(|d| format!("{cell_pad}long q{d} = w{d} * st{d} + u{d} - lo{d};\n"))
        .collect();
    let q_inside = (0..rank)
        .map(|d| format!("q{d} >= 0 && q{d} < o{d}"))
        .collect::<Vec<_>>()
        .join(" && ");
    let q_index = (0..rank)
        .map(|d| format!("q{d} * is{d}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let load_x = load_as(&in_info, &k.read(0, "qi"), &cmp_t)?;
    let same_cell = (0..rank)
        .map(|d| format!("s{d} == c{d}"))
        .collect::<Vec<_>>()
        .join(" && ");
    let record: String = (0..rank).map(|d| format!("s{d} = q{d}; ")).collect();
    let cell_inner = format!(
        "{cell_pos}{cell_pad}{ct} x = x0;\n{cell_pad}if ({q_inside}) {{ long qi = {q_index}; x = {load_x}; }}\n{cell_pad}if (first || x {sym} best) {{ best = x; {record}first = 0; }}\n",
        q_inside = if rank == 0 { "1".to_string() } else { q_inside },
        q_index = if rank == 0 { "0".to_string() } else { q_index },
    );
    let cells = nest_at(
        rank,
        rank + 1,
        |d| format!("for (long u{d} = 0; u{d} < wd{d}; u{d}++)"),
        &cell_inner,
    );
    let src_load = load_as(&src_info, &k.read(1, "si"), &acc)?;
    let sidx = (0..rank)
        .map(|d| format!("w{d} * ss{d}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let window_inner = format!(
        "{pad}{ct} best = x0;\n{pad}int first = 1;\n{decl}{cells}{pad}if ({same}) {{ long si = {sidx}; acc = add_fn({src_load}, acc); }}\n",
        pad = "  ".repeat(rank + 1),
        decl = (0..rank)
            .map(|d| format!("{}long s{d} = 0;\n", "  ".repeat(rank + 1)))
            .collect::<String>(),
        same = if rank == 0 { "1".to_string() } else { same_cell },
        sidx = if rank == 0 { "0".to_string() } else { sidx },
    );
    let windows = nest(
        rank,
        |d| format!("for (long w{d} = a{d}; w{d} <= b{d}; w{d}++)"),
        &window_inner,
    );
    let body = format!(
        "{coords}{ranges}  {at} acc = {init_acc};\n  {ct} x0 = {init_cmp};\n{windows}  out[i] = {store};\n",
        coords = coords(rank),
        store = store_as(&out_info, "acc")
    );
    let entry = format!(
        "ws_{}_{}_{}_r{}",
        if max { "max" } else { "min" },
        in_info.dtype.name(),
        out_info.dtype.name(),
        rank
    );
    Ok(finish(instr, k, &head, entry, &body))
}

/// Like `nest`, with the first loop's indent at `depth - rank`.
fn nest_at(rank: usize, depth: usize, header: impl Fn(usize) -> String, inner: &str) -> String {
    let mut s = String::new();
    for d in 0..rank {
        s.push_str(&"  ".repeat(depth + d));
        s.push_str(&header(d));
        s.push_str(" {\n");
    }
    s.push_str(inner);
    for d in (0..rank).rev() {
        s.push_str(&"  ".repeat(depth + d));
        s.push_str("}\n");
    }
    s
}

pub fn generate(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    match instr.op.as_str() {
        "window_scatter_max" | "window_scatter_min" => scatter(instr, caps),
        _ => fold(instr, caps),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{buf, ints, plan};
    use super::super::types::Kind;
    use super::super::{Attr, DType};
    use super::*;

    const S32: DType = DType::new(Kind::S, 32);
    const F32: DType = DType::new(Kind::F, 32);
    const C64: DType = DType::new(Kind::C, 64);

    /// Rank 2: dims, strides, dilations, then (low, high) per axis.
    fn window(dims: [i64; 2], strides: [i64; 2], pad: [(i64, i64); 2]) -> Vec<Attr> {
        ints(&[
            dims[0], dims[1], strides[0], strides[1], 1, 1, pad[0].0, pad[0].1, pad[1].0, pad[1].1,
        ])
    }

    fn fold(op: &str, dtype: DType, attrs: Vec<Attr>, out_shape: &[usize]) -> Instr {
        Instr {
            op: op.to_string(),
            operands: vec![buf(dtype, &[6, 8])],
            out: dtype,
            out_shape: out_shape.to_vec(),
            attrs,
        }
    }

    #[test]
    fn each_canonical_op_is_one_launch_with_one_thread_per_output() {
        for op in ["window_sum", "window_product", "window_max", "window_min"] {
            let i = fold(op, F32, window([2, 2], [1, 1], [(0, 0), (0, 0)]), &[5, 7]);
            let p = plan(&i);
            assert_eq!((p.passes.len(), p.scratch.len()), (1, 0), "{op}");
            assert_eq!(p.passes[0].global, vec![35], "{op}");
        }
    }

    #[test]
    fn out_of_bounds_cells_read_as_the_identity() {
        let sum = plan(&fold(
            "window_sum",
            F32,
            window([3, 3], [1, 1], [(1, 1), (1, 1)]),
            &[6, 8],
        ));
        let max = plan(&fold(
            "window_max",
            F32,
            window([3, 3], [1, 1], [(1, 1), (1, 1)]),
            &[6, 8],
        ));
        // The padded cells fold the identity of the op: 0 for a sum, -inf for a max.
        assert!(sum.passes[0].source.contains("(float)0"));
        assert!(max.passes[0].source.contains("(-INFINITY)"));
        assert_ne!(sum.passes[0].source, max.passes[0].source);
    }

    #[test]
    fn window_geometry_is_not_part_of_the_source() {
        let a = plan(&fold(
            "window_sum",
            F32,
            window([2, 2], [1, 1], [(0, 0), (0, 0)]),
            &[5, 7],
        ));
        let b = plan(&fold(
            "window_sum",
            F32,
            window([2, 2], [2, 2], [(0, 0), (0, 0)]),
            &[3, 4],
        ));
        // Window size, strides and padding are arguments: one kernel serves both.
        assert_eq!(a.passes[0].source, b.passes[0].source);
    }

    #[test]
    fn a_wrong_attr_count_and_a_zero_window_are_errors() {
        let bad = fold("window_sum", F32, ints(&[2, 2, 1, 1]), &[5, 7]);
        assert!(generate(&bad, &Caps::all())
            .unwrap_err()
            .contains("expected 10 attrs"));
        let zero = fold(
            "window_sum",
            F32,
            window([0, 2], [1, 1], [(0, 0), (0, 0)]),
            &[5, 7],
        );
        assert!(generate(&zero, &Caps::all())
            .unwrap_err()
            .contains("positive"));
    }

    #[test]
    fn complex_max_min_and_scatter_are_errors_and_complex_sum_is_not() {
        let a = window([2, 2], [1, 1], [(0, 0), (0, 0)]);
        assert!(generate(&fold("window_max", C64, a.clone(), &[5, 7]), &Caps::all()).is_err());
        assert!(generate(&fold("window_sum", C64, a.clone(), &[5, 7]), &Caps::all()).is_ok());
        assert!(generate(&fold("window_product", C64, a, &[5, 7]), &Caps::all()).is_ok());
    }

    #[test]
    fn window_reduce_takes_an_initial_value_and_the_steps_of_its_body() {
        let mut attrs = vec![
            Attr::Int(1),
            Attr::Atom("add".to_string()),
            Attr::Int(0),
            Attr::Int(1),
        ];
        attrs.extend(window([2, 2], [1, 1], [(0, 0), (0, 0)]));
        let mut i = fold("window_reduce", S32, attrs, &[5, 7]);
        // The initial value is operand 1.
        assert!(generate(&i, &Caps::all()).is_err());
        i.operands.push(buf(S32, &[]));
        let p = plan(&i);
        assert_eq!(p.passes.len(), 1);
        assert!(p.passes[0].source.contains("step0"));
    }

    #[test]
    fn scatter_gathers_over_the_windows_that_cover_an_input_element() {
        let i = Instr {
            op: "window_scatter_max".to_string(),
            operands: vec![buf(F32, &[6, 8]), buf(F32, &[3, 4]), buf(F32, &[])],
            out: F32,
            out_shape: vec![6, 8],
            attrs: window([2, 2], [2, 2], [(0, 0), (0, 0)]),
        };
        let p = plan(&i);
        // One launch, one thread per input element, no atomics.
        assert_eq!(p.passes.len(), 1);
        assert_eq!(p.passes[0].global, vec![48]);
        assert!(!p.passes[0].source.contains("atomic"));
        let min = Instr {
            op: "window_scatter_min".to_string(),
            ..i.clone()
        };
        assert_ne!(plan(&min).passes[0].source, p.passes[0].source);
        let c = Instr {
            operands: vec![buf(C64, &[6, 8]), buf(C64, &[3, 4]), buf(C64, &[])],
            out: C64,
            ..i
        };
        assert!(generate(&c, &Caps::all()).is_err());
    }
}
