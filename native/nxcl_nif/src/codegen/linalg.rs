//! Dense linear algebra: `triangular_solve` and `cholesky`.
//!
//! `triangular_solve` solves `op(A) X = B` (`left_side`) or `X op(A) = B` for a
//! batch of triangular matrices. One work item solves one right-hand side by
//! substitution. The matrix is read through two strides, so a transpose or the
//! other side costs no copy; only the named triangle is read.
//!
//! `cholesky` runs one work group per matrix, right-looking: per column, take
//! the square root of the pivot, scale the column below, update the trailing
//! triangle, with a barrier between the steps. A matrix that is not positive
//! definite gives NaN. Complex input is taken as Hermitian, so the update
//! conjugates; a negative real pivot gives an imaginary one, as the reference
//! does. The upper triangle of the result is zero.
//!
//! Operands and output share one type; the lowering casts to it. `bf16` and
//! `f16` compute in `float` and round on store.

use super::complex;
use super::movement::{pragmas, Kernel};
use super::types::{bf16_helpers, load_as, math_of, store_as, TypeInfo};
use super::{checked_product, expect_operands, Attr, Caps, Instr, Limits, Plan};

const OPS: [&str; 2] = ["triangular_solve", "cholesky"];

pub fn handles(op: &str) -> bool {
    OPS.contains(&op)
}

struct Arith {
    complex: bool,
    info: TypeInfo,
    math: TypeInfo,
}

impl Arith {
    fn new(instr: &Instr) -> Result<Self, String> {
        let info = super::types::type_info(instr.out)?;
        if !info.is_float() && !info.is_complex() {
            return Err(format!(
                "{}: needs a float or complex type, got {}",
                instr.op, instr.out
            ));
        }
        Ok(Self {
            complex: info.is_complex(),
            info,
            math: math_of(instr.out)?,
        })
    }

    fn t(&self) -> &'static str {
        self.math.math
    }

    fn load(&self, raw: &str) -> Result<String, String> {
        load_as(&self.info, raw, &self.math)
    }

    fn store(&self, expr: &str) -> String {
        store_as(&self.info, expr)
    }

    fn sub(&self, a: &str, b: &str) -> String {
        if self.complex {
            format!("{}({}, {})", complex::fn_name(&self.math, "sub"), a, b)
        } else {
            format!("(({}) - ({}))", a, b)
        }
    }

    fn mul(&self, a: &str, b: &str) -> String {
        if self.complex {
            format!("{}({}, {})", complex::fn_name(&self.math, "mul"), a, b)
        } else {
            format!("(({}) * ({}))", a, b)
        }
    }

    fn div(&self, a: &str, b: &str) -> String {
        if self.complex {
            format!("{}({}, {})", complex::fn_name(&self.math, "div"), a, b)
        } else {
            format!("(({}) / ({}))", a, b)
        }
    }

    fn conj(&self, a: &str) -> String {
        if self.complex {
            format!("{}({})", complex::fn_name(&self.math, "conj"), a)
        } else {
            a.to_string()
        }
    }

    fn sqrt(&self, a: &str) -> String {
        if self.complex {
            format!("{}({})", complex::fn_name(&self.math, "sqrt"), a)
        } else {
            format!("sqrt({})", a)
        }
    }

    fn helpers(&self, names: &[&str]) -> String {
        format!(
            "{}{}",
            bf16_helpers(&[&self.info]),
            complex::preamble(&self.math, names)
        )
    }

    fn zero(&self) -> String {
        if self.complex {
            format!("({})(0, 0)", self.t())
        } else {
            "0".to_string()
        }
    }
}

fn check_operands(instr: &Instr, n: usize) -> Result<(), String> {
    expect_operands(instr, n)?;
    for (k, o) in instr.operands.iter().enumerate() {
        if o.dtype != instr.out {
            return Err(format!(
                "{}: operand {} is {}, the output {}",
                instr.op, k, o.dtype, instr.out
            ));
        }
    }
    Ok(())
}

/// Leading batch size and the matrix order of a stack of square matrices.
fn square_batch(op: &str, shape: &[usize]) -> Result<(usize, usize), String> {
    match shape {
        [batch @ .., n, m] if n == m => Ok((checked_product("the matrix batch", batch)?, *n)),
        _ => Err(format!(
            "{}: expected square matrices, got shape {:?}",
            op, shape
        )),
    }
}

fn atom(attrs: &[Attr], i: usize, what: &str) -> Result<String, String> {
    match attrs.get(i) {
        Some(Attr::Atom(a)) => Ok(a.clone()),
        _ => Err(format!("attr {} ({}) must be an atom", i, what)),
    }
}

fn flag(attrs: &[Attr], i: usize, what: &str) -> Result<bool, String> {
    match attrs.get(i) {
        Some(Attr::Int(v)) => Ok(*v != 0),
        _ => Err(format!("attr {} ({}) must be an integer", i, what)),
    }
}

/// Attrs: `left_side` (0 or 1), `lower` (0 or 1), `transform_a`
/// (`none`, `transpose` or `conjugate`).
fn triangular_solve(instr: &Instr, caps: &Caps) -> Result<Plan, String> {
    check_operands(instr, 2)?;
    let arith = Arith::new(instr)?;
    let left = flag(&instr.attrs, 0, "left_side")?;
    let lower = flag(&instr.attrs, 1, "lower")?;
    let transform = atom(&instr.attrs, 2, "transform_a")?;
    if !["none", "transpose", "conjugate"].contains(&transform.as_str()) {
        return Err(format!("triangular_solve: transform_a {}", transform));
    }
    let (a, b) = (&instr.operands[0], &instr.operands[1]);
    let (batch, n) = square_batch(&instr.op, &a.shape)?;
    if b.shape != instr.out_shape {
        return Err("triangular_solve: the output shape must equal the shape of b".to_string());
    }
    // A vector b has one axis fewer than a: one right-hand side per batch.
    let vector = b.shape.len() + 1 == a.shape.len();
    let (m, sx, so) = if vector {
        if b.shape[..b.shape.len() - 1] != a.shape[..a.shape.len() - 2]
            || b.shape[b.shape.len() - 1] != n
        {
            return Err("triangular_solve: b does not match a".to_string());
        }
        (1, 1, 0)
    } else if b.shape.len() == a.shape.len() {
        let r = b.shape.len();
        if b.shape[..r - 2] != a.shape[..r - 2] {
            return Err("triangular_solve: b does not match a".to_string());
        }
        let (rows, cols) = (b.shape[r - 2], b.shape[r - 1]);
        if left && rows == n {
            (cols, cols, 1)
        } else if !left && cols == n {
            (rows, 1, cols)
        } else {
            return Err("triangular_solve: b does not match a".to_string());
        }
    } else {
        return Err("triangular_solve: b does not match a".to_string());
    };
    // The effective matrix T has T(i, j) = A[i * a_rs + j * a_cs]. A right
    // side solve is the left side solve of the transpose.
    let transpose = transform == "transpose";
    let swap = left == transpose;
    let (a_rs, a_cs) = if swap { (1, n) } else { (n, 1) };
    let lower_eff = lower != swap;
    let conj = transform == "conjugate";

    let t = arith.t();
    let ta = |i: &str, j: &str| -> Result<String, String> {
        let v = arith.load(&format!("in0[abase + ({i}) * a_rs + ({j}) * a_cs]"))?;
        Ok(if conj { arith.conj(&v) } else { v })
    };
    let x_at = |i: &str| -> Result<String, String> { arith.load(&format!("out[xo + ({i}) * sx]")) };
    let b_at = arith.load("in1[xo + i * sx]")?;
    let step = |range: &str, start: &str| -> Result<String, String> {
        let upd = arith.sub("acc", &arith.mul(&ta("i", "j")?, &x_at("j")?));
        Ok(format!(
            "    {t} acc = {b_at};
    for (long j = {start}; {range}; j++) acc = {upd};
    out[xo + i * sx] = {store};
",
            store = arith.store(&arith.div("acc", &ta("i", "i")?)),
        ))
    };
    let forward = step("j < i", "0")?;
    let backward = step("j < n", "i + 1")?;
    let body = format!(
        "  long gid = (long)get_global_id(0);
  long batch = gid / m;
  long r = gid % m;
  long abase = batch * n * n;
  long xo = batch * n * m + r * so;
  if (lower) {{
    for (long i = 0; i < n; i++) {{
{forward}    }}
  }} else {{
    for (long i = n - 1; i >= 0; i--) {{
{backward}    }}
  }}
"
    );

    let mut k = Kernel::new(instr);
    k.operand(0)?;
    k.operand(1)?;
    k.output()?;
    k.long("n".to_string(), n as i64);
    k.long("m".to_string(), m as i64);
    let total = checked_product("the solve count", &[batch, m])?;
    k.long("a_rs".to_string(), a_rs as i64);
    k.long("a_cs".to_string(), a_cs as i64);
    k.long("sx".to_string(), sx as i64);
    k.long("so".to_string(), so as i64);
    k.long("lower".to_string(), lower_eff as i64);

    let entry = format!("ts_{}{}", instr.out.name(), if conj { "_conj" } else { "" });
    let helpers = arith.helpers(&["sub", "mul", "div", "conj"]);
    k.finish_over(caps, entry, &helpers, &body, total)
        .map(Plan::single)
}

/// No attrs. One group per matrix.
fn cholesky(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    check_operands(instr, 1)?;
    let arith = Arith::new(instr)?;
    let a = &instr.operands[0];
    let (batch, n) = square_batch(&instr.op, &a.shape)?;
    if a.shape != instr.out_shape {
        return Err("cholesky: the output shape must equal the input shape".to_string());
    }
    let wg = limits.group_size(0);
    let t = arith.t();
    let l = |idx: &str| arith.load(&format!("out[base + {}]", idx));
    let put = |idx: &str, expr: &str| format!("out[base + {}] = {};", idx, arith.store(expr));
    let pivot = l("k * n + k")?;
    let copy = arith.load("in0[base + idx]")?;
    let body = format!(
        "  long lid = (long)get_local_id(0);
  long base = (long)get_group_id(0) * n * n;
  for (long idx = lid; idx < n * n; idx += {wg}) {{
    long i = idx / n;
    long j = idx % n;
    out[base + idx] = j <= i ? {copy_store} : {zero_store};
  }}
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (long k = 0; k < n; k++) {{
    if (lid == 0) {{
      {t} s = {sqrt};
      {put_pivot}
    }}
    barrier(CLK_GLOBAL_MEM_FENCE);
    {t} d = {pivot};
    for (long i = k + 1 + lid; i < n; i += {wg}) {{
      {put_col}
    }}
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (long i = k + 1 + lid; i < n; i += {wg}) {{
      {t} lik = {lik};
      for (long j = k + 1; j <= i; j++) {{
        {t} ljk = {ljk};
        {put_upd}
      }}
    }}
    barrier(CLK_GLOBAL_MEM_FENCE);
  }}
",
        copy_store = arith.store(&copy),
        zero_store = arith.store(&arith.zero()),
        sqrt = arith.sqrt(&pivot),
        put_pivot = put("k * n + k", "s"),
        put_col = put("i * n + k", &arith.div(&l("i * n + k")?, "d")),
        lik = l("i * n + k")?,
        ljk = l("j * n + k")?,
        put_upd = put(
            "i * n + j",
            &arith.sub(&l("i * n + j")?, &arith.mul("lik", &arith.conj("ljk")))
        ),
    );
    let mut k = Kernel::new(instr);
    k.operand(0)?;
    k.output()?;
    k.long("n".to_string(), n as i64);

    let entry = format!("chol_{}", instr.out.name());
    let helpers = arith.helpers(&["sub", "mul", "div", "conj", "sqrt"]);
    let pragma_text = pragmas(&instr.dtypes(), caps)?;
    let head = format!("{pragma_text}{helpers}");
    let global = checked_product("the cholesky launch size", &[batch.max(1), wg])?;
    Ok(Plan::single(k.into_spec(
        &head,
        entry,
        &body,
        vec![global],
        vec![wg],
    )))
}

pub fn generate(instr: &Instr, caps: &Caps, limits: &Limits) -> Result<Plan, String> {
    match instr.op.as_str() {
        "triangular_solve" => triangular_solve(instr, caps),
        "cholesky" => cholesky(instr, caps, limits),
        other => Err(format!("no linalg codegen for op {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::plan;
    use super::super::types::Kind;
    use super::super::{DType, KernelSpec, Operand};
    use super::*;

    const F32: DType = DType::new(Kind::F, 32);
    const C64: DType = DType::new(Kind::C, 64);
    const S32: DType = DType::new(Kind::S, 32);

    fn operand(dtype: DType, shape: &[usize]) -> Operand {
        Operand {
            dtype,
            shape: shape.to_vec(),
            constant: None,
        }
    }

    fn ts(dtype: DType, a: &[usize], b: &[usize], attrs: (i64, i64, &str)) -> Instr {
        Instr {
            op: "triangular_solve".to_string(),
            operands: vec![operand(dtype, a), operand(dtype, b)],
            out: dtype,
            out_shape: b.to_vec(),
            attrs: vec![
                Attr::Int(attrs.0),
                Attr::Int(attrs.1),
                Attr::Atom(attrs.2.to_string()),
            ],
        }
    }

    fn chol(dtype: DType, shape: &[usize]) -> Instr {
        Instr {
            op: "cholesky".to_string(),
            operands: vec![operand(dtype, shape)],
            out: dtype,
            out_shape: shape.to_vec(),
            attrs: vec![],
        }
    }

    fn scalars(spec: &KernelSpec) -> Vec<i64> {
        spec.args
            .iter()
            .filter_map(|a| match a {
                super::super::KernelArg::Scalar(b) => {
                    Some(i64::from_ne_bytes(b.as_slice().try_into().ok()?))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_solve_is_one_launch_with_one_item_per_column() {
        let p = plan(&ts(F32, &[3, 5, 5], &[3, 5, 4], (1, 1, "none")));
        assert_eq!(p.passes.len(), 1);
        assert_eq!(p.passes[0].global, vec![12]);
        // n, m, a_rs, a_cs, sx, so, lower, then the guard's total.
        assert_eq!(scalars(&p.passes[0]), vec![5, 4, 5, 1, 4, 1, 1, 12]);
    }

    #[test]
    fn the_right_side_and_a_transpose_swap_the_strides() {
        // X A = B with A lower is the left solve of A^T, which is upper.
        let p = plan(&ts(F32, &[5, 5], &[4, 5], (0, 1, "none")));
        assert_eq!(scalars(&p.passes[0]), vec![5, 4, 1, 5, 1, 5, 0, 4]);
        let p = plan(&ts(F32, &[5, 5], &[5, 4], (1, 1, "transpose")));
        assert_eq!(scalars(&p.passes[0]), vec![5, 4, 1, 5, 4, 1, 0, 4]);
    }

    #[test]
    fn a_vector_b_has_one_column_per_batch() {
        let p = plan(&ts(F32, &[2, 3, 3], &[2, 3], (1, 0, "none")));
        assert_eq!(p.passes[0].global, vec![2]);
        assert_eq!(scalars(&p.passes[0]), vec![3, 1, 3, 1, 1, 0, 0, 2]);
    }

    #[test]
    fn conjugate_changes_the_source_and_complex_pulls_in_helpers() {
        let a = plan(&ts(C64, &[3, 3], &[3, 2], (1, 1, "none")));
        let b = plan(&ts(C64, &[3, 3], &[3, 2], (1, 1, "conjugate")));
        assert_ne!(a.passes[0].source, b.passes[0].source);
        assert!(a.passes[0].source.contains("cf_div"));
        assert!(!plan(&ts(F32, &[3, 3], &[3], (1, 1, "none"))).passes[0]
            .source
            .contains("cf_"));
    }

    #[test]
    fn mismatched_shapes_and_types_are_errors() {
        let caps = Caps::all();
        let lim = Limits::apple();
        let bad = ts(F32, &[3, 4], &[3, 2], (1, 1, "none"));
        assert!(generate(&bad, &caps, &lim).unwrap_err().contains("square"));
        let bad = ts(F32, &[3, 3], &[4, 2], (1, 1, "none"));
        assert!(generate(&bad, &caps, &lim).unwrap_err().contains("match"));
        let bad = ts(S32, &[3, 3], &[3, 2], (1, 1, "none"));
        assert!(generate(&bad, &caps, &lim).unwrap_err().contains("float"));
        let bad = ts(F32, &[3, 3], &[3, 2], (1, 1, "other"));
        assert!(generate(&bad, &caps, &lim).is_err());
    }

    #[test]
    fn cholesky_runs_one_group_per_matrix() {
        let p = plan(&chol(F32, &[4, 7, 7]));
        assert_eq!(p.passes.len(), 1);
        assert_eq!(p.passes[0].local, vec![256]);
        assert_eq!(p.passes[0].global, vec![4 * 256]);
        assert!(!p.passes[0].source.contains("atomic"));
        let c = plan(&chol(C64, &[3, 3]));
        assert!(c.passes[0].source.contains("cf_conj"));
        assert!(generate(&chol(F32, &[3, 4]), &Caps::all(), &Limits::apple()).is_err());
    }
}
