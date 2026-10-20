//! Elementwise kernels: unary and binary ops, `as_type`, `select`, `clip`.
//!
//! One kernel per `{op, operand types, output type}`. Operands have the output
//! shape, except scalar constants, which are scalar arguments. Arithmetic follows
//! C99 on explicit casts: integers wrap (computed in an unsigned type of at least
//! 32 bits), shifts and division are guarded, and NaN rules use `isnan`-style
//! comparisons. The rule for each case is noted beside its template.

use std::collections::BTreeMap;

use super::complex;
use super::movement::{operand_expr, pragmas, Kernel};
use super::types::{load_as, math_of, type_info, DType, Half, Kind, TypeInfo, BF16_HELPERS};
use super::{checked_element_count, expect_operands, Caps, Instr, KernelSpec};

const UNARY_FLOAT_FNS: [&str; 21] = [
    "exp", "expm1", "log", "log1p", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
    "atan", "asinh", "acosh", "atanh", "sqrt", "rsqrt", "cbrt", "erf", "erfc",
];
const UNARY_OTHER: [&str; 13] = [
    "abs",
    "negate",
    "sign",
    "ceil",
    "floor",
    "round",
    "sigmoid",
    "erf_inv",
    "bitwise_not",
    "count_leading_zeros",
    "population_count",
    "is_nan",
    "is_infinity",
];
const BINARY_ARITH: [&str; 12] = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "quotient",
    "remainder",
    "pow",
    "atan2",
    "max",
    "min",
    "left_shift",
    "right_shift",
];
const BINARY_BITWISE: [&str; 3] = ["bitwise_and", "bitwise_or", "bitwise_xor"];
const COMPARE: [&str; 6] = [
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
];
const LOGICAL: [&str; 3] = ["logical_and", "logical_or", "logical_xor"];
/// Ops that have a complex form: `real`, `imag`, `conjugate` and `phase` exist
/// for real operands too. The rest of the complex ops reuse the names above.
const COMPLEX_PARTS: [&str; 4] = ["real", "imag", "conjugate", "phase"];
/// Unary ops whose complex form returns a complex value, and the preamble
/// helper each calls.
const COMPLEX_UNARY: [(&str, &str); 23] = [
    ("negate", "neg"),
    ("conjugate", "conj"),
    ("sign", "sign"),
    ("exp", "exp"),
    ("expm1", "expm1"),
    ("log", "log"),
    ("log1p", "log1p"),
    ("sqrt", "sqrt"),
    ("rsqrt", "rsqrt"),
    ("cbrt", "cbrt"),
    ("sin", "sin"),
    ("cos", "cos"),
    ("tan", "tan"),
    ("sinh", "sinh"),
    ("cosh", "cosh"),
    ("tanh", "tanh"),
    ("asin", "asin"),
    ("acos", "acos"),
    ("atan", "atan"),
    ("asinh", "asinh"),
    ("acosh", "acosh"),
    ("atanh", "atanh"),
    ("sigmoid", "sigmoid"),
];

/// The C operator for an op, in the tables below.
fn symbol(table: &[(&'static str, &'static str)], op: &str) -> &'static str {
    table
        .iter()
        .find(|(name, _)| *name == op)
        .map_or("", |(_, sym)| sym)
}

const ARITH_SYMS: [(&str, &str); 3] = [("add", "+"), ("subtract", "-"), ("multiply", "*")];
const BITWISE_SYMS: [(&str, &str); 3] = [
    ("bitwise_and", "&"),
    ("bitwise_or", "|"),
    ("bitwise_xor", "^"),
];
const COMPARE_SYMS: [(&str, &str); 6] = [
    ("equal", "=="),
    ("not_equal", "!="),
    ("greater", ">"),
    ("greater_equal", ">="),
    ("less", "<"),
    ("less_equal", "<="),
];
const LOGICAL_SYMS: [(&str, &str); 3] = [
    ("logical_and", "&&"),
    ("logical_or", "||"),
    ("logical_xor", "!="),
];

pub fn handles(op: &str) -> bool {
    UNARY_FLOAT_FNS.contains(&op)
        || UNARY_OTHER.contains(&op)
        || BINARY_ARITH.contains(&op)
        || BINARY_BITWISE.contains(&op)
        || COMPARE.contains(&op)
        || LOGICAL.contains(&op)
        || COMPLEX_PARTS.contains(&op)
        || matches!(op, "logical_not" | "as_type" | "select" | "clip")
}

/// Integer `pow` by repeated squaring, wrapping in the unsigned type. A
/// negative exponent gives the integer limit: 1 and -1 keep their value, the
/// rest go to 0.
fn pow_helper(t: &TypeInfo) -> String {
    let name = t.dtype.name();
    let (c, w) = (t.math, t.wide);
    let neg = if t.is_signed() {
        format!(
            "  if (e < 0) return base == 1 ? ({c})1 : (base == -1 ? ((e & 1) ? ({c})-1 : ({c})1) : ({c})0);\n"
        )
    } else {
        String::new()
    };
    format!(
        "static {c} pow_{name}({c} base, {c} e) {{\n{neg}  {w} r = 1; {w} x = ({w})base; {w} n = ({w})e;\n  while (n != 0) {{ if (n & 1) r *= x; x *= x; n >>= 1; }}\n  return ({c})r;\n}}\n"
    )
}

/// `erf_inv` by Giles' polynomial, then Newton steps on `erf`/`erfc`. `y`
/// outside [-1, 1] and NaN give NaN; `+-1` give `+-inf`.
fn erf_inv_helper(ft: &str) -> String {
    let (s, nan, inf, steps) = if ft == "double" {
        (
            "",
            "as_double(0x7ff8000000000000UL)",
            "as_double(0x7ff0000000000000UL)",
            3,
        )
    } else {
        ("f", "as_float(0x7fc00000)", "as_float(0x7f800000)", 1)
    };
    format!(
        "static {ft} erf_inv_{ft}({ft} y) {{
  {ft} ay = fabs(y);
  if (!(ay <= 1.0{s})) return {nan};
  if (ay == 1.0{s}) return y > 0 ? {inf} : -{inf};
  {ft} w = -log((1.0{s} - ay) * (1.0{s} + ay));
  {ft} p;
  if (w < 5.0{s}) {{
    w = w - 2.5{s};
    p = 2.81022636e-08{s};
    p = 3.43273939e-07{s} + p * w;
    p = -3.5233877e-06{s} + p * w;
    p = -4.39150654e-06{s} + p * w;
    p = 0.00021858087{s} + p * w;
    p = -0.00125372503{s} + p * w;
    p = -0.00417768164{s} + p * w;
    p = 0.246640727{s} + p * w;
    p = 1.50140941{s} + p * w;
  }} else {{
    w = sqrt(w) - 3.0{s};
    p = -0.000200214257{s};
    p = 0.000100950558{s} + p * w;
    p = 0.00134934322{s} + p * w;
    p = -0.00367342844{s} + p * w;
    p = 0.00573950773{s} + p * w;
    p = -0.0076224613{s} + p * w;
    p = 0.00943887047{s} + p * w;
    p = 1.00167406{s} + p * w;
    p = 2.83297682{s} + p * w;
  }}
  {ft} x = p * ay;
  for (int k = 0; k < {steps}; k++) {{
    {ft} err = ay > 0.5{s} ? ((1.0{s} - ay) - erfc(x)) : (erf(x) - ay);
    x -= err / (1.1283791670955126{s} * exp(-x * x));
  }}
  return copysign(x, y);
}}
"
    )
}

/// Float to integer: NaN gives 0, infinities saturate, finite values truncate
/// toward zero and wrap to the width of the target (what `BinaryBackend` does).
fn f2i_helper(src: &TypeInfo, out: &TypeInfo) -> String {
    let (ft, s) = (src.math, if src.math == "double" { "" } else { "f" });
    let (c, name) = (out.math, out.dtype.name());
    format!(
        "static {c} f2i_{ft}_{name}({ft} x) {{
  if (x != x) return ({c})0;
  if (isinf(x)) return x > 0 ? ({c}){max} : ({c}){min};
  x = trunc(x);
  if (fabs(x) < 9.223372036854775808e18{s}) return ({c})(long)x;
  {ft} r = fmod(x, 1.8446744073709552e19{s});
  if (r < 0) r += 1.8446744073709552e19{s};
  return ({c})(ulong)r;
}}
",
        max = out.max,
        min = out.min
    )
}

fn float_lit(m: &TypeInfo, text: &str) -> String {
    if m.math == "double" {
        text.to_string()
    } else {
        format!("{}f", text)
    }
}

struct Gen<'a> {
    instr: &'a Instr,
    helpers: BTreeMap<String, String>,
    locals: Vec<String>,
    /// Complex helpers the body calls, per width prefix.
    cfns: BTreeMap<&'static str, (TypeInfo, Vec<String>)>,
}

impl<'a> Gen<'a> {
    fn operand_info(&self, k: usize) -> Result<TypeInfo, String> {
        let operand = self
            .instr
            .operands
            .get(k)
            .ok_or_else(|| format!("{} is missing operand {}", self.instr.op, k))?;
        type_info(operand.dtype)
    }

    fn need_half(&mut self, half: Option<Half>) {
        if let Some(Half::Bf16) = half {
            self.helpers
                .insert("a_bf16".to_string(), BF16_HELPERS.to_string());
        }
    }

    /// The value of operand `k` as a C expression of the math type `to`.
    fn load(&mut self, k: usize, to: &TypeInfo) -> Result<String, String> {
        let info = self.operand_info(k)?;
        let raw = operand_expr(self.instr, k, "i");
        self.need_half(info.half);
        load_as(&info, &raw, to).map_err(|e| format!("{}: {}", self.instr.op, e))
    }

    fn operand(&mut self, k: usize, to: &TypeInfo) -> Result<String, String> {
        let value = self.load(k, to)?;
        self.locals
            .push(format!("  {} x{} = {};", to.math, k, value));
        Ok(format!("x{}", k))
    }

    fn store(&mut self, expr: &str, from: &TypeInfo) -> Result<String, String> {
        let out = type_info(self.instr.out)?;
        self.need_half(out.half);
        Ok(match out.half {
            Some(Half::Bf16) => format!("f32_to_bf16((float)({}))", expr),
            None if out.storage == "half" => format!("(half)({})", expr),
            None if from.math == out.math => expr.to_string(),
            None => format!("({})({})", out.math, expr),
        })
    }

    /// A C comparison `a sym b` that is exact for any pair of operand types.
    /// Floats compare as `float` (`double` if either is). Integers compare as
    /// `long` or `ulong`; when one is signed and the other unsigned, a sign
    /// check comes first, so a large unsigned value never reads as negative.
    fn compare(sym: &str, a: &str, ta: &TypeInfo, b: &str, tb: &TypeInfo) -> String {
        if ta.math == tb.math {
            return format!("({a} {sym} {b})");
        }
        if ta.is_float() || tb.is_float() {
            let fm = if ta.math == "double" || tb.math == "double" {
                "double"
            } else {
                "float"
            };
            return format!("(({fm}){a} {sym} ({fm}){b})");
        }
        match (ta.is_signed(), tb.is_signed()) {
            (true, true) => format!("((long){a} {sym} (long){b})"),
            (false, false) => format!("((ulong){a} {sym} (ulong){b})"),
            (true, false) => Self::mixed_sign_compare(sym, a, b),
            (false, true) => {
                let flipped = match sym {
                    "<" => ">",
                    "<=" => ">=",
                    ">" => "<",
                    ">=" => "<=",
                    other => other,
                };
                Self::mixed_sign_compare(flipped, b, a)
            }
        }
    }

    /// `a sym b` with `a` signed and `b` unsigned, both integers.
    fn mixed_sign_compare(sym: &str, a: &str, b: &str) -> String {
        match sym {
            "==" => format!("({a} >= 0 && (ulong){a} == (ulong){b})"),
            "!=" => format!("({a} < 0 || (ulong){a} != (ulong){b})"),
            "<" | "<=" => format!("({a} < 0 || (ulong){a} {sym} (ulong){b})"),
            _ => format!("({a} >= 0 && (ulong){a} {sym} (ulong){b})"),
        }
    }

    /// True when the output or an operand is complex, or the op is one of
    /// `real`, `imag`, `conjugate`, `phase`.
    fn is_complex(&self) -> Result<bool, String> {
        if COMPLEX_PARTS.contains(&self.instr.op.as_str()) {
            return Ok(true);
        }
        let mut any = type_info(self.instr.out)?.is_complex();
        for o in &self.instr.operands {
            any |= type_info(o.dtype)?.is_complex();
        }
        Ok(any)
    }

    /// The C name of helper `name` for the complex type `v`; the kernel gets
    /// the helper and what it calls.
    fn cfn(&mut self, v: &TypeInfo, name: &str) -> String {
        let entry = self
            .cfns
            .entry(complex::prefix(v))
            .or_insert_with(|| (*v, Vec::new()));
        entry.1.push(name.to_string());
        complex::fn_name(v, name)
    }

    /// The complex type a kernel computes in when the output is real: the
    /// widest of the operands (`c128` if any operand is `c128` or `f64`).
    fn complex_width(&self) -> Result<TypeInfo, String> {
        let mut wide = false;
        for o in &self.instr.operands {
            let t = type_info(o.dtype)?;
            wide |= t.dtype.bits == 128 || (t.math == "double" && !t.is_complex());
        }
        type_info(DType::new(Kind::C, if wide { 128 } else { 64 }))
    }

    /// The float type of one part of `v`.
    fn part_of(v: &TypeInfo) -> Result<TypeInfo, String> {
        type_info(DType::new(
            Kind::F,
            if v.component == "double" { 64 } else { 32 },
        ))
    }

    /// Declare local `x{k}` holding operand `k` as the complex type `v`. A real
    /// operand gets a zero imaginary part; a complex one of the other width
    /// converts part by part.
    fn complex_operand(&mut self, k: usize, v: &TypeInfo) -> Result<String, String> {
        let info = self.operand_info(k)?;
        let value = if info.is_complex() {
            let raw = operand_expr(self.instr, k, "i");
            if info.math == v.math {
                raw
            } else {
                format!("convert_{}({})", v.math, raw)
            }
        } else {
            let part = Self::part_of(v)?;
            let re = self.load(k, &part)?;
            format!("({})(({})({}), ({})0)", v.math, part.math, re, part.math)
        };
        self.locals
            .push(format!("  {} x{} = {};", v.math, k, value));
        Ok(format!("x{}", k))
    }

    /// The truth value of operand `k` as an `int` expression: a complex value
    /// is true when either part is not zero.
    fn truthy(&mut self, k: usize) -> Result<String, String> {
        let info = self.operand_info(k)?;
        if info.is_complex() {
            let a = self.complex_operand(k, &info)?;
            let f = self.cfn(&info, "nonzero");
            return Ok(format!("{}({})", f, a));
        }
        let own = math_of(info.dtype)?;
        let a = self.operand(k, &own)?;
        Ok(format!("({} != 0)", a))
    }

    fn complex_body(&mut self) -> Result<(String, TypeInfo), String> {
        let op = self.instr.op.as_str().to_string();
        let op = op.as_str();
        let out = type_info(self.instr.out)?;
        let rejected = || format!("{} does not accept complex operands", op);

        // The output is real: `abs`, `real`, `imag`, tests, casts to real.
        if !out.is_complex() {
            let m = math_of(self.instr.out)?;
            let first = self
                .operand_info(0)
                .map_err(|_| format!("{} is missing operand 0", op))?;
            return match op {
                "abs" | "real" | "imag" | "phase" => {
                    expect_operands(self.instr, 1)?;
                    let v = if first.is_complex() {
                        first
                    } else {
                        self.complex_width()?
                    };
                    let a = self.complex_operand(0, &v)?;
                    let e = match op {
                        "abs" => format!("{}({})", self.cfn(&v, "abs"), a),
                        "real" => format!("{}.x", a),
                        "phase" => format!("atan2({a}.y, {a}.x)"),
                        _ => format!("{}.y", a),
                    };
                    Ok((format!("(({})({}))", m.math, e), m))
                }
                "is_nan" | "is_infinity" => {
                    expect_operands(self.instr, 1)?;
                    let a = self.complex_operand(0, &first)?;
                    let f = self.cfn(&first, if op == "is_nan" { "isnan" } else { "isinf" });
                    Ok((format!("(uchar)({}({}))", f, a), out))
                }
                "logical_not" => {
                    expect_operands(self.instr, 1)?;
                    let t = self.truthy(0)?;
                    Ok((format!("(uchar)(!{})", t), out))
                }
                "logical_and" | "logical_or" | "logical_xor" => {
                    expect_operands(self.instr, 2)?;
                    let (a, b) = (self.truthy(0)?, self.truthy(1)?);
                    let sym = symbol(&LOGICAL_SYMS, op);
                    Ok((format!("(uchar)({} {} {})", a, sym, b), out))
                }
                "equal" | "not_equal" => {
                    expect_operands(self.instr, 2)?;
                    let v = self.complex_width()?;
                    let (a, b) = (self.complex_operand(0, &v)?, self.complex_operand(1, &v)?);
                    let f = self.cfn(&v, if op == "equal" { "eq" } else { "ne" });
                    Ok((format!("(uchar)({}({}, {}))", f, a, b), out))
                }
                "select" => {
                    expect_operands(self.instr, 3)?;
                    let p = self.truthy(0)?;
                    let (a, b) = (self.operand(1, &m)?, self.operand(2, &m)?);
                    Ok((format!("({} ? {} : {})", p, a, b), m))
                }
                "as_type" => {
                    expect_operands(self.instr, 1)?;
                    if !first.is_complex() {
                        return Err(rejected());
                    }
                    // The imaginary part is dropped; the real part casts as a float would.
                    let part = Self::part_of(&first)?;
                    let a = self.complex_operand(0, &first)?;
                    let x = format!("{}.x", a);
                    let e = if m.is_float() {
                        format!("({})({})", m.math, x)
                    } else {
                        self.helpers.insert(
                            format!("e_f2i_{}_{}", part.math, m.dtype.name()),
                            f2i_helper(&part, &m),
                        );
                        format!("f2i_{}_{}({})", part.math, m.dtype.name(), x)
                    };
                    Ok((e, m))
                }
                _ => Err(rejected()),
            };
        }

        // The output is complex: every operand converts to it.
        let v = out;
        match op {
            "as_type" => {
                expect_operands(self.instr, 1)?;
                let a = self.complex_operand(0, &v)?;
                Ok((a, v))
            }
            "add" | "subtract" | "multiply" | "divide" | "pow" => {
                expect_operands(self.instr, 2)?;
                let (a, b) = (self.complex_operand(0, &v)?, self.complex_operand(1, &v)?);
                let name = match op {
                    "add" => "add",
                    "subtract" => "sub",
                    "multiply" => "mul",
                    "divide" => "div",
                    _ => "pow",
                };
                Ok((format!("{}({}, {})", self.cfn(&v, name), a, b), v))
            }
            "select" => {
                expect_operands(self.instr, 3)?;
                let p = self.truthy(0)?;
                let (a, b) = (self.complex_operand(1, &v)?, self.complex_operand(2, &v)?);
                Ok((format!("({} ? {} : {})", p, a, b), v))
            }
            _ => match COMPLEX_UNARY.iter().find(|(name, _)| *name == op) {
                Some((_, helper)) => {
                    expect_operands(self.instr, 1)?;
                    let a = self.complex_operand(0, &v)?;
                    Ok((format!("{}({})", self.cfn(&v, helper), a), v))
                }
                None => Err(rejected()),
            },
        }
    }

    /// The result expression and the math type it has.
    fn body(&mut self) -> Result<(String, TypeInfo), String> {
        if self.is_complex()? {
            return self.complex_body();
        }
        let op = self.instr.op.as_str();
        let out = type_info(self.instr.out)?;
        let m = math_of(self.instr.out)?;
        if UNARY_FLOAT_FNS.contains(&op) || matches!(op, "sigmoid" | "erf_inv") {
            self.float_unary(op, m)
        } else if matches!(
            op,
            "abs"
                | "negate"
                | "sign"
                | "ceil"
                | "floor"
                | "round"
                | "bitwise_not"
                | "count_leading_zeros"
                | "population_count"
                | "is_nan"
                | "is_infinity"
                | "logical_not"
        ) {
            self.plain_unary(op, m, out)
        } else if matches!(
            op,
            "add"
                | "subtract"
                | "multiply"
                | "divide"
                | "quotient"
                | "remainder"
                | "pow"
                | "atan2"
                | "max"
                | "min"
                | "bitwise_and"
                | "bitwise_or"
                | "bitwise_xor"
                | "left_shift"
                | "right_shift"
        ) {
            self.binary_arith(op, m)
        } else {
            self.compare_select_cast(op, m, out)
        }
    }

    /// Float functions of one operand: the libm names, `sigmoid`, `erf_inv`.
    fn float_unary(&mut self, op: &str, m: TypeInfo) -> Result<(String, TypeInfo), String> {
        let t = m.math;
        if UNARY_FLOAT_FNS.contains(&op) {
            expect_operands(self.instr, 1)?;
            require_float(op, &m)?;
            let a = self.operand(0, &m)?;
            if op == "tanh" {
                // PoCL's libm returns 1 - 2^-24 for tanh(inf); +-1 is exact.
                let one = float_lit(&m, "1.0");
                return Ok((
                    format!("(isinf({a}) ? copysign({one}, {a}) : tanh({a}))"),
                    m,
                ));
            }
            return Ok((format!("{}({})", op, a), m));
        }

        match op {
            "sigmoid" => {
                expect_operands(self.instr, 1)?;
                require_float(op, &m)?;
                let a = self.operand(0, &m)?;
                let (one, zero) = (float_lit(&m, "1.0"), float_lit(&m, "0.0"));
                // Both branches keep exp's argument at or below 0, so large
                // negative inputs stay finite.
                Ok((
                    format!(
                        "({a} >= {zero} ? {one} / ({one} + exp(-{a})) : exp({a}) / ({one} + exp({a})))"
                    ),
                    m,
                ))
            }
            "erf_inv" => {
                expect_operands(self.instr, 1)?;
                require_float(op, &m)?;
                self.helpers
                    .insert(format!("d_erf_inv_{}", t), erf_inv_helper(t));
                let a = self.operand(0, &m)?;
                Ok((format!("erf_inv_{}({})", t, a), m))
            }
            other => Err(format!("no elementwise template for op {}", other)),
        }
    }

    /// `abs`, `negate`, `sign`, rounding, bit counts, the float tests and `logical_not`.
    fn plain_unary(
        &mut self,
        op: &str,
        m: TypeInfo,
        out: TypeInfo,
    ) -> Result<(String, TypeInfo), String> {
        let (t, w, u) = (m.math, m.wide, m.unsigned);
        match op {
            "abs"
            | "negate"
            | "sign"
            | "ceil"
            | "floor"
            | "round"
            | "bitwise_not"
            | "count_leading_zeros"
            | "population_count" => {
                expect_operands(self.instr, 1)?;
                let a = self.operand(0, &m)?;
                let e = if m.is_float() {
                    match op {
                        "abs" => format!("fabs({})", a),
                        "negate" => format!("(-{})", a),
                        // NaN and signed zero fall through to `a` itself.
                        "sign" => format!(
                            "({a} > {zero} ? {one} : ({a} < {zero} ? -{one} : {a}))",
                            zero = float_lit(&m, "0.0"),
                            one = float_lit(&m, "1.0")
                        ),
                        "ceil" => format!("ceil({})", a),
                        "floor" => format!("floor({})", a),
                        "round" => format!("round({})", a),
                        _ => return Err(format!("{} needs an integer type", op)),
                    }
                } else {
                    match op {
                        "abs" if m.is_signed() => {
                            format!("({a} < 0 ? ({t})(({w})0 - ({w}){a}) : {a})")
                        }
                        "abs" | "ceil" | "floor" | "round" => a,
                        "negate" => format!("({t})(({w})0 - ({w}){a})"),
                        "sign" if m.is_signed() => format!("({t})(({a} > 0) - ({a} < 0))"),
                        "sign" => format!("({t})({a} > 0)"),
                        "bitwise_not" => format!("({t})(~{a})"),
                        "count_leading_zeros" => format!("({t})clz(({u}){a})"),
                        _ => format!("({t})popcount(({u}){a})"),
                    }
                };
                Ok((e, m))
            }
            "is_nan" | "is_infinity" => {
                expect_operands(self.instr, 1)?;
                let own = math_of(self.instr.operands[0].dtype)?;
                let a = self.operand(0, &own)?;
                let e = match (own.is_float(), op) {
                    (false, _) => "0".to_string(),
                    (true, "is_nan") => format!("isnan({})", a),
                    (true, _) => format!("isinf({})", a),
                };
                Ok((format!("(uchar)({})", e), out))
            }
            "logical_not" => {
                expect_operands(self.instr, 1)?;
                let own = math_of(self.instr.operands[0].dtype)?;
                let a = self.operand(0, &own)?;
                Ok((format!("(uchar)({} == 0)", a), out))
            }
            other => Err(format!("no elementwise template for op {}", other)),
        }
    }

    /// Arithmetic, bitwise and shift ops of two operands.
    fn binary_arith(&mut self, op: &str, m: TypeInfo) -> Result<(String, TypeInfo), String> {
        let (t, w, u) = (m.math, m.wide, m.unsigned);
        let bits = m.math_bits;
        match op {
            "add" | "subtract" | "multiply" => {
                expect_operands(self.instr, 2)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let sym = symbol(&ARITH_SYMS, op);
                // Integers wrap: the operation runs in an unsigned type of at
                // least 32 bits and the cast keeps the low bits.
                let e = if m.is_float() {
                    format!("({a} {sym} {b})")
                } else {
                    format!("({t})(({w}){a} {sym} ({w}){b})")
                };
                Ok((e, m))
            }
            "divide" => {
                expect_operands(self.instr, 2)?;
                require_float(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                Ok((format!("({a} / {b})"), m))
            }
            "quotient" => {
                expect_operands(self.instr, 2)?;
                require_int(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                // Division by zero gives all ones. MIN / -1 wraps to MIN.
                let e = if m.is_signed() {
                    format!(
                        "({b} == 0 ? ({t})-1 : ({b} == ({t})-1 ? ({t})(({w})0 - ({w}){a}) : ({t})({a} / {b})))"
                    )
                } else {
                    format!("({b} == 0 ? ({t})(~({w})0) : ({t})({a} / {b}))")
                };
                Ok((e, m))
            }
            "remainder" => {
                expect_operands(self.instr, 2)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                // The result takes the sign of the dividend. Integer
                // remainder by zero returns the dividend; by -1 it is 0.
                let e = if m.is_float() {
                    format!("fmod({a}, {b})")
                } else if m.is_signed() {
                    format!("({b} == 0 ? {a} : ({b} == ({t})-1 ? ({t})0 : ({t})({a} % {b})))")
                } else {
                    format!("({b} == 0 ? {a} : ({t})({a} % {b}))")
                };
                Ok((e, m))
            }
            "pow" => {
                expect_operands(self.instr, 2)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let e = if m.is_float() {
                    // `pow`, not `powr`: negative bases and 0^0 follow C99.
                    format!("pow({a}, {b})")
                } else {
                    self.helpers
                        .insert(format!("c_pow_{}", m.dtype.name()), pow_helper(&m));
                    format!("pow_{}({a}, {b})", m.dtype.name())
                };
                Ok((e, m))
            }
            "atan2" => {
                expect_operands(self.instr, 2)?;
                require_float(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                Ok((format!("atan2({a}, {b})"), m))
            }
            "max" | "min" => {
                expect_operands(self.instr, 2)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let e = if m.is_float() {
                    // Either side NaN gives NaN.
                    format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : f{op}({a}, {b})))")
                } else if op == "max" {
                    format!("({a} > {b} ? {a} : {b})")
                } else {
                    format!("({a} < {b} ? {a} : {b})")
                };
                Ok((e, m))
            }
            "bitwise_and" | "bitwise_or" | "bitwise_xor" => {
                expect_operands(self.instr, 2)?;
                require_int(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let sym = symbol(&BITWISE_SYMS, op);
                Ok((format!("({t})({a} {sym} {b})"), m))
            }
            "left_shift" | "right_shift" => {
                expect_operands(self.instr, 2)?;
                require_int(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                // A count at or above the width gives 0 (left, and right on
                // unsigned) or the sign fill (right, signed). A negative
                // count reads as a large unsigned one.
                let count = format!("(({u}){b})");
                let e = if op == "left_shift" {
                    format!("({count} >= {bits} ? ({t})0 : ({t})(({w}){a} << {count}))")
                } else if m.is_signed() {
                    // Arithmetic shift written without relying on the sign
                    // handling of `>>`: shift the complement of a negative
                    // value, then complement back.
                    format!(
                        "({count} >= {bits} ? ({a} < 0 ? ({t})-1 : ({t})0) : ({a} < 0 ? ({t})(~(({w})(~{a}) >> {count})) : ({t})(({w}){a} >> {count})))"
                    )
                } else {
                    format!("({count} >= {bits} ? ({t})0 : ({t})(({w}){a} >> {count}))")
                };
                Ok((e, m))
            }
            other => Err(format!("no elementwise template for op {}", other)),
        }
    }

    /// Comparisons, logical ops, `select`, `clip` and `as_type`.
    fn compare_select_cast(
        &mut self,
        op: &str,
        m: TypeInfo,
        out: TypeInfo,
    ) -> Result<(String, TypeInfo), String> {
        let t = m.math;
        match op {
            "equal" | "not_equal" | "greater" | "greater_equal" | "less" | "less_equal" => {
                expect_operands(self.instr, 2)?;
                let (own_a, own_b) = (
                    math_of(self.instr.operands[0].dtype)?,
                    math_of(self.instr.operands[1].dtype)?,
                );
                let (a, b) = (self.operand(0, &own_a)?, self.operand(1, &own_b)?);
                let sym = symbol(&COMPARE_SYMS, op);
                // C comparisons are false for NaN, except `!=`: the same rule
                // as Nx. Each operand keeps its own type; see `compare`.
                Ok((
                    format!("(uchar){}", Self::compare(sym, &a, &own_a, &b, &own_b)),
                    out,
                ))
            }
            "logical_and" | "logical_or" | "logical_xor" => {
                expect_operands(self.instr, 2)?;
                let (own_a, own_b) = (
                    math_of(self.instr.operands[0].dtype)?,
                    math_of(self.instr.operands[1].dtype)?,
                );
                let (a, b) = (self.operand(0, &own_a)?, self.operand(1, &own_b)?);
                let sym = symbol(&LOGICAL_SYMS, op);
                Ok((format!("(uchar)(({a} != 0) {sym} ({b} != 0))"), out))
            }
            "select" => {
                expect_operands(self.instr, 3)?;
                let own = math_of(self.instr.operands[0].dtype)?;
                let p = self.operand(0, &own)?;
                let (a, b) = (self.operand(1, &m)?, self.operand(2, &m)?);
                Ok((format!("({p} != 0 ? {a} : {b})"), m))
            }
            "clip" => {
                expect_operands(self.instr, 3)?;
                let own: Vec<TypeInfo> = self
                    .instr
                    .operands
                    .iter()
                    .map(|o| math_of(o.dtype))
                    .collect::<Result<_, _>>()?;
                let (x, lo, hi) = (
                    self.operand(0, &own[0])?,
                    self.operand(1, &own[1])?,
                    self.operand(2, &own[2])?,
                );
                // Same steps as `BinaryBackend`: raise to `lo`, then cap at
                // `hi`. A NaN input therefore becomes `lo`. The comparisons
                // use the operands' own types, so they stay exact.
                let above_lo = Self::compare(">", &x, &own[0], &lo, &own[1]);
                let below_hi = Self::compare("<", &x, &own[0], &hi, &own[2]);
                let lo_below_hi = Self::compare("<", &lo, &own[1], &hi, &own[2]);
                self.locals.push(format!("  int above_lo = {};", above_lo));
                self.locals.push(format!(
                    "  int below_hi = above_lo ? {} : {};",
                    below_hi, lo_below_hi
                ));
                Ok((
                    format!("(below_hi ? (above_lo ? ({t}){x} : ({t}){lo}) : ({t}){hi})"),
                    m,
                ))
            }
            "as_type" => {
                expect_operands(self.instr, 1)?;
                let src = type_info(self.instr.operands[0].dtype)?;
                let own = math_of(src.dtype)?;
                let a = self.operand(0, &own)?;
                let e = match (own.is_float(), m.is_float()) {
                    (true, false) => {
                        self.helpers.insert(
                            format!("e_f2i_{}_{}", own.math, m.dtype.name()),
                            f2i_helper(&own, &m),
                        );
                        format!("f2i_{}_{}({})", own.math, m.dtype.name(), a)
                    }
                    _ => format!("({})({})", t, a),
                };
                Ok((e, m))
            }
            other => Err(format!("no elementwise template for op {}", other)),
        }
    }
}

/// Binary ops a scalar combiner may use: the result has the operand type.
pub(super) const COMBINER_OPS: [&str; 15] = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "quotient",
    "remainder",
    "pow",
    "atan2",
    "max",
    "min",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
];

/// Source for one scalar function built from an elementwise template: the
/// helpers it calls, the complex helpers it needs per width, and the
/// function text. Several can merge into one kernel without duplicate helpers.
#[derive(Default)]
pub struct ScalarCode {
    pub helpers: BTreeMap<String, String>,
    pub cfns: BTreeMap<&'static str, (TypeInfo, Vec<String>)>,
    pub funcs: String,
}

impl ScalarCode {
    pub fn merge(&mut self, other: ScalarCode) {
        self.helpers.extend(other.helpers);
        for (prefix, (info, names)) in other.cfns {
            let entry = self
                .cfns
                .entry(prefix)
                .or_insert_with(|| (info, Vec::new()));
            entry.1.extend(names);
        }
        self.funcs.push_str(&other.funcs);
    }

    /// Complex preamble, helpers, then the functions.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (info, names) in self.cfns.values() {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            out.push_str(&complex::preamble(info, &names));
        }
        for h in self.helpers.values() {
            out.push_str(h);
        }
        out.push_str(&self.funcs);
        out
    }
}

/// The scalar function `static T name(T in0, T in1)` that applies binary op
/// `op` to two values of `dtype`, with the same arithmetic as the elementwise
/// kernel for `op`. `dtype` is a math type: `bf16` and `f16` callers pass `f32`.
pub(super) fn scalar_binary(op: &str, dtype: DType, name: &str) -> Result<ScalarCode, String> {
    if !COMBINER_OPS.contains(&op) {
        return Err(format!("{} is not a binary op a combiner can use", op));
    }
    let info = type_info(dtype)?;
    let operand = super::Operand {
        dtype,
        shape: Vec::new(),
        constant: Some(Vec::new()),
    };
    let instr = Instr {
        op: op.to_string(),
        operands: vec![operand.clone(), operand],
        out: dtype,
        out_shape: Vec::new(),
        attrs: Vec::new(),
    };
    let mut gen = Gen {
        instr: &instr,
        helpers: BTreeMap::new(),
        locals: Vec::new(),
        cfns: BTreeMap::new(),
    };
    let (expr, compute) = gen.body()?;
    let result = gen.store(&expr, &compute)?;
    let t = info.storage;
    let locals = gen.locals.join("\n");
    Ok(ScalarCode {
        funcs: format!(
            "static {t} {name}({t} in0, {t} in1) {{\n{locals}\n  return {result};\n}}\n"
        ),
        helpers: gen.helpers,
        cfns: gen.cfns,
    })
}

fn require_float(op: &str, m: &TypeInfo) -> Result<(), String> {
    if m.is_float() {
        Ok(())
    } else {
        Err(format!("{} needs a float type, got {}", op, m.dtype))
    }
}

fn require_int(op: &str, m: &TypeInfo) -> Result<(), String> {
    if m.is_float() {
        Err(format!("{} needs an integer type, got {}", op, m.dtype))
    } else {
        Ok(())
    }
}

pub fn generate(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    // Every operand has the output shape, or is a scalar constant.
    for (k, operand) in instr.operands.iter().enumerate() {
        let scalar_constant = operand.constant.is_some() && operand.shape.is_empty();
        if operand.shape != instr.out_shape && !scalar_constant {
            return Err(format!(
                "{}: operand {} has shape {:?}, the output {:?}; every operand must match the output or be a scalar constant",
                instr.op, k, operand.shape, instr.out_shape
            ));
        }
    }

    let mut gen = Gen {
        instr,
        helpers: BTreeMap::new(),
        locals: Vec::new(),
        cfns: BTreeMap::new(),
    };
    let (expr, compute) = gen.body()?;
    let result = gen.store(&expr, &compute)?;

    // Extension check over every type the kernel touches, the compute type too.
    let mut dtypes = instr.dtypes();
    dtypes.push(compute.dtype);
    let pragma_text = pragmas(&dtypes, caps)?;

    let mut kernel = Kernel::new(instr);
    let mut names = Vec::new();
    for (k, operand) in instr.operands.iter().enumerate() {
        names.push(operand.dtype.name());
        kernel.operand(k)?;
    }
    kernel.output()?;

    let entry = format!("ew_{}_{}_{}", instr.op, names.join("_"), instr.out.name());
    // The complex helpers each width calls, after the pragmas that enable it.
    let mut preamble = String::new();
    for (info, names) in gen.cfns.values() {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        preamble.push_str(&complex::preamble(info, &names));
    }
    let helpers: String = gen.helpers.values().cloned().collect();
    let locals = gen.locals.join("\n");
    let head = format!("{pragma_text}{preamble}{helpers}");
    let body = format!("  size_t i = get_global_id(0);\n{locals}\n  out[i] = {result};\n");
    let total = checked_element_count(&instr.out_shape)?;
    kernel.seal(&head, entry, &body, total)
}

#[cfg(test)]
#[path = "tests/elementwise.rs"]
mod tests;
