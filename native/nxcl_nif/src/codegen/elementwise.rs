//! Elementwise kernels: unary and binary ops, `as_type`, `select`, `clip`.
//!
//! One kernel per `{op, operand types, output type}`, written from the
//! templates below. Operands have the output shape, except scalar constants,
//! which the kernel takes as scalar arguments. Arithmetic follows C99 on
//! explicit casts: integers wrap (computed in an unsigned type of at least 32
//! bits), shifts and division have guards, NaN rules come from `isnan`-style
//! comparisons. The rule for each case is noted beside its template.

use std::collections::BTreeMap;

use super::complex;
use super::types::{type_info, DType, Half, Kind, TypeInfo};
use super::{element_count, Caps, Instr, KernelArg, KernelSpec};

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

// ── C helpers ───────────────────────────────────────────

pub(super) const BF16_HELPERS: &str = "\
static float bf16_to_f32(ushort x) { return as_float(((uint)x) << 16); }
static ushort f32_to_bf16(float f) {
  uint u = as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (ushort)(u >> 16);
}
";

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

// ── Generator ───────────────────────────────────────────

/// The type an expression computes in: `bf16` computes as `f32`.
fn math_of(dtype: DType) -> Result<TypeInfo, String> {
    let info = type_info(dtype)?;
    if info.half.is_some() {
        type_info(DType::new(Kind::F, 32))
    } else {
        Ok(info)
    }
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
        match half {
            Some(Half::Bf16) => {
                self.helpers
                    .insert("a_bf16".to_string(), BF16_HELPERS.to_string());
            }
            None => {}
        }
    }

    /// The value of operand `k` as a C expression of the math type `to`.
    fn load(&mut self, k: usize, to: &TypeInfo) -> Result<String, String> {
        let info = self.operand_info(k)?;
        let raw = if self.instr.operands[k].constant.is_some() {
            format!("in{}", k)
        } else {
            format!("in{}[i]", k)
        };
        self.need_half(info.half);
        let value = match info.half {
            Some(Half::Bf16) => format!("bf16_to_f32({})", raw),
            // `half` needs `cl_khr_fp16`; the cast reads it as the math type.
            None if info.storage == "half" => format!("(float)({})", raw),
            None => raw,
        };
        let own = math_of(info.dtype)?;
        if own.math == to.math {
            Ok(value)
        } else if own.is_float() && !to.is_float() {
            Err(format!(
                "{}: cannot read a {} operand as {}",
                self.instr.op, info.dtype, to.dtype
            ))
        } else {
            Ok(format!("(({})({}))", to.math, value))
        }
    }

    /// Declare a local `x{k}` holding operand `k` in the math type `to`.
    fn operand(&mut self, k: usize, to: &TypeInfo) -> Result<String, String> {
        let value = self.load(k, to)?;
        self.locals
            .push(format!("  {} x{} = {};", to.math, k, value));
        Ok(format!("x{}", k))
    }

    /// Wrap an expression of math type `from` for the output buffer.
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

    // ── Complex ──

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
            let raw = if self.instr.operands[k].constant.is_some() {
                format!("in{}", k)
            } else {
                format!("in{}[i]", k)
            };
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
                    self.expect(1)?;
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
                    self.expect(1)?;
                    let a = self.complex_operand(0, &first)?;
                    let f = self.cfn(&first, if op == "is_nan" { "isnan" } else { "isinf" });
                    Ok((format!("(uchar)({}({}))", f, a), out))
                }
                "logical_not" => {
                    self.expect(1)?;
                    let t = self.truthy(0)?;
                    Ok((format!("(uchar)(!{})", t), out))
                }
                "logical_and" | "logical_or" | "logical_xor" => {
                    self.expect(2)?;
                    let (a, b) = (self.truthy(0)?, self.truthy(1)?);
                    let sym = match op {
                        "logical_and" => "&&",
                        "logical_or" => "||",
                        _ => "!=",
                    };
                    Ok((format!("(uchar)({} {} {})", a, sym, b), out))
                }
                "equal" | "not_equal" => {
                    self.expect(2)?;
                    let v = self.complex_width()?;
                    let (a, b) = (self.complex_operand(0, &v)?, self.complex_operand(1, &v)?);
                    let f = self.cfn(&v, if op == "equal" { "eq" } else { "ne" });
                    Ok((format!("(uchar)({}({}, {}))", f, a, b), out))
                }
                "select" => {
                    self.expect(3)?;
                    let p = self.truthy(0)?;
                    let (a, b) = (self.operand(1, &m)?, self.operand(2, &m)?);
                    Ok((format!("({} ? {} : {})", p, a, b), m))
                }
                "as_type" => {
                    self.expect(1)?;
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
                self.expect(1)?;
                let a = self.complex_operand(0, &v)?;
                Ok((a, v))
            }
            "add" | "subtract" | "multiply" | "divide" | "pow" => {
                self.expect(2)?;
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
                self.expect(3)?;
                let p = self.truthy(0)?;
                let (a, b) = (self.complex_operand(1, &v)?, self.complex_operand(2, &v)?);
                Ok((format!("({} ? {} : {})", p, a, b), v))
            }
            _ => match COMPLEX_UNARY.iter().find(|(name, _)| *name == op) {
                Some((_, helper)) => {
                    self.expect(1)?;
                    let a = self.complex_operand(0, &v)?;
                    Ok((format!("{}({})", self.cfn(&v, helper), a), v))
                }
                None => Err(rejected()),
            },
        }
    }

    fn expect(&self, n: usize) -> Result<(), String> {
        if self.instr.operands.len() == n {
            Ok(())
        } else {
            Err(format!(
                "{} takes {} operands, got {}",
                self.instr.op,
                n,
                self.instr.operands.len()
            ))
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
        let (t, w, u) = (m.math, m.wide, m.unsigned);
        let bits = m.math_bits;

        if UNARY_FLOAT_FNS.contains(&op) {
            self.expect(1)?;
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
                self.expect(1)?;
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
                self.expect(1)?;
                require_float(op, &m)?;
                self.helpers
                    .insert(format!("d_erf_inv_{}", t), erf_inv_helper(t));
                let a = self.operand(0, &m)?;
                Ok((format!("erf_inv_{}({})", t, a), m))
            }
            "abs"
            | "negate"
            | "sign"
            | "ceil"
            | "floor"
            | "round"
            | "bitwise_not"
            | "count_leading_zeros"
            | "population_count" => {
                self.expect(1)?;
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
                self.expect(1)?;
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
                self.expect(1)?;
                let own = math_of(self.instr.operands[0].dtype)?;
                let a = self.operand(0, &own)?;
                Ok((format!("(uchar)({} == 0)", a), out))
            }
            "add" | "subtract" | "multiply" => {
                self.expect(2)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let sym = match op {
                    "add" => "+",
                    "subtract" => "-",
                    _ => "*",
                };
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
                self.expect(2)?;
                require_float(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                Ok((format!("({a} / {b})"), m))
            }
            "quotient" => {
                self.expect(2)?;
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
                self.expect(2)?;
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
                self.expect(2)?;
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
                self.expect(2)?;
                require_float(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                Ok((format!("atan2({a}, {b})"), m))
            }
            "max" | "min" => {
                self.expect(2)?;
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
                self.expect(2)?;
                require_int(op, &m)?;
                let (a, b) = (self.operand(0, &m)?, self.operand(1, &m)?);
                let sym = match op {
                    "bitwise_and" => "&",
                    "bitwise_or" => "|",
                    _ => "^",
                };
                Ok((format!("({t})({a} {sym} {b})"), m))
            }
            "left_shift" | "right_shift" => {
                self.expect(2)?;
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
            "equal" | "not_equal" | "greater" | "greater_equal" | "less" | "less_equal" => {
                self.expect(2)?;
                let (own_a, own_b) = (
                    math_of(self.instr.operands[0].dtype)?,
                    math_of(self.instr.operands[1].dtype)?,
                );
                let (a, b) = (self.operand(0, &own_a)?, self.operand(1, &own_b)?);
                let sym = match op {
                    "equal" => "==",
                    "not_equal" => "!=",
                    "greater" => ">",
                    "greater_equal" => ">=",
                    "less" => "<",
                    _ => "<=",
                };
                // C comparisons are false for NaN, except `!=`: the same rule
                // as Nx. Each operand keeps its own type; see `compare`.
                Ok((
                    format!("(uchar){}", Self::compare(sym, &a, &own_a, &b, &own_b)),
                    out,
                ))
            }
            "logical_and" | "logical_or" | "logical_xor" => {
                self.expect(2)?;
                let (own_a, own_b) = (
                    math_of(self.instr.operands[0].dtype)?,
                    math_of(self.instr.operands[1].dtype)?,
                );
                let (a, b) = (self.operand(0, &own_a)?, self.operand(1, &own_b)?);
                let sym = match op {
                    "logical_and" => "&&",
                    "logical_or" => "||",
                    _ => "!=",
                };
                Ok((format!("(uchar)(({a} != 0) {sym} ({b} != 0))"), out))
            }
            "select" => {
                self.expect(3)?;
                let own = math_of(self.instr.operands[0].dtype)?;
                let p = self.operand(0, &own)?;
                let (a, b) = (self.operand(1, &m)?, self.operand(2, &m)?);
                Ok((format!("({p} != 0 ? {a} : {b})"), m))
            }
            "clip" => {
                self.expect(3)?;
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
                self.expect(1)?;
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

/// Generate the kernel for an elementwise instruction.
pub fn generate(instr: &Instr, caps: &Caps) -> Result<KernelSpec, String> {
    // Every operand has the output shape, or is a scalar constant.
    for (k, operand) in instr.operands.iter().enumerate() {
        let scalar_constant = operand.constant.is_some() && operand.shape.is_empty();
        if operand.shape != instr.out_shape && !scalar_constant {
            return Err(format!(
                "{}: operand {} has shape {:?}, the output {:?}; broadcast is not lowered yet",
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

    // Extension check over every type the kernel touches.
    let mut dtypes: Vec<DType> = instr.operands.iter().map(|o| o.dtype).collect();
    dtypes.push(instr.out);
    dtypes.push(compute.dtype);
    let mut pragmas = String::new();
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
            if !pragmas.contains(&pragma) {
                pragmas.push_str(&pragma);
            }
        }
    }

    let mut params = Vec::new();
    let mut args = Vec::new();
    let mut names = Vec::new();
    for (k, operand) in instr.operands.iter().enumerate() {
        let storage = type_info(operand.dtype)?.storage;
        names.push(operand.dtype.name());
        match &operand.constant {
            Some(bytes) => {
                params.push(format!("const {} in{}", storage, k));
                args.push(KernelArg::Scalar(bytes.clone()));
            }
            None => {
                params.push(format!("__global const {}* in{}", storage, k));
                args.push(KernelArg::Operand(k));
            }
        }
    }
    let out_storage = type_info(instr.out)?.storage;
    params.push(format!("__global {}* out", out_storage));
    args.push(KernelArg::Output);

    let entry = format!("ew_{}_{}_{}", instr.op, names.join("_"), instr.out.name());
    // The complex helpers each width calls, after the pragmas that enable it.
    let mut preamble = String::new();
    for (info, names) in gen.cfns.values() {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        preamble.push_str(&complex::preamble(info, &names));
    }
    let helpers: String = gen.helpers.values().cloned().collect();
    let locals = gen.locals.join("\n");
    let source = format!(
        "{pragmas}{preamble}{helpers}__kernel void {entry}({params}) {{\n  size_t i = get_global_id(0);\n{locals}\n  out[i] = {result};\n}}\n",
        params = params.join(", ")
    );

    Ok(KernelSpec {
        source,
        entry,
        build_options: String::new(),
        global: vec![element_count(&instr.out_shape).max(1)],
        local: Vec::new(),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::super::types::{encode_const, supported, Class, ConstVal};
    use super::super::{Attr, Operand};
    use super::*;

    const U8: DType = DType::new(Kind::U, 8);
    const S8: DType = DType::new(Kind::S, 8);
    const S32: DType = DType::new(Kind::S, 32);
    const F32: DType = DType::new(Kind::F, 32);
    const F64: DType = DType::new(Kind::F, 64);
    const BF16: DType = DType::new(Kind::Bf, 16);
    const F16: DType = DType::new(Kind::F, 16);

    fn buf(dtype: DType) -> Operand {
        Operand {
            dtype,
            shape: vec![4],
            constant: None,
        }
    }

    fn constant(dtype: DType, value: ConstVal) -> Operand {
        Operand {
            dtype,
            shape: vec![],
            constant: Some(encode_const(dtype, value).unwrap()),
        }
    }

    fn instr(op: &str, operands: Vec<Operand>, out: DType, attrs: Vec<Attr>) -> Instr {
        Instr {
            op: op.to_string(),
            operands,
            out,
            out_shape: vec![4],
            attrs,
        }
    }

    fn src(op: &str, ins: &[DType], out: DType) -> String {
        let operands = ins.iter().map(|&d| buf(d)).collect();
        generate(&instr(op, operands, out, vec![]), &Caps::all())
            .unwrap_or_else(|e| panic!("{} {:?}: {}", op, ins, e))
            .source
    }

    fn compare_src(op: &str, a: DType, b: DType) -> String {
        src(op, &[a, b], U8)
    }

    // ── Wrap, overflow ──

    #[test]
    fn u8_add_wraps_through_an_explicit_cast() {
        let s = src("add", &[U8, U8], U8);
        assert!(s.contains("(uchar)((uint)x0 + (uint)x1)"), "{}", s);
    }

    #[test]
    fn s32_add_wraps_through_unsigned() {
        let s = src("add", &[S32, S32], S32);
        assert!(s.contains("(int)((uint)x0 + (uint)x1)"), "{}", s);
        let s = src(
            "multiply",
            &[DType::new(Kind::S, 64); 2],
            DType::new(Kind::S, 64),
        );
        assert!(s.contains("(long)((ulong)x0 * (ulong)x1)"), "{}", s);
    }

    #[test]
    fn u16_multiply_runs_in_unsigned_not_promoted_int() {
        let u16t = DType::new(Kind::U, 16);
        let s = src("multiply", &[u16t, u16t], u16t);
        assert!(s.contains("(ushort)((uint)x0 * (uint)x1)"), "{}", s);
    }

    #[test]
    fn negate_and_abs_of_integers_wrap() {
        assert!(src("negate", &[U8], U8).contains("(uchar)((uint)0 - (uint)x0)"));
        let s = src("abs", &[S8], S8);
        assert!(
            s.contains("x0 < 0 ? (char)((uint)0 - (uint)x0) : x0"),
            "{}",
            s
        );
        assert!(src("abs", &[U8], U8).contains("out[i] = x0;"));
    }

    // ── NaN ──

    #[test]
    fn float_max_and_min_return_nan_if_either_side_is_nan() {
        for op in ["max", "min"] {
            let s = src(op, &[F32, F32], F32);
            assert!(
                s.contains(&format!(
                    "x0 != x0 ? x0 : (x1 != x1 ? x1 : f{}(x0, x1))",
                    op
                )),
                "{}",
                s
            );
        }
        // Integers have no NaN: a plain compare.
        assert!(src("max", &[S32, S32], S32).contains("x0 > x1 ? x0 : x1"));
    }

    #[test]
    fn comparisons_use_c_operators_so_nan_is_false_except_not_equal() {
        let s = compare_src("equal", F32, F32);
        assert!(s.contains("(uchar)(x0 == x1)"), "{}", s);
        let s = compare_src("not_equal", F32, F32);
        assert!(s.contains("(uchar)(x0 != x1)"), "{}", s);
        let s = compare_src("less_equal", F32, F32);
        assert!(s.contains("(uchar)(x0 <= x1)"), "{}", s);
    }

    #[test]
    fn comparison_of_an_int_and_a_float_runs_in_float() {
        let s = compare_src("greater", S32, F32);
        assert!(s.contains("int x0 = in0[i];"), "{}", s);
        assert!(s.contains("(uchar)((float)x0 > (float)x1)"), "{}", s);
        let s = compare_src("less", S32, F64);
        assert!(s.contains("((double)x0 < (double)x1)"), "{}", s);
    }

    #[test]
    fn comparison_of_signed_and_unsigned_checks_the_sign_first() {
        let u64t = DType::new(Kind::U, 64);
        let s = compare_src("less", u64t, S32);
        // u64 < s32 reads as: the signed side is positive and the unsigned is smaller.
        assert!(s.contains("(x1 >= 0 && (ulong)x1 > (ulong)x0)"), "{}", s);
        let s = compare_src("less", S32, u64t);
        assert!(s.contains("(x0 < 0 || (ulong)x0 < (ulong)x1)"), "{}", s);
        let s = compare_src("equal", U8, S8);
        assert!(s.contains("(x1 >= 0 && (ulong)x1 == (ulong)x0)"), "{}", s);
        let s = compare_src("not_equal", S8, U8);
        assert!(s.contains("(x0 < 0 || (ulong)x0 != (ulong)x1)"), "{}", s);
        // Same signedness, different width: one wide type.
        let s = compare_src("less", S8, S32);
        assert!(s.contains("((long)x0 < (long)x1)"), "{}", s);
    }

    #[test]
    fn is_nan_uses_isnan_and_integers_are_never_nan() {
        let s = src("is_nan", &[F32], U8);
        assert!(s.contains("(uchar)(isnan(x0))"), "{}", s);
        assert!(src("is_nan", &[S32], U8).contains("(uchar)(0)"));
        assert!(src("is_infinity", &[F32], U8).contains("isinf(x0)"));
    }

    #[test]
    fn logical_ops_test_against_zero_and_nan_counts_as_true() {
        let s = src("logical_and", &[F32, S32], U8);
        assert!(s.contains("(uchar)((x0 != 0) && (x1 != 0))"), "{}", s);
        assert!(src("logical_xor", &[U8, U8], U8).contains("(x0 != 0) != (x1 != 0)"));
        assert!(src("logical_not", &[F32], U8).contains("(uchar)(x0 == 0)"));
    }

    #[test]
    fn tanh_of_an_infinity_is_exactly_one() {
        let s = src("tanh", &[F32], F32);
        assert!(
            s.contains("isinf(x0) ? copysign(1.0f, x0) : tanh(x0)"),
            "{}",
            s
        );
    }

    #[test]
    fn clip_of_u64_by_signed_bounds_compares_exactly() {
        let u64t = DType::new(Kind::U, 64);
        let s64 = DType::new(Kind::S, 64);
        let s = src("clip", &[u64t, S32, S32], s64);
        assert!(
            s.contains("int above_lo = (x1 < 0 || (ulong)x1 < (ulong)x0);"),
            "{}",
            s
        );
    }

    #[test]
    fn sign_keeps_nan_and_signed_zero() {
        let s = src("sign", &[F32], F32);
        assert!(
            s.contains("x0 > 0.0f ? 1.0f : (x0 < 0.0f ? -1.0f : x0)"),
            "{}",
            s
        );
        assert!(src("sign", &[S32], S32).contains("(int)((x0 > 0) - (x0 < 0))"));
        assert!(src("sign", &[U8], U8).contains("(uchar)(x0 > 0)"));
    }

    #[test]
    fn clip_follows_the_reference_so_nan_becomes_the_lower_bound() {
        let s = src("clip", &[F32, F32, F32], F32);
        assert!(s.contains("int above_lo = (x0 > x1);"), "{}", s);
        assert!(
            s.contains("int below_hi = above_lo ? (x0 < x2) : (x1 < x2);"),
            "{}",
            s
        );
        assert!(
            s.contains("out[i] = (below_hi ? (above_lo ? (float)x0 : (float)x1) : (float)x2);"),
            "{}",
            s
        );
    }

    // ── Zero divisor, integer division ──

    #[test]
    fn float_divide_is_plain_division_so_zero_gives_inf_and_nan() {
        let s = src("divide", &[F32, F32], F32);
        assert!(s.contains("out[i] = (x0 / x1);"), "{}", s);
        // Integer operands are converted to the float output type first.
        let s = src("divide", &[S32, S32], F32);
        assert!(s.contains("float x0 = ((float)(in0[i]));"), "{}", s);
    }

    #[test]
    fn integer_quotient_truncates_and_guards_zero_and_min_over_minus_one() {
        let s = src("quotient", &[S32, S32], S32);
        assert!(s.contains("x1 == 0 ? (int)-1"), "{}", s);
        assert!(
            s.contains("x1 == (int)-1 ? (int)((uint)0 - (uint)x0)"),
            "{}",
            s
        );
        assert!(s.contains("(int)(x0 / x1)"), "{}", s);
        let s = src("quotient", &[U8, U8], U8);
        assert!(s.contains("x1 == 0 ? (uchar)(~(uint)0)"), "{}", s);
    }

    #[test]
    fn integer_remainder_takes_the_dividend_sign_and_guards_zero() {
        let s = src("remainder", &[S32, S32], S32);
        assert!(s.contains("x1 == 0 ? x0"), "{}", s);
        assert!(s.contains("x1 == (int)-1 ? (int)0"), "{}", s);
        assert!(s.contains("(int)(x0 % x1)"), "{}", s);
        // Floats: fmod, which has the dividend's sign.
        assert!(src("remainder", &[F32, F32], F32).contains("fmod(x0, x1)"));
    }

    #[test]
    fn divide_and_quotient_reject_the_wrong_class() {
        let ops = |op: &str, d: DType| {
            generate(&instr(op, vec![buf(d), buf(d)], d, vec![]), &Caps::all())
        };
        assert!(ops("divide", S32).is_err());
        assert!(ops("quotient", F32).is_err());
    }

    // ── Shifts ──

    #[test]
    fn shift_counts_at_or_above_the_width_are_guarded() {
        let s = src("left_shift", &[U8, U8], U8);
        assert!(s.contains("((uchar)x1) >= 8 ? (uchar)0"), "{}", s);
        let s = src("left_shift", &[S32, S32], S32);
        assert!(s.contains("((uint)x1) >= 32 ? (int)0"), "{}", s);
        let s64 = DType::new(Kind::S, 64);
        let s = src("right_shift", &[s64, s64], s64);
        assert!(s.contains("((ulong)x1) >= 64"), "{}", s);
    }

    #[test]
    fn signed_right_shift_is_arithmetic_and_fills_with_the_sign() {
        let s = src("right_shift", &[S8, S8], S8);
        assert!(s.contains("x0 < 0 ? (char)-1 : (char)0"), "{}", s);
        assert!(s.contains("(char)(~((uint)(~x0) >> ((uchar)x1)))"), "{}", s);
        // Unsigned: a plain logical shift.
        let s = src("right_shift", &[U8, U8], U8);
        assert!(s.contains("(uchar)((uint)x0 >> ((uchar)x1))"), "{}", s);
    }

    // ── Pow, round ──

    #[test]
    fn integer_pow_is_repeated_squaring_with_a_negative_exponent_rule() {
        let s = src("pow", &[S32, S32], S32);
        assert!(s.contains("static int pow_s32(int base, int e)"), "{}", s);
        assert!(s.contains("while (n != 0) { if (n & 1) r *= x; x *= x; n >>= 1; }"));
        assert!(s.contains("if (e < 0) return base == 1"), "{}", s);
        assert!(s.contains("out[i] = pow_s32(x0, x1);"));
        // Unsigned has no negative exponent.
        assert!(!src("pow", &[U8, U8], U8).contains("e < 0"));
    }

    #[test]
    fn float_pow_calls_pow_not_powr() {
        let s = src("pow", &[F32, F32], F32);
        assert!(s.contains("pow(x0, x1)"), "{}", s);
        assert!(!s.contains("powr"));
    }

    #[test]
    fn round_is_half_away_from_zero() {
        // OpenCL `round` rounds halfway cases away from zero.
        assert!(src("round", &[F32], F32).contains("round(x0)"));
        assert!(src("round", &[S32], S32).contains("out[i] = x0;"));
    }

    // ── Casts ──

    #[test]
    fn as_type_float_to_int_goes_through_the_saturating_helper() {
        let s = src("as_type", &[F32], U8);
        assert!(s.contains("static uchar f2i_float_u8(float x)"), "{}", s);
        assert!(s.contains("if (x != x) return (uchar)0;"), "{}", s);
        assert!(s.contains("if (isinf(x)) return x > 0 ? (uchar)255 : (uchar)0;"));
        assert!(s.contains("out[i] = f2i_float_u8(x0);"));
    }

    #[test]
    fn as_type_int_to_narrower_int_is_a_c_cast() {
        let s = src("as_type", &[S32], U8);
        assert!(s.contains("out[i] = (uchar)(x0);"), "{}", s);
    }

    #[test]
    fn as_type_float_to_bf16_rounds_on_store_and_to_f16_uses_half() {
        let s = src("as_type", &[F32], BF16);
        assert!(
            s.contains("out[i] = f32_to_bf16((float)((float)(x0)));"),
            "{}",
            s
        );
        let s = src("as_type", &[F32], F16);
        assert!(s.starts_with("#pragma OPENCL EXTENSION cl_khr_fp16 : enable"));
        assert!(s.contains("__global half* out"), "{}", s);
        assert!(s.contains("out[i] = (half)((float)(x0));"), "{}", s);
        assert!(!s.contains("f32_to_f16"), "{}", s);
    }

    // ── Special functions ──

    #[test]
    fn sigmoid_keeps_exp_argument_non_positive() {
        let s = src("sigmoid", &[F32], F32);
        assert!(
            s.contains("x0 >= 0.0f ? 1.0f / (1.0f + exp(-x0)) : exp(x0) / (1.0f + exp(x0))"),
            "{}",
            s
        );
    }

    #[test]
    fn erf_inv_handles_the_tails_and_out_of_range() {
        let s = src("erf_inv", &[F32], F32);
        assert!(s.contains("static float erf_inv_float(float y)"), "{}", s);
        assert!(s.contains("if (!(ay <= 1.0f)) return as_float(0x7fc00000);"));
        assert!(s.contains("if (ay == 1.0f) return y > 0 ? as_float(0x7f800000)"));
        assert!(s.contains("copysign(x, y)"));
        let s = src("erf_inv", &[F64], F64);
        assert!(s.contains("erf_inv_double") && s.contains("k < 3"), "{}", s);
    }

    #[test]
    fn math_ops_on_integer_operands_convert_to_the_float_output() {
        let s = src("exp", &[S32], F32);
        assert!(s.contains("float x0 = ((float)(in0[i]));"), "{}", s);
        assert!(s.contains("out[i] = exp(x0);"));
    }

    #[test]
    fn count_leading_zeros_and_population_count_cover_every_width() {
        for dtype in supported() {
            if dtype.kind == Kind::F || dtype.kind == Kind::Bf {
                continue;
            }
            let info = type_info(dtype).unwrap();
            let s = src("count_leading_zeros", &[dtype], dtype);
            assert!(
                s.contains(&format!("({})clz(({})x0)", info.math, info.unsigned)),
                "{} {}",
                dtype,
                s
            );
            let s = src("population_count", &[dtype], dtype);
            assert!(s.contains("popcount(("), "{}", s);
        }
    }

    // ── Types ──

    #[test]
    fn bf16_add_computes_in_float_and_stores_with_rounding() {
        let s = src("add", &[BF16, BF16], BF16);
        assert!(s.contains("static float bf16_to_f32(ushort x)"), "{}", s);
        assert!(s.contains("float x0 = bf16_to_f32(in0[i]);"), "{}", s);
        assert!(
            s.contains("out[i] = f32_to_bf16((float)((x0 + x1)));"),
            "{}",
            s
        );
        assert!(s.contains("__global const ushort* in0"));
    }

    #[test]
    fn f64_needs_the_extension_and_returns_a_tagged_error_without_it() {
        let i = instr("add", vec![buf(F64), buf(F64)], F64, vec![]);
        let none = Caps {
            fp64: false,
            fp16: true,
        };
        let err = generate(&i, &none).unwrap_err();
        assert!(err.starts_with("missing extension cl_khr_fp64"), "{}", err);
        let s = generate(&i, &Caps::all()).unwrap().source;
        assert!(s.starts_with("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"));
        // A kernel with no f64 never asks for it.
        assert!(generate(&instr("add", vec![buf(F32), buf(F32)], F32, vec![]), &none).is_ok());
    }

    #[test]
    fn caps_read_the_extension_string() {
        assert!(Caps::from_extensions("cl_khr_byte_addressable_store cl_khr_fp64").fp64);
        assert!(!Caps::from_extensions("cl_khr_fp16").fp64);
        assert!(Caps::from_extensions("cl_khr_fp16").fp16);
        assert!(!Caps::from_extensions("cl_khr_fp64").fp16);
    }

    #[test]
    fn f16_needs_the_extension_and_returns_a_tagged_error_without_it() {
        let none = Caps {
            fp64: true,
            fp16: false,
        };
        let i = instr("add", vec![buf(F16), buf(F16)], F16, vec![]);
        let err = generate(&i, &none).unwrap_err();
        assert!(err.starts_with("missing extension cl_khr_fp16"), "{}", err);
        // Reading or writing f16 alone is enough to need it.
        let i = instr("as_type", vec![buf(F32)], F16, vec![]);
        assert!(generate(&i, &none)
            .unwrap_err()
            .starts_with("missing extension cl_khr_fp16"));
        let i = instr("as_type", vec![buf(F16)], F32, vec![]);
        assert!(generate(&i, &none)
            .unwrap_err()
            .starts_with("missing extension cl_khr_fp16"));
        let s = generate(
            &instr("add", vec![buf(F16), buf(F16)], F16, vec![]),
            &Caps::all(),
        )
        .unwrap()
        .source;
        assert!(s.starts_with("#pragma OPENCL EXTENSION cl_khr_fp16 : enable"));
        assert!(s.contains("float x0 = (float)(in0[i]);"), "{}", s);
        assert!(s.contains("out[i] = (half)((x0 + x1));"), "{}", s);
    }

    // ── Operands, entry, cache key ──

    #[test]
    fn a_scalar_constant_is_a_scalar_argument() {
        let i = instr(
            "add",
            vec![buf(F32), constant(F32, ConstVal::Float(2.0))],
            F32,
            vec![],
        );
        let spec = generate(&i, &Caps::all()).unwrap();
        assert!(spec.source.contains("const float in1"), "{}", spec.source);
        assert!(spec.source.contains("float x1 = in1;"));
        assert_eq!(
            spec.args,
            vec![
                KernelArg::Operand(0),
                KernelArg::Scalar(2.0f32.to_ne_bytes().to_vec()),
                KernelArg::Output
            ]
        );
    }

    #[test]
    fn a_constant_of_another_type_converts_to_the_compute_type() {
        let i = instr(
            "add",
            vec![buf(U8), constant(S32, ConstVal::Int(300))],
            U8,
            vec![],
        );
        let spec = generate(&i, &Caps::all()).unwrap();
        assert!(
            spec.source.contains("uchar x1 = ((uchar)(in1));"),
            "{}",
            spec.source
        );
    }

    #[test]
    fn the_source_does_not_depend_on_the_constant_value_or_the_shape() {
        let make = |v: f64, n: usize| {
            let mut i = instr(
                "multiply",
                vec![buf(F32), constant(F32, ConstVal::Float(v))],
                F32,
                vec![],
            );
            i.out_shape = vec![n];
            i.operands[0].shape = vec![n];
            generate(&i, &Caps::all()).unwrap()
        };
        let (a, b) = (make(1.0, 4), make(7.5, 1000));
        assert_eq!(a.source, b.source);
        assert_eq!(a.entry, b.entry);
        assert_eq!(b.global, vec![1000]);
    }

    #[test]
    fn entry_names_carry_the_op_and_types() {
        let spec = generate(
            &instr("add", vec![buf(S32), buf(S32)], S32, vec![]),
            &Caps::all(),
        )
        .unwrap();
        assert_eq!(spec.entry, "ew_add_s32_s32_s32");
        assert!(spec.source.contains("__kernel void ew_add_s32_s32_s32("));
    }

    #[test]
    fn operands_of_different_shapes_need_broadcast_and_are_rejected() {
        let mut i = instr("add", vec![buf(F32), buf(F32)], F32, vec![]);
        i.operands[1].shape = vec![];
        let err = generate(&i, &Caps::all()).unwrap_err();
        assert!(err.contains("broadcast"), "{}", err);
    }

    // ── Coverage of every op family ──

    #[test]
    fn every_unary_op_generates_for_every_dtype_it_accepts() {
        let float_only: Vec<&str> = UNARY_FLOAT_FNS
            .iter()
            .copied()
            .chain(["sigmoid", "erf_inv"])
            .collect();
        for dtype in supported() {
            let info = type_info(dtype).unwrap();
            for op in &float_only {
                let r = generate(&instr(op, vec![buf(dtype)], dtype, vec![]), &Caps::all());
                assert_eq!(r.is_ok(), info.is_float(), "{} {}", op, dtype);
            }
            for op in ["abs", "negate", "sign", "ceil", "floor", "round"] {
                src(op, &[dtype], dtype);
            }
            for op in ["bitwise_not", "count_leading_zeros", "population_count"] {
                let r = generate(&instr(op, vec![buf(dtype)], dtype, vec![]), &Caps::all());
                assert_eq!(r.is_ok(), !info.is_float(), "{} {}", op, dtype);
            }
            src("is_nan", &[dtype], U8);
            src("is_infinity", &[dtype], U8);
            src("logical_not", &[dtype], U8);
        }
    }

    #[test]
    fn every_binary_op_generates_for_every_dtype_it_accepts() {
        for dtype in supported() {
            let info = type_info(dtype).unwrap();
            let both = |op: &str| {
                generate(
                    &instr(op, vec![buf(dtype), buf(dtype)], dtype, vec![]),
                    &Caps::all(),
                )
            };
            for op in [
                "add",
                "subtract",
                "multiply",
                "remainder",
                "pow",
                "max",
                "min",
            ] {
                assert!(both(op).is_ok(), "{} {}", op, dtype);
            }
            for op in ["divide", "atan2"] {
                assert_eq!(both(op).is_ok(), info.is_float(), "{} {}", op, dtype);
            }
            for op in [
                "quotient",
                "left_shift",
                "right_shift",
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
            ] {
                assert_eq!(both(op).is_ok(), !info.is_float(), "{} {}", op, dtype);
            }
            for op in COMPARE {
                compare_src(op, dtype, dtype);
            }
            for op in LOGICAL {
                src(op, &[dtype, dtype], U8);
            }
            src("select", &[U8, dtype, dtype], dtype);
            src("clip", &[dtype, dtype, dtype], dtype);
        }
    }

    #[test]
    fn as_type_generates_for_every_pair_of_dtypes() {
        for from in supported() {
            for to in supported() {
                src("as_type", &[from], to);
            }
        }
    }

    #[test]
    fn every_listed_op_is_handled() {
        for op in UNARY_FLOAT_FNS
            .iter()
            .chain(&UNARY_OTHER)
            .chain(&BINARY_ARITH)
            .chain(&BINARY_BITWISE)
            .chain(&COMPARE)
            .chain(&LOGICAL)
        {
            assert!(handles(op), "{}", op);
        }
        assert!(!handles("dot"));
        assert!(!handles("bitcast"));
    }

    // ── Complex ──

    const C64: DType = DType::new(Kind::C, 64);
    const C128: DType = DType::new(Kind::C, 128);

    #[test]
    fn complex_add_carries_the_preamble_and_calls_the_helper() {
        let s = src("add", &[C64, C64], C64);
        assert!(s.contains("#pragma OPENCL FP_CONTRACT OFF"), "{}", s);
        assert!(
            s.contains("static float2 cf_add(float2 a, float2 b)"),
            "{}",
            s
        );
        assert!(s.contains("out[i] = cf_add(x0, x1);"), "{}", s);
        assert!(s.contains("__global const float2* in0"), "{}", s);
        // Only the helpers the kernel calls.
        assert!(!s.contains("cf_sin") && !s.contains("double"), "{}", s);
    }

    #[test]
    fn complex_division_is_smiths_method() {
        let s = src("divide", &[C64, C64], C64);
        assert!(s.contains("fabs(b.x) >= fabs(b.y)"), "{}", s);
        assert!(!s.contains('$'), "{}", s);
    }

    #[test]
    fn c128_needs_fp64_and_gets_its_own_helpers() {
        let s = src("multiply", &[C128, C128], C128);
        assert!(
            s.starts_with("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),
            "{}",
            s
        );
        assert!(
            s.contains("static double2 cd_mul(double2 a, double2 b)"),
            "{}",
            s
        );
        let no64 = Caps {
            fp64: false,
            fp16: true,
        };
        let operands = vec![buf(C128), buf(C128)];
        let err = generate(&instr("add", operands, C128, vec![]), &no64).unwrap_err();
        assert!(err.contains("missing extension cl_khr_fp64"), "{}", err);
    }

    #[test]
    fn kernels_without_complex_types_have_no_preamble() {
        for (op, ins, out) in [
            ("add", vec![F32, F32], F32),
            ("sqrt", vec![F64], F64),
            ("equal", vec![S32, S32], U8),
            ("as_type", vec![F32], S32),
            ("real", vec![F32], F32),
        ] {
            let s = src(op, &ins, out);
            assert!(
                !s.contains("FP_CONTRACT") && !s.contains("cf_"),
                "{} {}",
                op,
                s
            );
        }
    }

    #[test]
    fn complex_abs_returns_the_real_type_and_tests_return_u8() {
        assert!(src("abs", &[C64], F32).contains("out[i] = ((float)(cf_abs(x0)));"));
        let s = src("abs", &[C128], F64);
        assert!(
            s.contains("__global double* out") && s.contains("cd_abs(x0)"),
            "{}",
            s
        );
        assert!(src("is_nan", &[C64], U8).contains("(uchar)(cf_isnan(x0))"));
        assert!(src("is_infinity", &[C64], U8).contains("(uchar)(cf_isinf(x0))"));
        assert!(src("equal", &[C64, C64], U8).contains("(uchar)(cf_eq(x0, x1))"));
        assert!(src("not_equal", &[C64, C64], U8).contains("(uchar)(cf_ne(x0, x1))"));
        assert!(src("logical_not", &[C64], U8).contains("(uchar)(!cf_nonzero(x0))"));
    }

    #[test]
    fn real_imag_and_conjugate_read_and_flip_the_parts() {
        assert!(src("real", &[C64], F32).contains("x0.x"));
        assert!(src("imag", &[C64], F32).contains("x0.y"));
        assert!(src("conjugate", &[C64], C64).contains("cf_conj(x0)"));
        // A real operand gets a zero imaginary part first; conjugate flips it to -0.
        let s = src("conjugate", &[F32], C64);
        assert!(s.contains("(float2)((float)(in0[i]), (float)0)"), "{}", s);
        assert!(src("imag", &[F32], F32).contains("x0.y"));
    }

    #[test]
    fn phase_is_atan2_of_the_parts_for_complex_and_real_operands() {
        assert!(src("phase", &[C64], F32).contains("atan2(x0.y, x0.x)"));
        let s = src("phase", &[F32], F32);
        assert!(
            s.contains("atan2(x0.y, x0.x)") && s.contains("(float)0"),
            "{}",
            s
        );
    }

    #[test]
    fn casts_to_and_from_complex() {
        // Real to complex: zero imaginary part.
        let s = src("as_type", &[F32], C64);
        assert!(s.contains("(float2)((float)(in0[i]), (float)0)"), "{}", s);
        let s = src("as_type", &[S32], C64);
        assert!(s.contains("(float2)((float)("), "{}", s);
        // Complex to real: the imaginary part is dropped.
        assert!(src("as_type", &[C64], F32).contains("(float)(x0.x)"));
        assert!(src("as_type", &[C64], S32).contains("f2i_float_s32(x0.x)"));
        // c64 <-> c128 convert part by part.
        assert!(src("as_type", &[C64], C128).contains("convert_double2(in0[i])"));
        assert!(src("as_type", &[C128], C64).contains("convert_float2(in0[i])"));
    }

    #[test]
    fn a_complex_constant_is_one_vector_argument() {
        let operands = vec![buf(C64), constant(C64, ConstVal::Complex(1.5, -2.0))];
        let spec = generate(&instr("multiply", operands, C64, vec![]), &Caps::all()).unwrap();
        assert!(spec.source.contains("const float2 in1"), "{}", spec.source);
        let mut want = 1.5f32.to_ne_bytes().to_vec();
        want.extend_from_slice(&(-2.0f32).to_ne_bytes());
        assert_eq!(spec.args[1], KernelArg::Scalar(want));
        let operands = vec![buf(C128), constant(C128, ConstVal::Complex(1.5, -2.0))];
        let spec = generate(&instr("add", operands, C128, vec![]), &Caps::all()).unwrap();
        assert!(spec.source.contains("const double2 in1"));
        assert_eq!(
            spec.args[1],
            KernelArg::Scalar([1.5f64.to_ne_bytes(), (-2.0f64).to_ne_bytes()].concat())
        );
    }

    #[test]
    fn mixed_real_and_complex_operands_convert_to_complex() {
        let s = src("multiply", &[C64, F32], C64);
        assert!(
            s.contains("float2 x1 = (float2)((float)(in1[i]), (float)0);"),
            "{}",
            s
        );
        assert!(s.contains("cf_mul(x0, x1)"), "{}", s);
    }

    #[test]
    fn no_complex_op_template_uses_the_vector_operators_for_products() {
        // `float2 * float2` is a component product, not a complex product.
        for op in ["multiply", "divide", "pow"] {
            let s = src(op, &[C64, C64], C64);
            assert!(
                !s.contains("x0 * x1") && !s.contains("x0 / x1"),
                "{} {}",
                op,
                s
            );
        }
    }

    #[test]
    fn complex_ops_nx_rejects_have_no_template() {
        for op in [
            "max",
            "min",
            "greater",
            "less",
            "atan2",
            "remainder",
            "ceil",
            "erf",
            "bitwise_and",
        ] {
            let ins = if matches!(op, "ceil" | "erf") {
                vec![C64]
            } else {
                vec![C64, C64]
            };
            let operands = ins.iter().map(|&d| buf(d)).collect();
            let out = if matches!(
                op,
                "max" | "min" | "atan2" | "remainder" | "ceil" | "erf" | "bitwise_and"
            ) {
                C64
            } else {
                U8
            };
            let err = generate(&instr(op, operands, out, vec![]), &Caps::all()).unwrap_err();
            assert!(
                err.contains("does not accept complex operands"),
                "{} {}",
                op,
                err
            );
        }
    }

    #[test]
    fn every_complex_unary_op_generates_for_both_widths() {
        for (op, helper) in COMPLEX_UNARY {
            for t in [C64, C128] {
                let s = src(op, &[t], t);
                let p = if t == C64 { "cf" } else { "cd" };
                assert!(s.contains(&format!("{}_{}(", p, helper)), "{} {}", op, s);
            }
        }
        for op in ["add", "subtract", "multiply", "divide", "pow"] {
            for t in [C64, C128] {
                src(op, &[t, t], t);
            }
        }
    }

    #[test]
    fn the_kernel_build_enables_no_fast_math() {
        let operands = vec![buf(C64), buf(C64)];
        let spec = generate(&instr("divide", operands, C64, vec![]), &Caps::all()).unwrap();
        assert_eq!(spec.build_options, "");
        assert!(!spec.source.contains("fast"));
    }

    #[test]
    fn classes_report_signedness() {
        assert_eq!(type_info(S8).unwrap().class, Class::Signed);
        assert_eq!(type_info(U8).unwrap().class, Class::Unsigned);
    }
}
