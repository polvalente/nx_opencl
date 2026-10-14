//! Complex math for kernels. OpenCL C has no complex type, so `c64` is a
//! `float2` and `c128` a `double2` (real part in `.x`, imaginary part in `.y`),
//! and every operation is a plain function written out here. The vector
//! operators of `float2` work component by component, not as complex
//! arithmetic, so the helpers never rely on them for products or quotients.
//!
//! A kernel gets the helpers it calls (and what those call), once per width it
//! uses. Kernels without a complex type get nothing. The functions follow
//! C99 Annex G where it says something (the sign of zero on a branch cut,
//! infinity and NaN), and Nx's own rules elsewhere. `sqrt`, `log`, `asin`,
//! `acos`, `atan`, `asinh`, `acosh` and `atanh` keep the sign of a zero
//! imaginary part, which picks the side of the cut. `FP_CONTRACT` is off, so a
//! compiler never fuses a product and a sum into one rounding. The kernels are
//! built without any fast-math option.

use std::collections::BTreeSet;

use super::types::{Preamble, TypeInfo};

/// One helper: its name (without the width prefix), the helpers it calls, and
/// its source. `$V` is the vector type, `$T` its part type, `$P` the name
/// prefix, `$s` the float literal suffix (`$i` after an integer, to make it a
/// float literal), `$thr` the `tanh` cut-over and
/// `$big` the size above which a square would overflow.
struct Func {
    name: &'static str,
    deps: &'static [&'static str],
    src: &'static str,
}

/// In dependency order: a helper comes after everything it calls.
const FUNCS: &[Func] = &[
    Func {
        name: "add",
        deps: &[],
        src: "static $V $P_add($V a, $V b) { return ($V)(a.x + b.x, a.y + b.y); }\n",
    },
    Func {
        name: "sub",
        deps: &[],
        src: "static $V $P_sub($V a, $V b) { return ($V)(a.x - b.x, a.y - b.y); }\n",
    },
    Func {
        name: "mul",
        deps: &[],
        src: "static $V $P_mul($V a, $V b) { return ($V)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }\n",
    },
    // Smith's method: divide by the larger part of the divisor, so nothing
    // overflows when the parts are large. Division by zero gives NaN parts.
    Func {
        name: "div",
        deps: &[],
        src: "static $V $P_div($V a, $V b) {
  if (fabs(b.x) >= fabs(b.y)) {
    $T r = b.y / b.x;
    $T den = b.x + b.y * r;
    return ($V)((a.x + a.y * r) / den, (a.y - a.x * r) / den);
  }
  $T r = b.x / b.y;
  $T den = b.x * r + b.y;
  return ($V)((a.x * r + a.y) / den, (a.y * r - a.x) / den);
}
",
    },
    Func {
        name: "neg",
        deps: &[],
        src: "static $V $P_neg($V a) { return ($V)(-a.x, -a.y); }\n",
    },
    Func {
        name: "conj",
        deps: &[],
        src: "static $V $P_conj($V a) { return ($V)(a.x, -a.y); }\n",
    },
    // Scaled by the larger part, so it neither overflows nor underflows
    // where `hypot` is not exact. An infinite part wins over NaN.
    Func {
        name: "abs",
        deps: &[],
        src: "static $T $P_abs($V a) {
  $T x = fabs(a.x), y = fabs(a.y);
  if (isinf(x) || isinf(y)) return ($T)INFINITY;
  if (isnan(x) || isnan(y)) return ($T)NAN;
  $T hi = fmax(x, y), lo = fmin(x, y);
  if (hi == 0$i) return 0$i;
  $T r = lo / hi;
  return hi * sqrt(1$i + r * r);
}
",
    },
    Func {
        name: "isnan",
        deps: &[],
        src: "static int $P_isnan($V a) { return isnan(a.x) || isnan(a.y); }\n",
    },
    Func {
        name: "isinf",
        deps: &[],
        src: "static int $P_isinf($V a) { return isinf(a.x) || isinf(a.y); }\n",
    },
    // A complex value is true when either part is not zero (NaN counts).
    Func {
        name: "nonzero",
        deps: &[],
        src: "static int $P_nonzero($V a) { return a.x != 0$i || a.y != 0$i; }\n",
    },
    Func {
        name: "eq",
        deps: &[],
        src: "static int $P_eq($V a, $V b) { return a.x == b.x && a.y == b.y; }\n",
    },
    Func {
        name: "ne",
        deps: &[],
        src: "static int $P_ne($V a, $V b) { return a.x != b.x || a.y != b.y; }\n",
    },
    // z / |z|; zero stays zero.
    Func {
        name: "sign",
        deps: &["abs"],
        src: "static $V $P_sign($V a) {
  $T m = $P_abs(a);
  if (m == 0$i) return ($V)(0$i, 0$i);
  return ($V)(a.x / m, a.y / m);
}
",
    },
    // log |z|, accurate near 1 (through log1p) and free of overflow.
    Func {
        name: "logabs",
        deps: &[],
        src: "static $T $P_logabs($V a) {
  $T x = fabs(a.x), y = fabs(a.y);
  if (isinf(x) || isinf(y)) return ($T)INFINITY;
  if (isnan(x) || isnan(y)) return ($T)NAN;
  $T hi = fmax(x, y), lo = fmin(x, y);
  if (hi == 0$i) return -($T)INFINITY;
  if (hi >= 0.5$s && hi <= 2$i) return 0.5$s * log1p((hi - 1$i) * (hi + 1$i) + lo * lo);
  $T r = lo / hi;
  return log(hi) + 0.5$s * log1p(r * r);
}
",
    },
    // The imaginary part is `atan2`, so -0 on the cut gives -pi.
    Func {
        name: "log",
        deps: &["logabs"],
        src: "static $V $P_log($V a) { return ($V)($P_logabs(a), atan2(a.y, a.x)); }\n",
    },
    Func {
        name: "exp",
        deps: &[],
        src: "static $V $P_exp($V a) {
  $T e = exp(a.x);
  if (a.y == 0$i) return ($V)(e, a.y);
  return ($V)(e * cos(a.y), e * sin(a.y));
}
",
    },
    // exp(z) - 1 without the cancellation near 0.
    Func {
        name: "expm1",
        deps: &[],
        src: "static $V $P_expm1($V a) {
  $T s = sin(0.5$s * a.y);
  $T re = expm1(a.x) * cos(a.y) - 2$i * s * s;
  if (a.y == 0$i) return ($V)(re, a.y);
  return ($V)(re, exp(a.x) * sin(a.y));
}
",
    },
    // log(1 + z), through log1p of |1 + z|^2 - 1 while the square is safe.
    Func {
        name: "log1p",
        deps: &["log"],
        src: "static $V $P_log1p($V a) {
  if (fabs(a.x) > $big || fabs(a.y) > $big) return $P_log(($V)(1$i + a.x, a.y));
  $T re = 0.5$s * log1p(a.x * (2$i + a.x) + a.y * a.y);
  return ($V)(re, atan2(a.y, 1$i + a.x));
}
",
    },
    // C99 csqrt: infinities and NaN first, then the stable formula that
    // never subtracts nearly equal numbers. The sign of a zero imaginary part
    // goes to the result.
    Func {
        name: "sqrt",
        deps: &["abs"],
        src: "static $V $P_sqrt($V a) {
  $T x = a.x, y = a.y;
  if (isinf(y)) return ($V)(($T)INFINITY, y);
  if (isnan(x)) return ($V)(($T)NAN, ($T)NAN);
  if (isinf(x)) {
    if (x > 0$i) return ($V)(x, isnan(y) ? y : copysign(0$i, y));
    return ($V)(isnan(y) ? y : 0$i, copysign(($T)INFINITY, y));
  }
  if (isnan(y)) return ($V)(($T)NAN, ($T)NAN);
  if (x == 0$i && y == 0$i) return ($V)(0$i, y);
  $T t = sqrt(0.5$s * fabs(x) + 0.5$s * $P_abs(a));
  if (x >= 0$i) return ($V)(t, y / (2$i * t));
  return ($V)(fabs(y) / (2$i * t), copysign(t, y));
}
",
    },
    Func {
        name: "rsqrt",
        deps: &["div", "sqrt"],
        src: "static $V $P_rsqrt($V a) { return $P_div(($V)(1$i, 0$i), $P_sqrt(a)); }\n",
    },
    // Principal cube root: cbrt|z| at a third of the angle.
    Func {
        name: "cbrt",
        deps: &["abs"],
        src: "static $V $P_cbrt($V a) {
  $T m = $P_abs(a);
  if (m == 0$i) return ($V)(0$i, 0$i);
  $T r = cbrt(m);
  $T t = atan2(a.y, a.x) / 3$i;
  return ($V)(r * cos(t), r * sin(t));
}
",
    },
    // Polar form, as Nx does it: z^0 = 1, 0^w = 0, z^1 = z, z^-1 = 1/z.
    Func {
        name: "pow",
        deps: &["abs", "div"],
        src: "static $V $P_pow($V a, $V b) {
  if (a.x == 0$i && a.y == 0$i) return (b.x == 0$i && b.y == 0$i) ? ($V)(1$i, 0$i) : ($V)(0$i, 0$i);
  if (b.x == 1$i && b.y == 0$i) return a;
  if (b.x == -1$i && b.y == 0$i) return $P_div(($V)(1$i, 0$i), a);
  $T rho = $P_abs(a);
  $T theta = atan2(a.y, a.x);
  $T s = pow(rho, b.x) * exp(-b.y * theta);
  $T r = b.x * theta + b.y * log(rho);
  return ($V)(s * cos(r), s * sin(r));
}
",
    },
    Func {
        name: "sin",
        deps: &[],
        src: "static $V $P_sin($V a) { return ($V)(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)); }\n",
    },
    Func {
        name: "cos",
        deps: &[],
        src: "static $V $P_cos($V a) { return ($V)(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)); }\n",
    },
    Func {
        name: "sinh",
        deps: &[],
        src: "static $V $P_sinh($V a) { return ($V)(sinh(a.x) * cos(a.y), cosh(a.x) * sin(a.y)); }\n",
    },
    Func {
        name: "cosh",
        deps: &[],
        src: "static $V $P_cosh($V a) { return ($V)(cosh(a.x) * cos(a.y), sinh(a.x) * sin(a.y)); }\n",
    },
    // FreeBSD ctanh: a form that stays finite where sinh and cosh overflow.
    Func {
        name: "tanh",
        deps: &[],
        src: "static $V $P_tanh($V a) {
  $T x = a.x, y = a.y;
  if (isinf(x)) return ($V)(copysign(1$i, x), (isinf(y) || isnan(y)) ? copysign(0$i, y) : copysign(0$i, sin(2$i * y)));
  if (isnan(x)) return y == 0$i ? a : ($V)(($T)NAN, ($T)NAN);
  if (isinf(y) || isnan(y)) return ($V)(($T)NAN, ($T)NAN);
  if (fabs(x) >= $thr) {
    $T e = exp(-fabs(x));
    return ($V)(copysign(1$i, x), 4$i * sin(y) * cos(y) * e * e);
  }
  $T t = tan(y);
  $T beta = 1$i + t * t;
  $T s = sinh(x);
  $T rho = sqrt(1$i + s * s);
  $T den = 1$i + beta * s * s;
  return ($V)(beta * rho * s / den, t / den);
}
",
    },
    // tan z = -i tanh(i z)
    Func {
        name: "tan",
        deps: &["tanh"],
        src: "static $V $P_tan($V a) {
  $V w = $P_tanh(($V)(-a.y, a.x));
  return ($V)(w.y, -w.x);
}
",
    },
    // Kahan's arcsine: the square roots of 1 - z and 1 + z keep the sign of
    // a zero imaginary part, which puts z on the right side of the cuts.
    Func {
        name: "asin",
        deps: &["sqrt"],
        src: "static $V $P_asin($V a) {
  $V s1 = $P_sqrt(($V)(1$i - a.x, -a.y));
  $V s2 = $P_sqrt(($V)(1$i + a.x, a.y));
  return ($V)(atan2(a.x, s1.x * s2.x - s1.y * s2.y), asinh(s1.x * s2.y - s1.y * s2.x));
}
",
    },
    Func {
        name: "acos",
        deps: &["sqrt"],
        src: "static $V $P_acos($V a) {
  $V s1 = $P_sqrt(($V)(1$i - a.x, -a.y));
  $V s2 = $P_sqrt(($V)(1$i + a.x, a.y));
  return ($V)(2$i * atan2(s1.x, s2.x), asinh(s2.x * s1.y - s2.y * s1.x));
}
",
    },
    // asinh z = -i asin(i z)
    Func {
        name: "asinh",
        deps: &["asin"],
        src: "static $V $P_asinh($V a) {
  $V w = $P_asin(($V)(-a.y, a.x));
  return ($V)(w.y, -w.x);
}
",
    },
    // 2 log(sqrt((z + 1) / 2) + sqrt((z - 1) / 2)): right on the cut and
    // free of overflow for large z. The real part is never below zero;
    // rounding in the log could leave it a hair under.
    Func {
        name: "acosh",
        deps: &["sqrt", "log"],
        src: "static $V $P_acosh($V a) {
  $V p = $P_sqrt(($V)(0.5$s * (a.x + 1$i), 0.5$s * a.y));
  $V m = $P_sqrt(($V)(0.5$s * (a.x - 1$i), 0.5$s * a.y));
  $V l = $P_log(($V)(p.x + m.x, p.y + m.y));
  return ($V)(l.x <= 0$i ? 0$i : 2$i * l.x, 2$i * l.y);
}
",
    },
    Func {
        name: "atanh",
        deps: &[],
        src: "static $V $P_atanh($V a) {
  $T x = a.x, y = a.y;
  $T om = 1$i - x;
  return ($V)(0.25$s * log1p(4$i * x / (om * om + y * y)), 0.5$s * atan2(2$i * y, om * (1$i + x) - y * y));
}
",
    },
    // atan z = -i atanh(i z)
    Func {
        name: "atan",
        deps: &["atanh"],
        src: "static $V $P_atan($V a) {
  $V w = $P_atanh(($V)(-a.y, a.x));
  return ($V)(w.y, -w.x);
}
",
    },
    Func {
        name: "sigmoid",
        deps: &["add", "div", "exp", "neg"],
        src: "static $V $P_sigmoid($V a) {
  return $P_div(($V)(1$i, 0$i), $P_add(($V)(1$i, 0$i), $P_exp($P_neg(a))));
}
",
    },
];

/// The prefix of the helper names for the type's width: `cf_` for `float2`,
/// `cd_` for `double2`.
pub fn prefix(info: &TypeInfo) -> &'static str {
    if info.component == "double" {
        "cd"
    } else {
        "cf"
    }
}

/// The C name of helper `name` for the type.
pub fn fn_name(info: &TypeInfo, name: &str) -> String {
    format!("{}_{}", prefix(info), name)
}

fn fill(src: &str, info: &TypeInfo) -> String {
    let double = info.component == "double";
    src.replace("$V", info.math)
        .replace("$T", info.component)
        .replace("$P", prefix(info))
        .replace("$i", if double { ".0" } else { ".0f" })
        .replace("$s", if double { "" } else { "f" })
        .replace("$thr", if double { "22.0" } else { "11.0f" })
        .replace("$big", if double { "1.0e150" } else { "1.0e15f" })
}

/// Names of every helper, for tests.
#[cfg(test)]
pub fn all_names() -> Vec<&'static str> {
    FUNCS.iter().map(|f| f.name).collect()
}

/// Add `name` and everything it calls to `set`.
fn close(name: &str, set: &mut BTreeSet<&'static str>) {
    if let Some(f) = FUNCS.iter().find(|f| f.name == name) {
        if set.insert(f.name) {
            for dep in f.deps {
                close(dep, set);
            }
        }
    }
}

/// The source for the helpers `names` (and what they call) over the width of
/// `info`. Empty for a real type, so a kernel without complex values keeps
/// its source, hash and build as before.
pub fn preamble(info: &TypeInfo, names: &[&str]) -> String {
    if info.preamble != Preamble::Complex {
        return String::new();
    }
    let mut needed = BTreeSet::new();
    for name in names {
        close(name, &mut needed);
    }
    let mut out = String::from("#pragma OPENCL FP_CONTRACT OFF\n");
    for f in FUNCS.iter().filter(|f| needed.contains(f.name)) {
        out.push_str(&fill(f.src, info));
    }
    out
}

// ── Fragments for later stages ──────────────────────────

/// A complex literal of the type's width: `(float2)(re, im)`.
#[allow(dead_code)]
pub fn literal(info: &TypeInfo, re: f64, im: f64) -> String {
    let s = if info.component == "double" { "" } else { "f" };
    format!("({})({:?}{s}, {:?}{s})", info.math, re, im, s = s)
}

#[allow(dead_code)]
pub fn zero(info: &TypeInfo) -> String {
    literal(info, 0.0, 0.0)
}

#[allow(dead_code)]
pub fn one(info: &TypeInfo) -> String {
    literal(info, 1.0, 0.0)
}

/// The real and imaginary part of element `idx` of buffer `buf`.
#[allow(dead_code)]
pub fn load_parts(buf: &str, idx: &str) -> (String, String) {
    (format!("{buf}[{idx}].x"), format!("{buf}[{idx}].y"))
}

/// A statement that stores the complex value `value` at element `idx`.
#[allow(dead_code)]
pub fn store(buf: &str, idx: &str, value: &str) -> String {
    format!("{buf}[{idx}] = {value};")
}

/// The helpers `accumulate` calls for `op` (`"sum"` or `"product"`).
#[allow(dead_code)]
pub fn accumulate_needs(op: &str) -> Result<&'static [&'static str], String> {
    match op {
        "sum" => Ok(&["add"]),
        "product" => Ok(&["mul"]),
        other => Err(format!("no complex accumulate for {}", other)),
    }
}

/// The expression that folds `val` into `acc` for a complex sum or product.
#[allow(dead_code)]
pub fn accumulate(info: &TypeInfo, op: &str, acc: &str, val: &str) -> Result<String, String> {
    let name = match op {
        "sum" => "add",
        "product" => "mul",
        other => return Err(format!("no complex accumulate for {}", other)),
    };
    Ok(format!("{}({}, {})", fn_name(info, name), acc, val))
}

#[cfg(test)]
mod tests {
    use super::super::types::{complex_types, supported, type_info, DType, Kind};
    use super::*;

    #[test]
    fn every_helper_comes_after_what_it_calls() {
        for (i, f) in FUNCS.iter().enumerate() {
            for dep in f.deps {
                let at = FUNCS
                    .iter()
                    .position(|g| g.name == *dep)
                    .expect("known dep");
                assert!(at < i, "{} calls {} which comes later", f.name, dep);
            }
        }
    }

    #[test]
    fn full_preamble_defines_every_function_for_each_width() {
        let want = all_names();
        for dtype in complex_types() {
            let info = type_info(dtype).unwrap();
            let text = preamble(&info, &want);
            for name in &want {
                let def = format!(" {}(", fn_name(&info, name));
                assert!(text.contains(&def), "{} lacks {}", dtype, name);
            }
            assert!(text.starts_with("#pragma OPENCL FP_CONTRACT OFF"));
            assert!(!text.contains('$'), "{} has an unfilled placeholder", dtype);
        }
    }

    #[test]
    fn widths_use_their_own_types_and_prefix() {
        let c64 = type_info(DType::new(Kind::C, 64)).unwrap();
        let c128 = type_info(DType::new(Kind::C, 128)).unwrap();
        let (a, b) = (preamble(&c64, &["div"]), preamble(&c128, &["div"]));
        assert!(a.contains("float2 cf_div(float2 a, float2 b)") && !a.contains("double"));
        assert!(b.contains("double2 cd_div(double2 a, double2 b)") && !b.contains("float"));
    }

    #[test]
    fn a_helper_pulls_in_what_it_calls_and_nothing_more() {
        let c64 = type_info(DType::new(Kind::C, 64)).unwrap();
        let text = preamble(&c64, &["rsqrt"]);
        for name in ["div", "sqrt", "abs", "rsqrt"] {
            assert!(text.contains(&format!(" cf_{}(", name)), "{}", name);
        }
        assert!(!text.contains(" cf_sin("));
    }

    #[test]
    fn real_types_get_no_preamble() {
        for dtype in supported() {
            let info = type_info(dtype).unwrap();
            assert_eq!(preamble(&info, &all_names()), "", "{}", dtype);
        }
    }

    #[test]
    fn literals_and_fragments_follow_the_width() {
        let c64 = type_info(DType::new(Kind::C, 64)).unwrap();
        let c128 = type_info(DType::new(Kind::C, 128)).unwrap();
        assert_eq!(literal(&c64, 1.5, -2.0), "(float2)(1.5f, -2.0f)");
        assert_eq!(zero(&c128), "(double2)(0.0, 0.0)");
        assert_eq!(one(&c128), "(double2)(1.0, 0.0)");
        assert_eq!(
            load_parts("a", "i"),
            ("a[i].x".to_string(), "a[i].y".to_string())
        );
        assert_eq!(store("o", "i", "v"), "o[i] = v;");
        assert_eq!(accumulate(&c64, "sum", "s", "v").unwrap(), "cf_add(s, v)");
        assert_eq!(
            accumulate(&c128, "product", "s", "v").unwrap(),
            "cd_mul(s, v)"
        );
        assert!(accumulate(&c64, "max", "s", "v").is_err());
        assert_eq!(accumulate_needs("sum").unwrap(), &["add"]);
    }
}
