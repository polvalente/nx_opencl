//! The Nx-to-OpenCL type table. One entry per supported dtype: C type of the
//! buffer element, C type of the math value, byte width, class, and the device
//! extension it needs.
//!
//! `bf16` is stored as `ushort` and computed as `float`; the kernel helpers
//! `bf16_to_f32` and `f32_to_bf16` convert (round to nearest even on store). It
//! needs no device extension. `f16` is stored as `half` and computed as `float`;
//! it needs `cl_khr_fp16`, and codegen returns a missing-extension error on a
//! device without it. Bool is `u8`.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    S,
    U,
    F,
    Bf,
    C,
    F8E4M3FN,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DType {
    pub kind: Kind,
    pub bits: u32,
}

impl DType {
    pub const fn new(kind: Kind, bits: u32) -> Self {
        Self { kind, bits }
    }

    /// Parse the `{kind, bits}` pair the wire sends.
    pub fn from_wire(kind: &str, bits: u32) -> Result<Self, String> {
        let kind = match kind {
            "s" => Kind::S,
            "u" => Kind::U,
            "f" => Kind::F,
            "bf" => Kind::Bf,
            "c" => Kind::C,
            "f8_e4m3fn" => Kind::F8E4M3FN,
            other => return Err(format!("unknown dtype kind {}", other)),
        };
        Ok(Self { kind, bits })
    }

    /// Short readable name, such as `s32` or `bf16`. Used in entry names.
    pub fn name(&self) -> String {
        let kind = match self.kind {
            Kind::S => "s",
            Kind::U => "u",
            Kind::F => "f",
            Kind::Bf => "bf",
            Kind::C => "c",
            Kind::F8E4M3FN => "f8e4m3fn",
        };
        format!("{}{}", kind, self.bits)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Signed,
    Unsigned,
    Float,
}

/// Half-width float formats held in a `ushort`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ext {
    Fp64,
    Fp16,
}

impl Ext {
    pub fn name(&self) -> &'static str {
        match self {
            Ext::Fp64 => "cl_khr_fp64",
            Ext::Fp16 => "cl_khr_fp16",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TypeInfo {
    pub dtype: DType,
    /// C type of a buffer element.
    pub storage: &'static str,
    /// C type of the value an expression computes with.
    pub math: &'static str,
    pub bytes: usize,
    pub class: Class,
    pub half: Option<Half>,
    pub ext: Option<Ext>,
    /// Bit width of the math value (32 for the half formats).
    pub math_bits: u32,
    /// Unsigned C type of the same width as `math`.
    pub unsigned: &'static str,
    /// Unsigned C type used for wrapping integer arithmetic: at least 32 bits.
    pub wide: &'static str,
    /// C expressions for the largest and smallest value (integers only).
    pub max: &'static str,
    pub min: &'static str,
}

impl TypeInfo {
    pub fn is_float(&self) -> bool {
        self.class == Class::Float
    }

    pub fn is_signed(&self) -> bool {
        self.class == Class::Signed
    }
}

fn int(
    dtype: DType,
    c: &'static str,
    unsigned: &'static str,
    wide: &'static str,
    class: Class,
    max: &'static str,
    min: &'static str,
) -> TypeInfo {
    TypeInfo {
        dtype,
        storage: c,
        math: c,
        bytes: (dtype.bits / 8) as usize,
        class,
        half: None,
        ext: None,
        math_bits: dtype.bits,
        unsigned,
        wide,
        max,
        min,
    }
}

fn float(
    dtype: DType,
    storage: &'static str,
    math: &'static str,
    half: Option<Half>,
    ext: Option<Ext>,
) -> TypeInfo {
    let math_bits = if math == "double" { 64 } else { 32 };
    TypeInfo {
        dtype,
        storage,
        math,
        bytes: (dtype.bits / 8) as usize,
        class: Class::Float,
        half,
        ext,
        math_bits,
        unsigned: if math_bits == 64 { "ulong" } else { "uint" },
        wide: if math_bits == 64 { "ulong" } else { "uint" },
        max: "",
        min: "",
    }
}

/// The table entry for `dtype`, or an error naming why it has none.
pub fn type_info(dtype: DType) -> Result<TypeInfo, String> {
    use Class::{Signed, Unsigned};
    let d = dtype;
    Ok(match (d.kind, d.bits) {
        (Kind::U, 8) => int(d, "uchar", "uchar", "uint", Unsigned, "255", "0"),
        (Kind::S, 8) => int(d, "char", "uchar", "uint", Signed, "127", "(-128)"),
        (Kind::U, 16) => int(d, "ushort", "ushort", "uint", Unsigned, "65535", "0"),
        (Kind::S, 16) => int(d, "short", "ushort", "uint", Signed, "32767", "(-32768)"),
        (Kind::U, 32) => int(d, "uint", "uint", "uint", Unsigned, "4294967295U", "0"),
        (Kind::S, 32) => int(
            d,
            "int",
            "uint",
            "uint",
            Signed,
            "2147483647",
            "(-2147483647-1)",
        ),
        (Kind::U, 64) => int(
            d,
            "ulong",
            "ulong",
            "ulong",
            Unsigned,
            "18446744073709551615UL",
            "0",
        ),
        (Kind::S, 64) => int(
            d,
            "long",
            "ulong",
            "ulong",
            Signed,
            "9223372036854775807L",
            "(-9223372036854775807L-1L)",
        ),
        (Kind::F, 16) => float(d, "half", "float", None, Some(Ext::Fp16)),
        (Kind::Bf, 16) => float(d, "ushort", "float", Some(Half::Bf16), None),
        (Kind::F, 32) => float(d, "float", "float", None, None),
        (Kind::F, 64) => float(d, "double", "double", None, Some(Ext::Fp64)),
        _ => return Err(format!("dtype {} has no OpenCL type yet", dtype)),
    })
}

/// Every dtype the table covers, in a fixed order.
#[cfg(test)]
pub fn supported() -> Vec<DType> {
    let mut all = Vec::new();
    for bits in [8, 16, 32, 64] {
        all.push(DType::new(Kind::U, bits));
        all.push(DType::new(Kind::S, bits));
    }
    all.push(DType::new(Kind::F, 16));
    all.push(DType::new(Kind::Bf, 16));
    all.push(DType::new(Kind::F, 32));
    all.push(DType::new(Kind::F, 64));
    all
}

// ── Constants ───────────────────────────────────────────

/// A scalar constant as the wire sends it: an integer, a float, or one of the
/// atoms `nan`, `infinity`, `neg_infinity`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstVal {
    Int(i128),
    Float(f64),
}

/// The constant as the bytes of one buffer element of `dtype` (native endian).
/// Integers wrap to the width of the type; a float value for an integer type
/// is an error.
pub fn encode_const(dtype: DType, value: ConstVal) -> Result<Vec<u8>, String> {
    let info = type_info(dtype)?;
    if info.is_float() {
        let v = match value {
            ConstVal::Int(i) => i as f64,
            ConstVal::Float(f) => f,
        };
        return Ok(match (dtype.kind, dtype.bits) {
            (Kind::F, 64) => v.to_ne_bytes().to_vec(),
            (Kind::F, 32) => (v as f32).to_ne_bytes().to_vec(),
            (Kind::F, 16) => f32_to_f16_bits(v as f32).to_ne_bytes().to_vec(),
            _ => f32_to_bf16_bits(v as f32).to_ne_bytes().to_vec(),
        });
    }
    let v = match value {
        ConstVal::Int(i) => i,
        ConstVal::Float(f) => {
            return Err(format!("float constant {} for integer type {}", f, dtype))
        }
    };
    let bytes = v.to_ne_bytes();
    let n = info.bytes;
    #[cfg(target_endian = "little")]
    return Ok(bytes[..n].to_vec());
    #[cfg(target_endian = "big")]
    return Ok(bytes[bytes.len() - n..].to_vec());
}

// ── Half-width conversions (host side; the kernels carry the same logic) ──

pub fn f32_to_bf16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub fn f32_to_f16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    let s = ((u >> 16) & 0x8000) as u16;
    let a = u & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return s | 0x7e00;
    }
    if a >= 0x477f_f000 {
        return s | 0x7c00;
    }
    if a < 0x3880_0000 {
        let exp = a >> 23;
        if exp < 102 {
            return s;
        }
        let m = (a & 0x007f_ffff) | 0x0080_0000;
        let n = 126 - exp;
        let mut r = m >> n;
        let rem = m & ((1 << n) - 1);
        let half = 1 << (n - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return s | r as u16;
    }
    let mut r = (a - 0x3800_0000) >> 13;
    let rem = a & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    s | r as u16
}

#[cfg(test)]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let mut e = ((h >> 10) & 0x1f) as i32;
    let mut m = (h & 0x3ff) as u32;
    let u = if e == 0 {
        if m == 0 {
            s
        } else {
            e = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            s | (((e + 112) as u32) << 23) | (m << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e + 112) as u32) << 23) | (m << 13)
    };
    f32::from_bits(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_an_entry_for_every_supported_dtype() {
        let all = supported();
        assert_eq!(all.len(), 12);
        for dtype in all {
            let info = type_info(dtype).unwrap();
            assert_eq!(info.dtype, dtype);
            assert!(!info.storage.is_empty() && !info.math.is_empty());
            assert!(info.bytes == (dtype.bits / 8) as usize);
        }
    }

    #[test]
    fn c_types_match_the_plan() {
        let cases = [
            (Kind::U, 8, "uchar"),
            (Kind::S, 8, "char"),
            (Kind::U, 16, "ushort"),
            (Kind::S, 16, "short"),
            (Kind::U, 32, "uint"),
            (Kind::S, 32, "int"),
            (Kind::U, 64, "ulong"),
            (Kind::S, 64, "long"),
            (Kind::F, 32, "float"),
            (Kind::F, 64, "double"),
        ];
        for (kind, bits, c) in cases {
            let info = type_info(DType::new(kind, bits)).unwrap();
            assert_eq!((info.storage, info.math), (c, c));
        }
    }

    #[test]
    fn bf16_is_stored_as_u16_and_computed_as_f32() {
        let info = type_info(DType::new(Kind::Bf, 16)).unwrap();
        assert_eq!(info.storage, "ushort");
        assert_eq!(info.math, "float");
        assert_eq!(info.bytes, 2);
        assert_eq!(info.half, Some(Half::Bf16));
        assert_eq!(info.ext, None);
    }

    #[test]
    fn f16_is_stored_as_half_and_needs_fp16() {
        let info = type_info(DType::new(Kind::F, 16)).unwrap();
        assert_eq!((info.storage, info.math), ("half", "float"));
        assert_eq!(info.half, None);
        assert_eq!(info.ext.map(|e| e.name()), Some("cl_khr_fp16"));
    }

    #[test]
    fn only_f64_and_f16_need_an_extension() {
        for dtype in supported() {
            let ext = type_info(dtype).unwrap().ext;
            if dtype == DType::new(Kind::F, 64) {
                assert_eq!(ext.map(|e| e.name()), Some("cl_khr_fp64"));
            } else if dtype == DType::new(Kind::F, 16) {
                assert_eq!(ext.map(|e| e.name()), Some("cl_khr_fp16"));
            } else {
                assert_eq!(ext, None, "{}", dtype);
            }
        }
    }

    #[test]
    fn unsupported_dtypes_have_no_entry() {
        for (kind, bits) in [
            (Kind::C, 64),
            (Kind::C, 128),
            (Kind::S, 4),
            (Kind::U, 2),
            (Kind::F, 8),
            (Kind::F8E4M3FN, 8),
        ] {
            assert!(type_info(DType::new(kind, bits)).is_err());
        }
    }

    #[test]
    fn wire_kinds_parse() {
        assert_eq!(DType::from_wire("bf", 16).unwrap().name(), "bf16");
        assert_eq!(DType::from_wire("s", 32).unwrap().name(), "s32");
        assert!(DType::from_wire("q", 8).is_err());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1 + 2^-8 sits halfway between two bf16 values: ties go to even.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_bits(f32::INFINITY)),
            f32::INFINITY
        );
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn f16_conversions_round_trip_and_round_to_nearest_even() {
        for h in 0..=u16::MAX {
            let f = f16_bits_to_f32(h);
            if f.is_nan() {
                assert!(f16_bits_to_f32(f32_to_f16_bits(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16_bits(f), h, "bits {:#06x}", h);
            }
        }
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(1.0e-8), 0);
        // 2^-24 is the smallest subnormal; 2^-25 is a tie that goes to even (0).
        assert_eq!(f32_to_f16_bits(5.960_464_5e-8), 1);
        assert_eq!(f32_to_f16_bits(2.980_232_2e-8), 0);
    }

    #[test]
    fn constants_wrap_to_the_width_of_the_type() {
        let u8t = DType::new(Kind::U, 8);
        assert_eq!(encode_const(u8t, ConstVal::Int(300)).unwrap(), vec![44]);
        let s16 = DType::new(Kind::S, 16);
        assert_eq!(
            encode_const(s16, ConstVal::Int(-2)).unwrap(),
            (-2i16).to_ne_bytes().to_vec()
        );
        let f32t = DType::new(Kind::F, 32);
        assert_eq!(
            encode_const(f32t, ConstVal::Float(1.5)).unwrap(),
            1.5f32.to_ne_bytes().to_vec()
        );
        assert!(encode_const(u8t, ConstVal::Float(1.5)).is_err());
        let u64t = DType::new(Kind::U, 64);
        assert_eq!(
            encode_const(u64t, ConstVal::Int(u64::MAX as i128)).unwrap(),
            u64::MAX.to_ne_bytes().to_vec()
        );
    }
}
