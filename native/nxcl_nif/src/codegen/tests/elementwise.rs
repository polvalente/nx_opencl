use super::super::types::{encode_const, supported, Class, ConstVal};
use super::super::{Attr, KernelArg, Operand};
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
    let ops =
        |op: &str, d: DType| generate(&instr(op, vec![buf(d), buf(d)], d, vec![]), &Caps::all());
    assert!(ops("divide", S32).is_err());
    assert!(ops("quotient", F32).is_err());
}

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
        int64_atomics: true,
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
        int64_atomics: true,
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
            KernelArg::Output,
            KernelArg::Scalar(4i64.to_ne_bytes().to_vec()),
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
    assert!(err.contains("must match the output"), "{}", err);
}

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
        int64_atomics: true,
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
