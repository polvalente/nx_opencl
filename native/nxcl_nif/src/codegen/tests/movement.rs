use super::super::test_support::{buf, instr, ints};
use super::super::types::{encode_const, ConstVal, Kind};
use super::super::Operand;
use super::*;

const S32: DType = DType::new(Kind::S, 32);
const S64: DType = DType::new(Kind::S, 64);
const U64: DType = DType::new(Kind::U, 64);
const F32: DType = DType::new(Kind::F, 32);
const F64: DType = DType::new(Kind::F, 64);
const C64: DType = DType::new(Kind::C, 64);
const C128: DType = DType::new(Kind::C, 128);
const BF16: DType = DType::new(Kind::Bf, 16);

fn konst(dtype: DType, v: i128) -> Operand {
    Operand {
        dtype,
        shape: vec![],
        constant: Some(encode_const(dtype, ConstVal::Int(v)).unwrap()),
    }
}

fn gen(i: &Instr) -> KernelSpec {
    generate(i, &Caps::all()).unwrap_or_else(|e| panic!("{}: {}", i.op, e))
}

/// The `long` arguments of the kernel, without the guard's `total`.
fn longs(spec: &KernelSpec) -> Vec<i64> {
    let mut all: Vec<i64> = spec
        .args
        .iter()
        .filter_map(|a| match a {
            KernelArg::Scalar(b) if b.len() == 8 => {
                Some(i64::from_ne_bytes(b[..].try_into().unwrap()))
            }
            _ => None,
        })
        .collect();
    let total = all.pop().expect("a guarded kernel ends with its total");
    assert_eq!(total.max(1) as usize, spec.global[0]);
    all
}

#[test]
fn broadcast_zeroes_the_stride_of_an_expanded_axis() {
    // {3, 1} to {2, 3, 4}: input axes 1 and 2.
    let i = instr(
        "broadcast",
        vec![buf(F32, &[3, 1])],
        F32,
        &[2, 3, 4],
        ints(&[1, 2]),
    );
    let spec = gen(&i);
    assert_eq!(spec.entry, "mv_broadcast_r3_f32_f32");
    assert_eq!(spec.global, vec![24]);
    // off, o0..o2, s0..s2
    assert_eq!(longs(&spec), vec![0, 2, 3, 4, 0, 1, 0]);
    assert!(spec.source.contains("out[i] = in0[j];"), "{}", spec.source);
}

#[test]
fn a_constant_broadcast_fills_with_no_index_math() {
    let i = instr("broadcast", vec![konst(S32, 7)], S32, &[5, 5], vec![]);
    let spec = gen(&i);
    assert_eq!(spec.entry, "mv_broadcast_fill_s32");
    assert!(spec.source.contains("out[i] = in0;"), "{}", spec.source);
    assert!(spec.source.contains("const int in0"), "{}", spec.source);
    assert_eq!(spec.args[0], KernelArg::Scalar(7i32.to_ne_bytes().to_vec()));
}

#[test]
fn transpose_uses_the_input_strides_in_axis_order() {
    let i = instr(
        "transpose",
        vec![buf(F32, &[2, 3, 4])],
        F32,
        &[4, 2, 3],
        ints(&[2, 0, 1]),
    );
    // off, o (4 2 3), s: input strides 12 4 1 picked by axes 2 0 1
    assert_eq!(longs(&gen(&i)), vec![0, 4, 2, 3, 1, 12, 4]);
}

#[test]
fn reverse_negates_the_stride_and_starts_at_the_far_end() {
    let i = instr("reverse", vec![buf(F32, &[2, 5])], F32, &[2, 5], ints(&[1]));
    // off = 4, o = 2 5, s = 5 -1
    assert_eq!(longs(&gen(&i)), vec![4, 2, 5, 5, -1]);
}

#[test]
fn bad_axes_return_errors() {
    let t = |axes: &[i64]| {
        instr(
            "transpose",
            vec![buf(F32, &[2, 3])],
            F32,
            &[3, 2],
            ints(axes),
        )
    };
    assert!(generate(&t(&[0, 0]), &Caps::all()).is_err());
    assert!(generate(&t(&[0]), &Caps::all()).is_err());
    let r = instr("reverse", vec![buf(F32, &[2])], F32, &[2], ints(&[3]));
    assert!(generate(&r, &Caps::all()).is_err());
    let b = instr("broadcast", vec![buf(F32, &[3])], F32, &[2, 4], ints(&[1]));
    assert!(generate(&b, &Caps::all()).is_err());
}

#[test]
fn slice_clamps_the_start_to_zero_and_dim_minus_length() {
    let i = instr(
        "slice",
        vec![buf(F32, &[5, 6]), konst(S64, -3), buf(S32, &[])],
        F32,
        &[2, 3],
        ints(&[2, 3, 1, 1]),
    );
    let spec = gen(&i);
    // The clamp reads `s < 0 ? 0 : (s > h ? h : s)` with h = dim - length.
    assert!(
        spec.source
            .contains("s0 = s0 < 0 ? 0 : (s0 > h0 ? h0 : s0);"),
        "{}",
        spec.source
    );
    assert!(
        spec.source.contains("long s1 = (long)(in2[0]);"),
        "{}",
        spec.source
    );
    // o 2 3, h 3 3, ss 6 1, is 6 1; the constant start is a scalar argument.
    assert_eq!(longs(&spec), vec![-3, 2, 3, 3, 3, 6, 1, 6, 1]);
}

#[test]
fn slice_stride_scales_the_output_step_and_checks_the_output_size() {
    let i = instr(
        "slice",
        vec![buf(F32, &[10]), konst(S64, 1)],
        F32,
        &[3],
        ints(&[7, 3]),
    );
    // length 7, stride 3 gives (7 - 1) / 3 + 1 = 3 elements; h = 3, ss = 3
    let spec = gen(&i);
    assert_eq!(longs(&spec)[1..], [3, 3, 3, 1]);
    let bad = instr(
        "slice",
        vec![buf(F32, &[10]), konst(S64, 1)],
        F32,
        &[4],
        ints(&[7, 3]),
    );
    assert!(generate(&bad, &Caps::all()).is_err());
}

#[test]
fn a_u64_start_compares_as_unsigned() {
    let i = instr(
        "slice",
        vec![buf(F32, &[4]), buf(U64, &[])],
        F32,
        &[2],
        ints(&[2, 1]),
    );
    let spec = gen(&i);
    assert!(
        spec.source.contains("ulong u0 = in1[0];"),
        "{}",
        spec.source
    );
    assert!(
        spec.source.contains("u0 > (ulong)h0 ? h0 : (long)u0"),
        "{}",
        spec.source
    );
}

#[test]
fn a_float_start_is_an_error() {
    let i = instr(
        "slice",
        vec![buf(F32, &[4]), buf(F32, &[])],
        F32,
        &[2],
        ints(&[2, 1]),
    );
    assert!(generate(&i, &Caps::all()).is_err());
}

#[test]
fn put_slice_clamps_and_reads_the_update_inside_the_window() {
    let i = instr(
        "put_slice",
        vec![buf(F32, &[5]), buf(F32, &[2]), buf(S32, &[])],
        F32,
        &[5],
        vec![],
    );
    let spec = gen(&i);
    assert!(
        spec.source
            .contains("s0 = s0 < 0 ? 0 : (s0 > h0 ? h0 : s0);"),
        "{}",
        spec.source
    );
    assert!(
        spec.source.contains("out[i] = inside ? in1[k] : in0[i];"),
        "{}",
        spec.source
    );
    // o 5, h 3, l 2, us 1
    assert_eq!(longs(&spec), vec![5, 3, 2, 1]);
    // The output is a new buffer: no argument aliases the output to an input.
    assert_eq!(
        spec.args
            .iter()
            .filter(|a| **a == KernelArg::Output)
            .count(),
        1
    );
}

#[test]
fn pad_has_a_branch_for_negative_low_and_high_padding() {
    // {4}: low -1, high -2, interior 1 gives -1 - 2 + 4 + 3 = 4 elements.
    let i = instr(
        "pad",
        vec![buf(F32, &[4]), konst(F32, 0)],
        F32,
        &[4],
        ints(&[-1, -2, 1]),
    );
    let spec = gen(&i);
    assert!(
        spec.source
            .contains("if (p < 0 || p >= ex0 || p % st0 != 0) fill = 1;"),
        "{}",
        spec.source
    );
    // o 4, lo -1, st 2, ex 7, is 1
    assert_eq!(longs(&spec), vec![4, -1, 2, 7, 1]);
    assert!(
        spec.source.contains("out[i] = fill ? in1 : in0[j];"),
        "{}",
        spec.source
    );
}

#[test]
fn pad_checks_the_output_shape() {
    let i = instr(
        "pad",
        vec![buf(F32, &[4]), konst(F32, 0)],
        F32,
        &[5],
        ints(&[1, 1, 0]),
    );
    assert!(generate(&i, &Caps::all()).is_err());
}

#[test]
fn concatenate_chains_one_branch_per_input_with_run_ends() {
    let i = instr(
        "concatenate",
        vec![buf(F32, &[2, 3]), buf(F32, &[2, 1]), buf(F32, &[2, 2])],
        F32,
        &[2, 6],
        ints(&[1]),
    );
    let spec = gen(&i);
    assert_eq!(spec.entry, "mv_concatenate_n3_f32");
    assert_eq!(spec.global, vec![12]);
    // inner 1, total 6, b 0 3 4, e 3 4 6, l 3 1 2
    assert_eq!(longs(&spec), vec![1, 6, 0, 3, 4, 3, 4, 6, 3, 1, 2]);
    assert!(
        spec.source.contains("if (a < e0) out[i] = in0["),
        "{}",
        spec.source
    );
    assert!(
        spec.source.contains("else if (a < e1) out[i] = in1["),
        "{}",
        spec.source
    );
    assert!(
        spec.source.contains("else out[i] = in2["),
        "{}",
        spec.source
    );
}

#[test]
fn stack_is_a_concatenate_of_runs_of_one() {
    let i = instr(
        "stack",
        vec![buf(F32, &[3]), buf(F32, &[3])],
        F32,
        &[3, 2],
        ints(&[1]),
    );
    let spec = gen(&i);
    // inner 1, total 2, b 0 1, e 1 2, l 1 1
    assert_eq!(longs(&spec), vec![1, 2, 0, 1, 1, 2, 1, 1]);
}

#[test]
fn a_stack_of_scalar_constants_reads_them_by_value() {
    let i = instr(
        "stack",
        vec![konst(S32, 1), konst(S32, 2)],
        S32,
        &[2],
        ints(&[0]),
    );
    let spec = gen(&i);
    assert!(spec.source.contains("out[i] = in0;"), "{}", spec.source);
    assert!(
        spec.source.contains("else out[i] = in1;"),
        "{}",
        spec.source
    );
}

#[test]
fn concatenate_rejects_a_mismatched_run() {
    let i = instr(
        "concatenate",
        vec![buf(F32, &[2]), buf(F32, &[2])],
        F32,
        &[5],
        ints(&[0]),
    );
    assert!(generate(&i, &Caps::all()).is_err());
}

#[test]
fn a_large_concatenate_has_source_linear_in_its_inputs() {
    let n = 64;
    let i = instr(
        "concatenate",
        (0..n).map(|_| buf(F32, &[1, 2])).collect(),
        F32,
        &[1, 2 * n],
        ints(&[1]),
    );
    let spec = gen(&i);
    assert!(spec.source.len() < 200 * n, "{} bytes", spec.source.len());
    // Operands, output, inner, span, the runs' begin, end and length, total.
    assert_eq!(spec.args.len(), n + 1 + 2 + 3 * n + 1);
}

#[test]
fn iota_with_an_axis_divides_by_the_inner_size() {
    let i = instr("iota", vec![], S32, &[2, 3, 4], ints(&[1]));
    let spec = gen(&i);
    assert!(
        spec.source.contains("((long)i / inner) % dim"),
        "{}",
        spec.source
    );
    assert_eq!(longs(&spec), vec![4, 3]);
}

#[test]
fn iota_without_an_axis_is_the_flat_index() {
    let i = instr("iota", vec![], F32, &[2, 3], vec![Attr::Atom("nil".into())]);
    let spec = gen(&i);
    assert!(
        spec.source.contains("out[i] = (float)((long)i);"),
        "{}",
        spec.source
    );
}

#[test]
fn iota_and_eye_build_values_in_the_output_type() {
    let c = gen(&instr("iota", vec![], C64, &[4], ints(&[0])));
    assert!(c.source.contains("(float2)((float)("), "{}", c.source);
    let b = gen(&instr("iota", vec![], BF16, &[4], ints(&[0])));
    assert!(b.source.contains("f32_to_bf16"), "{}", b.source);
    let e = gen(&instr("eye", vec![], C128, &[3, 3], vec![]));
    assert!(e.source.contains("(double2)(1.0, 0.0)"), "{}", e.source);
    assert!(
        e.source.contains("#pragma OPENCL EXTENSION cl_khr_fp64"),
        "{}",
        e.source
    );
}

#[test]
fn eye_takes_the_last_two_axes_as_rows_and_columns() {
    let i = instr("eye", vec![], F32, &[2, 3, 4], vec![]);
    let spec = gen(&i);
    assert_eq!(longs(&spec), vec![3, 4]);
    assert!(
        spec.source.contains("row == col ? (float)(1) : (float)(0)"),
        "{}",
        spec.source
    );
    assert!(generate(&instr("eye", vec![], F32, &[3], vec![]), &Caps::all()).is_err());
}

#[test]
fn element_width_comes_from_the_type_table() {
    for (dtype, storage) in [
        (C64, "float2"),
        (C128, "double2"),
        (BF16, "ushort"),
        (S64, "long"),
    ] {
        let i = instr(
            "transpose",
            vec![buf(dtype, &[2, 3])],
            dtype,
            &[3, 2],
            ints(&[1, 0]),
        );
        let spec = gen(&i);
        assert!(
            spec.source
                .contains(&format!("__global const {}* in0", storage)),
            "{}",
            spec.source
        );
        // The same template: no complex preamble.
        assert!(!spec.source.contains("cf_"), "{}", spec.source);
    }
}

#[test]
fn the_source_does_not_depend_on_the_shape() {
    let a = gen(&instr(
        "transpose",
        vec![buf(F32, &[2, 3])],
        F32,
        &[3, 2],
        ints(&[1, 0]),
    ));
    let b = gen(&instr(
        "transpose",
        vec![buf(F32, &[40, 7])],
        F32,
        &[7, 40],
        ints(&[1, 0]),
    ));
    assert_eq!(a.source, b.source);
}

#[test]
fn f64_and_c128_need_the_extension() {
    let no64 = Caps {
        fp64: false,
        fp16: true,
        int64_atomics: true,
    };
    for dtype in [F64, C128] {
        let i = instr("reverse", vec![buf(dtype, &[2])], dtype, &[2], ints(&[0]));
        let err = generate(&i, &no64).unwrap_err();
        assert!(err.contains("cl_khr_fp64"), "{}", err);
    }
}

#[test]
fn a_rank_above_the_cap_is_an_error_and_the_cap_itself_works() {
    let shape = vec![1usize; index::MAX_RANK + 1];
    let axes: Vec<i64> = (0..=index::MAX_RANK as i64).collect();
    let i = instr(
        "transpose",
        vec![buf(F32, &shape)],
        F32,
        &shape,
        ints(&axes),
    );
    let err = generate(&i, &Caps::all()).unwrap_err();
    assert!(err.contains("rank 9"), "{}", err);
    let shape = vec![1usize; index::MAX_RANK];
    let axes: Vec<i64> = (0..index::MAX_RANK as i64).collect();
    let i = instr(
        "transpose",
        vec![buf(F32, &shape)],
        F32,
        &shape,
        ints(&axes),
    );
    assert!(generate(&i, &Caps::all()).is_ok());
}

#[test]
fn a_zero_size_output_still_gets_a_valid_spec() {
    let i = instr("broadcast", vec![buf(F32, &[0])], F32, &[0], ints(&[0]));
    let spec = gen(&i);
    // The runtime skips the launch when the output has no bytes.
    assert_eq!(spec.global, vec![1]);
}

#[test]
fn entry_names_carry_the_op_rank_and_types() {
    let e = |op: &str, o: Vec<Operand>, shape: &[usize], attrs: Vec<Attr>| {
        gen(&instr(op, o, F32, shape, attrs)).entry
    };
    assert_eq!(
        e("reverse", vec![buf(F32, &[3])], &[3], ints(&[0])),
        "mv_reverse_r1_f32_f32"
    );
    assert_eq!(
        e(
            "pad",
            vec![buf(F32, &[2]), konst(F32, 0)],
            &[3],
            ints(&[1, 0, 0])
        ),
        "mv_pad_r1_f32_f32_f32"
    );
}

#[test]
fn every_listed_op_is_handled() {
    for op in OPS {
        assert!(handles(op));
    }
    assert!(!handles("add"));
}

#[test]
fn pad_and_slice_configs_at_the_limits_of_i64_are_errors_not_wraps() {
    let pad = |attrs: &[i64], out: &[usize]| {
        instr(
            "pad",
            vec![buf(F32, &[3]), buf(F32, &[])],
            F32,
            out,
            ints(attrs),
        )
    };
    // Interior padding of i64::MAX: the output size cannot match.
    let err = generate(&pad(&[0, 0, i64::MAX], &[5]), &Caps::all()).unwrap_err();
    assert!(err.contains("should have"), "{err}");
    // A one-element axis ignores interior padding, so only `i + 1` can wrap.
    let one = instr(
        "pad",
        vec![buf(F32, &[1]), buf(F32, &[])],
        F32,
        &[1],
        ints(&[0, 0, i64::MAX]),
    );
    let err = generate(&one, &Caps::all()).unwrap_err();
    assert!(err.contains("too large"), "{err}");
    // A stride times the axis stride overflows.
    let slice = instr(
        "slice",
        vec![buf(F32, &[2, 3]), buf(S32, &[]), buf(S32, &[])],
        F32,
        &[1, 1],
        ints(&[1, 1, i64::MAX, 1]),
    );
    let err = generate(&slice, &Caps::all()).unwrap_err();
    assert!(err.contains("too large"), "{err}");
}

#[test]
fn an_empty_tensor_has_a_positive_global_size() {
    // The executor never launches this, and the spec stays well formed.
    let i = instr(
        "transpose",
        vec![buf(F32, &[0, 4])],
        F32,
        &[4, 0],
        ints(&[1, 0]),
    );
    assert!(gen(&i).global.iter().all(|&g| g > 0));
}

#[test]
fn a_padding_beyond_32_bits_is_an_error() {
    for lo in [i64::MIN, i64::MAX, 1 << 32] {
        let i = instr(
            "pad",
            vec![buf(F32, &[3]), konst(F32, 0)],
            F32,
            &[3],
            ints(&[lo, 0, 0]),
        );
        let err = generate(&i, &Caps::all()).unwrap_err();
        assert!(err.contains("does not fit in 32 bits"), "{err}");
    }
}

#[test]
fn every_guarded_kernel_tests_the_item_before_anything_else() {
    let i = instr(
        "broadcast",
        vec![buf(F32, &[3, 1])],
        F32,
        &[2, 3, 4],
        ints(&[1, 2]),
    );
    let spec = gen(&i);
    let body = spec.source.split_once(") {\n").unwrap().1;
    assert!(
        body.starts_with("  if ((long)get_global_id(0) >= total) return;\n"),
        "{}",
        spec.source
    );
    // An empty output gets a one-item launch and a total of 0.
    let empty = gen(&instr(
        "broadcast",
        vec![buf(F32, &[3, 1])],
        F32,
        &[0, 3, 4],
        ints(&[1, 2]),
    ));
    assert_eq!(empty.global, vec![1]);
    let last = empty.args.last().unwrap().clone();
    assert_eq!(last, KernelArg::Scalar(0i64.to_ne_bytes().to_vec()));
}

/// The per-axis rule of the `pad` kernel, on the host: the input index an
/// output coordinate reads, or `None` where it holds the pad value.
fn pad_source(c: i64, lo: i64, step: i64, extent: i64) -> Option<i64> {
    let p = c - lo;
    if p < 0 || p >= extent || p % step != 0 {
        None
    } else {
        Some(p / step)
    }
}

#[test]
fn the_pad_rule_matches_a_direct_pad() {
    // Input [a b c d], interior 1, low -1, high 2: drop `a`, spread, extend.
    let input = ['a', 'b', 'c', 'd'];
    let (lo, hi, interior) = (-1i64, 2i64, 1i64);
    let spread: Vec<Option<char>> = input
        .iter()
        .flat_map(|&x| [Some(x), None])
        .take(input.len() * 2 - 1)
        .collect();
    let want: Vec<Option<char>> = (0..(spread.len() as i64 + lo + hi))
        .map(|c| {
            let p = c - lo;
            if p >= 0 && (p as usize) < spread.len() {
                spread[p as usize]
            } else {
                None
            }
        })
        .collect();
    let step = interior + 1;
    let extent = (input.len() as i64 - 1) * step + 1;
    let got: Vec<Option<char>> = (0..want.len() as i64)
        .map(|c| pad_source(c, lo, step, extent).map(|j| input[j as usize]))
        .collect();
    assert_eq!(got, want);
}
