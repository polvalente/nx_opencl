use super::super::test_support::{buf, ints, plan};
use super::super::types::Kind as K;
use super::*;

const S32: DType = DType::new(K::S, 32);
const S64: DType = DType::new(K::S, 64);
const U8: DType = DType::new(K::U, 8);
const U64: DType = DType::new(K::U, 64);
const F32: DType = DType::new(K::F, 32);
const F64: DType = DType::new(K::F, 64);
const BF16: DType = DType::new(K::Bf, 16);
const C64: DType = DType::new(K::C, 64);

fn instr(
    op: &str,
    input: DType,
    shape: &[usize],
    out: DType,
    out_shape: &[usize],
    attrs: Vec<Attr>,
) -> Instr {
    Instr {
        op: op.to_string(),
        operands: vec![buf(input, shape)],
        out,
        out_shape: out_shape.to_vec(),
        attrs,
    }
}

fn sources(p: &Plan) -> Vec<&str> {
    p.passes.iter().map(|s| s.source.as_str()).collect()
}

#[test]
fn strategy_follows_extent_and_output_count() {
    let wg = 256;
    // Up to 64 elements one thread folds a line.
    assert_eq!(strategy(1, 1, wg), Strategy::Thread);
    assert_eq!(strategy(63, 1, wg), Strategy::Thread);
    assert_eq!(strategy(64, 1, wg), Strategy::Thread);
    // Past that, a group folds a line; 65 elements need one slice only.
    assert_eq!(strategy(65, 1, wg), Strategy::Group);
    assert_eq!(strategy(4096, 1, wg), Strategy::Group);
    // 4097 elements at 16 per item need two slices.
    assert_eq!(strategy(4097, 1, wg), Strategy::Sliced(2));
    assert_eq!(strategy(1 << 20, 1, wg), Strategy::Sliced(256));
    // Many outputs fill the device with one group each, however long the line.
    assert_eq!(strategy(4097, 64, wg), Strategy::Group);
    assert_eq!(strategy(1 << 20, 4096, wg), Strategy::Group);
    assert_eq!(strategy(4097, 63, wg), Strategy::Sliced(2));
    // The slice count stops at 1024.
    assert_eq!(strategy(1 << 30, 1, wg), Strategy::Sliced(1024));
    // A smaller group gives more slices.
    assert_eq!(strategy(4097, 1, 64), Strategy::Sliced(5));
}

#[test]
fn accumulator_has_the_width_of_the_nx_output_type() {
    let acc = |out: DType| {
        combiner(&CombKind::Canon(Canon::Sum), out)
            .unwrap()
            .acc
            .math
    };
    assert_eq!(acc(S32), "int");
    assert_eq!(acc(S64), "long");
    assert_eq!(acc(U64), "ulong");
    assert_eq!(acc(F32), "float");
    assert_eq!(acc(F64), "double");
    // Half-width floats accumulate in float.
    assert_eq!(acc(BF16), "float");
    assert_eq!(acc(C64), "float2");

    // `sum` of `u8` has output type `u64`: the kernel folds in `ulong`.
    let p = plan(&instr("sum", U8, &[100], U64, &[], ints(&[0])));
    assert!(sources(&p)[0].contains("ulong acc"));
}

#[test]
fn identities_per_op_and_dtype() {
    let id = |c, d| identity_of(c, &type_info(d).unwrap()).unwrap();
    assert_eq!(id(Canon::Sum, F32), "(float)0");
    assert_eq!(id(Canon::Product, F32), "(float)1");
    assert_eq!(id(Canon::Max, F32), "(-INFINITY)");
    assert_eq!(id(Canon::Min, F32), "INFINITY");
    assert_eq!(id(Canon::Sum, S64), "(long)0");
    // Integers use the limits of their own type.
    for d in [S32, S64, U8, U64] {
        let info = type_info(d).unwrap();
        assert_eq!(id(Canon::Max, d), info.min);
        assert_eq!(id(Canon::Min, d), info.max);
    }
    // Complex values have no order, so max and min have no identity.
    let c = type_info(C64).unwrap();
    assert!(identity_of(Canon::Max, &c).is_err());
    assert!(identity_of(Canon::Sum, &c).is_ok());
}

#[test]
fn argmax_orders_nan_first_and_ties_by_tie_break() {
    let src = |op: &str, tie: &str| {
        let a = vec![Attr::Atom(tie.to_string()), Attr::Int(0)];
        sources(&plan(&instr(op, F32, &[100], S64, &[], a)))[0].to_string()
    };
    // NaN beats a number; of two NaNs the later index wins.
    assert!(src("argmax", "low").contains("if (isnan(v)) return isnan(w) ? i > k : 1;"));
    assert!(src("argmax", "low").contains("if (isnan(w)) return 0;"));
    // Ties go to the lowest or highest index.
    assert!(src("argmax", "low").contains("return i < k;"));
    assert!(src("argmax", "high").contains("return i > k;"));
    assert!(src("argmax", "low").contains("v > w"));
    assert!(src("argmin", "low").contains("v < w"));
    // Integers have no NaN arm.
    let a = vec![Attr::Atom("low".to_string()), Attr::Int(0)];
    let int = plan(&instr("argmax", S32, &[100], S64, &[], a));
    assert!(!sources(&int)[0].contains("isnan"));
    // The tie break is part of the kernel hash.
    assert_ne!(src("argmax", "low"), src("argmax", "high"));
}

#[test]
fn reduce_with_an_add_body_generates_the_source_of_sum() {
    let sum = plan(&instr("sum", F32, &[3, 100], F32, &[3], ints(&[1])));
    // One step, `add(v0, v1)`, then the axes.
    let attrs = vec![
        Attr::Int(1),
        Attr::Atom("add".to_string()),
        Attr::Int(0),
        Attr::Int(1),
        Attr::Int(1),
    ];
    let reduce = plan(&instr("reduce", F32, &[3, 100], F32, &[3], attrs));
    assert_eq!(sources(&sum), sources(&reduce));
    assert_eq!(
        sum.passes.iter().map(|p| &p.entry).collect::<Vec<_>>(),
        reduce.passes.iter().map(|p| &p.entry).collect::<Vec<_>>()
    );
}

#[test]
fn a_body_without_an_initial_value_must_be_a_monoid() {
    let attrs = vec![
        Attr::Int(2),
        Attr::Atom("multiply".to_string()),
        Attr::Int(0),
        Attr::Int(0),
        Attr::Atom("add".to_string()),
        Attr::Int(2),
        Attr::Int(1),
        Attr::Int(1),
    ];
    let err = generate(
        &instr("reduce", F32, &[3, 100], F32, &[3], attrs),
        &Caps::all(),
        &Limits::apple(),
    )
    .unwrap_err();
    assert!(err.contains("initial accumulator"), "{err}");
}

#[test]
fn plans_have_the_passes_the_strategy_names() {
    // One thread per line.
    let p = plan(&instr("sum", F32, &[3, 20], F32, &[3], ints(&[1])));
    assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
    // One group per line.
    let p = plan(&instr("sum", F32, &[128, 1000], F32, &[128], ints(&[1])));
    assert_eq!((p.passes.len(), p.scratch.len()), (1, 0));
    assert_eq!(p.passes[0].local, vec![256]);
    // Slices: partials in scratch, then a fold of them.
    let p = plan(&instr("sum", F32, &[2, 100_000], F32, &[2], ints(&[1])));
    assert_eq!(p.passes.len(), 2);
    assert_eq!(p.scratch, vec![2 * 25 * 4]);
    // The kernel never uses atomics.
    for s in sources(&p) {
        assert!(!s.contains("atomic"));
    }
}

#[test]
fn group_size_obeys_the_device_limits() {
    // A device with 64 threads and 1 KiB of local memory.
    let small = Limits::new(64, 1024);
    let i = instr("sum", F64, &[2, 100_000], F64, &[2], ints(&[1]));
    let p = generate(&i, &Caps::all(), &small).unwrap();
    for s in &p.passes {
        let wg = s.local.first().copied().unwrap_or(1);
        assert!(wg <= 64 && wg * 8 <= 1024, "{wg}");
    }
}

#[test]
fn an_empty_axis_fills_with_the_identity_and_arg_ops_refuse() {
    let p = plan(&instr("reduce_max", F32, &[3, 0], F32, &[3], ints(&[1])));
    assert_eq!(p.passes.len(), 1);
    assert!(p.passes[0].source.contains("-INFINITY"));
    let a = vec![Attr::Atom("low".to_string()), Attr::Int(1)];
    let err = generate(
        &instr("argmax", F32, &[3, 0], S64, &[3], a),
        &Caps::all(),
        &Limits::apple(),
    );
    assert!(err.unwrap_err().contains("empty axis"));
}

#[test]
fn a_step_count_beyond_the_attrs_is_an_error_not_an_allocation() {
    let mut attrs = vec![Attr::Int(i64::MAX)];
    attrs.extend([Attr::Atom("add".to_string()), Attr::Int(0), Attr::Int(1)]);
    let err = parse_steps(&attrs).unwrap_err();
    assert!(err.contains("steps need"), "{err}");
    // One step is declared and none follows.
    let err = parse_steps(&ints(&[1])).unwrap_err();
    assert!(err.contains("steps need"), "{err}");
    assert!(parse_steps(&ints(&[0])).is_err());
    assert!(parse_steps(&ints(&[-3])).is_err());
}

#[test]
fn an_extent_that_overflows_is_an_error() {
    let big = 1usize << 40;
    let i = instr("sum", F32, &[big, big], F32, &[], ints(&[0, 1]));
    let err = generate(&i, &Caps::all(), &Limits::apple()).unwrap_err();
    assert!(err.contains("the reduced extent is too large"), "{err}");
}

#[test]
fn an_empty_output_still_has_a_positive_global_size() {
    // Three empty lines of 5: the executor never launches this, and the
    // spec stays well formed.
    let p = plan(&instr("sum", F32, &[0, 5], F32, &[0], ints(&[1])));
    assert!(p.passes.iter().all(|s| s.global.iter().all(|&g| g > 0)));
    let p = plan(&instr("sum", F32, &[0, 5000], F32, &[0], ints(&[1])));
    assert!(p.passes.iter().all(|s| s.global.iter().all(|&g| g > 0)));
}
