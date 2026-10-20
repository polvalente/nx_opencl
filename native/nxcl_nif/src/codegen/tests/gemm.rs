use super::super::test_support::buf;
use super::super::types::Kind;
use super::*;

const F32: DType = DType::new(Kind::F, 32);
const S32: DType = DType::new(Kind::S, 32);
const C64: DType = DType::new(Kind::C, 64);

fn limits(max_wg: usize, local_mem: usize) -> Limits {
    Limits::new(max_wg, local_mem)
}

fn gemm(dtype: DType, batch: usize, m: usize, n: usize, k: usize) -> Instr {
    let op = |shape: [usize; 3]| buf(dtype, &shape);
    Instr {
        op: "batched_gemm".to_string(),
        operands: vec![op([batch, m, k]), op([batch, k, n])],
        out: dtype,
        out_shape: vec![batch, m, n],
        attrs: [batch, m, n, k]
            .iter()
            .map(|&v| Attr::Int(v as i64))
            .collect(),
    }
}

fn spec(i: &Instr, l: &Limits) -> KernelSpec {
    let plan = kernel(
        &Gemm {
            dtype: i.out,
            batch: i.out_shape[0],
            m: i.out_shape[1],
            n: i.out_shape[2],
            k: i.operands[0].shape[2],
        },
        &Binding {
            a: KernelArg::Operand(0),
            b: KernelArg::Operand(1),
            out: KernelArg::Output,
        },
        &Caps::all(),
        l,
    );
    plan.unwrap()
}

#[test]
fn a_big_problem_gets_the_largest_tile_and_a_medium_one_a_smaller_tile() {
    let l = limits(256, 48 * 1024);
    assert_eq!(select_tile(&l, 4, (1, 2048, 2048)).unwrap(), TABLE[0]);
    // 512 x 512 makes 16 groups of 128 x 128 and 64 of 64 x 64: fewer than
    // the target, so the 64 x 32 tile (128 groups) wins.
    assert_eq!(select_tile(&l, 4, (1, 512, 512)).unwrap(), TABLE[2]);
    // A batch adds groups, so a large tile serves small products.
    assert_eq!(select_tile(&l, 4, (128, 128, 128)).unwrap(), TABLE[0]);
}

#[test]
fn a_device_with_wide_groups_takes_the_cpu_table() {
    let t = select_tile(&limits(4096, 64 * 1024), 4, (1, 2048, 2048)).unwrap();
    assert_eq!(t, WIDE_GROUP_TABLE[0]);
    let small = select_tile(&limits(4096, 64 * 1024), 4, (1, 7, 7)).unwrap();
    assert!(small.bm <= 16);
}

#[test]
fn small_local_memory_picks_a_smaller_tile() {
    // The largest tile needs 8 KiB for f32; 2 KiB rules it out.
    let t = select_tile(&limits(256, 2 * 1024), 4, (1, 2048, 2048)).unwrap();
    assert!(t.bm < 128 && t.local_bytes(4) <= 2 * 1024);
    // Wide values take twice the room, so the choice shrinks again.
    let w = select_tile(&limits(256, 2 * 1024), 16, (1, 2048, 2048)).unwrap();
    assert!(w.local_bytes(16) <= 2 * 1024 && w.bm <= t.bm);
}

#[test]
fn a_small_group_limit_picks_a_tile_with_few_threads() {
    let t = select_tile(&limits(64, 32 * 1024), 4, (1, 512, 512)).unwrap();
    assert!(t.threads() <= 64);
    let one = select_tile(&limits(1, 32 * 1024), 4, (1, 512, 512)).unwrap();
    assert_eq!(one.threads(), 1);
}

#[test]
fn a_tiny_problem_does_not_take_the_largest_tile() {
    let t = select_tile(&limits(256, 32 * 1024), 4, (1, 7, 7)).unwrap();
    assert!(t.bm <= 16 && t.bn <= 16, "{:?}", t);
}

#[test]
fn every_tile_in_the_table_is_valid() {
    for t in TABLE {
        assert!(t.valid(), "{:?}", t);
    }
}

#[test]
fn nothing_fits_without_local_memory() {
    assert!(select_tile(&limits(256, 4), 4, (1, 64, 64)).is_err());
}

#[test]
fn the_tile_is_in_the_build_options() {
    let s = spec(&gemm(F32, 1, 2048, 2048, 64), &limits(256, 32 * 1024));
    assert_eq!(s.build_options, TABLE[0].options());
    assert!(s.build_options.contains("-DBM=128"));
    let small = spec(&gemm(F32, 1, 2048, 2048, 64), &limits(256, 2 * 1024));
    assert_ne!(small.build_options, s.build_options);
}

#[test]
fn the_tiled_kernel_bounds_checks_and_sizes_the_launch() {
    let s = spec(&gemm(F32, 128, 70, 33, 20), &limits(256, 32 * 1024));
    assert!(s.entry.starts_with("gemm_tiled_"));
    assert!(s.source.contains("gr < M && gk < K") && s.source.contains("gc < N"));
    // The batch of 128 leaves enough groups for the 128 by 128 tile, with 16 by 16
    // threads: 33 columns and 70 rows each need one group, and the batch is the third axis.
    assert_eq!(s.build_options, TABLE[0].options());
    assert_eq!(s.global, vec![16, 16, 128]);
    assert_eq!(s.local, vec![16, 16, 1]);
    assert_eq!(s.args.len(), 6);
}

#[test]
fn n_equal_one_uses_the_matrix_vector_kernel() {
    let s = spec(&gemm(F32, 2, 9, 1, 100), &limits(256, 32 * 1024));
    assert!(s.entry.starts_with("gemm_matvec_"));
    assert_eq!(s.build_options, "-DWG=128");
    assert_eq!(s.global, vec![9 * 128, 2]);
    assert_eq!(s.local, vec![128, 1]);
    // A vector dot a vector takes the same path.
    assert!(spec(&gemm(F32, 1, 1, 1, 5), &limits(256, 32 * 1024))
        .entry
        .starts_with("gemm_matvec_"));
}

#[test]
fn m_equal_one_uses_the_vector_matrix_kernel() {
    let s = spec(&gemm(F32, 2, 1, 40, 30), &limits(256, 32 * 1024));
    assert!(s.entry.starts_with("gemm_vecmat_"));
    assert_eq!(s.global, vec![40, 2]);
    assert!(s.local.is_empty());
}

#[test]
fn k_equal_zero_fills_zeros_without_a_gemm_kernel() {
    let s = spec(&gemm(F32, 2, 3, 4, 0), &limits(256, 32 * 1024));
    assert!(s.entry.starts_with("gemm_zero_"));
    assert!(!s.source.contains("gemm_mac"));
    assert_eq!(s.global, vec![24]);
    assert_eq!(s.args.len(), 2);
}

#[test]
fn integers_accumulate_in_a_wide_unsigned_type() {
    let s = spec(&gemm(S32, 1, 8, 8, 8), &limits(256, 32 * 1024));
    assert!(s.source.contains("static inline uint gemm_mac(uint acc"));
    assert!(s.source.contains("static inline int gemm_st(uint v)"));
    let s8 = spec(
        &gemm(DType::new(Kind::S, 8), 1, 8, 8, 8),
        &limits(256, 32 * 1024),
    );
    assert!(s8.source.contains("uint gemm_ld(char x)"));
}

#[test]
fn bf16_and_f16_accumulate_in_f32() {
    let bf = spec(
        &gemm(DType::new(Kind::Bf, 16), 1, 8, 8, 8),
        &limits(256, 32 * 1024),
    );
    assert!(bf.source.contains("float gemm_mac(float acc"));
    assert!(bf.source.contains("bf16_to_f32"));
    let h = spec(
        &gemm(DType::new(Kind::F, 16), 1, 8, 8, 8),
        &limits(256, 32 * 1024),
    );
    assert!(h.source.contains("float gemm_mac(float acc"));
    assert!(h.source.contains("cl_khr_fp16"));
}

#[test]
fn complex_uses_the_complex_helpers() {
    let s = spec(&gemm(C64, 1, 8, 8, 8), &limits(256, 32 * 1024));
    assert!(s.source.contains("cf_add(acc, cf_mul(a, b))"));
    assert!(s.source.contains("float2 cf_mul(float2 a, float2 b)"));
    assert!(s.source.starts_with("#pragma OPENCL FP_CONTRACT OFF"));
}

#[test]
fn wide_values_take_a_smaller_tile_when_local_memory_is_tight() {
    let c128 = DType::new(Kind::C, 128);
    let big = spec(&gemm(c128, 1, 2048, 2048, 64), &limits(256, 32 * 1024));
    let s = spec(&gemm(c128, 1, 2048, 2048, 64), &limits(256, 16 * 1024));
    assert_ne!(s.build_options, big.build_options);
    assert!(s.source.contains("cl_khr_fp64"));
}

#[test]
fn the_instruction_checks_types_and_shapes() {
    let ok = gemm(F32, 1, 4, 5, 6);
    assert!(generate(&ok, &Caps::all(), &Limits::apple()).is_ok());
    let mut bad = gemm(F32, 1, 4, 5, 6);
    bad.operands[1].shape = vec![1, 7, 5];
    assert!(generate(&bad, &Caps::all(), &Limits::apple())
        .unwrap_err()
        .contains("do not match"));
    let mut mixed = gemm(F32, 1, 4, 5, 6);
    mixed.operands[0].dtype = S32;
    assert!(generate(&mixed, &Caps::all(), &Limits::apple())
        .unwrap_err()
        .contains("output type"));
}

#[test]
fn f64_needs_the_extension() {
    let f64t = DType::new(Kind::F, 64);
    let none = Caps {
        fp64: false,
        fp16: false,
        int64_atomics: true,
    };
    assert!(generate(&gemm(f64t, 1, 4, 4, 4), &none, &Limits::apple())
        .unwrap_err()
        .contains("cl_khr_fp64"));
}

#[test]
fn each_side_of_the_group_obeys_its_own_work_item_limit() {
    // 256 items in total, but at most 4 along each of the first two axes.
    let mut l = limits(256, 32 * 1024);
    l.max_item_sizes = [4, 4, 1];
    let t = select_tile(&l, 4, (1, 512, 512)).unwrap();
    assert!(t.bn / t.tn <= 4 && t.bm / t.tm <= 4, "{:?}", t);
    // The 128 x 128 tile has 16 x 16 items and does not fit.
    assert!(!TABLE[0].fits(&l, 4));
}

#[test]
fn sizes_that_overflow_are_errors() {
    let big = 1usize << 30;
    let mut i = gemm(F32, 1, 1, 1, 1);
    i.operands = vec![buf(F32, &[big, big, 1]), buf(F32, &[big, 1, big])];
    i.out_shape = vec![big, big, big];
    i.attrs = vec![
        Attr::Int(big as i64),
        Attr::Int(big as i64),
        Attr::Int(big as i64),
        Attr::Int(1),
    ];
    let err = generate(&i, &Caps::all(), &limits(256, 32 * 1024)).unwrap_err();
    assert!(err.contains("the gemm output size is too large"), "{err}");
    // A negative size is not a size.
    let mut i = gemm(F32, 1, 2, 2, 2);
    i.attrs[1] = Attr::Int(-1);
    assert!(generate(&i, &Caps::all(), &limits(256, 32 * 1024))
        .unwrap_err()
        .contains("must be a size"));
}

#[test]
fn an_unknown_c_type_has_no_size() {
    assert_eq!(c_type_bytes("float2"), Ok(8));
    assert_eq!(c_type_bytes("double2"), Ok(16));
    assert!(c_type_bytes("quad").unwrap_err().contains("quad"));
}
