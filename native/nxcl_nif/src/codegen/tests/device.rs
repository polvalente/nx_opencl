//! A sample of generated kernels: every kernel guards its items first, and each
//! one builds on a real OpenCL device. The build test passes without a device:
//! it skips when none opens.

use super::test_support::{buf, instr, ints};
use super::test_support::{plan, starts_guarded};
use super::types::{DType, Kind};
use super::{generate_plan, Attr, Instr};
use crate::device_ctx::DeviceCtx;
use crate::kernel::CompiledKernel;

const F32: DType = DType::new(Kind::F, 32);
const C64: DType = DType::new(Kind::C, 64);

/// The first device, or `None` when there is no OpenCL library or device.
fn device() -> Option<DeviceCtx> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        DeviceCtx::new(None, 0).ok()
    }))
    .ok()
    .flatten()
}

fn samples() -> Vec<Instr> {
    let window = ints(&[2, 2, 1, 1, 1, 1, 0, 0, 0, 0]);
    let gemm = |m, n, k| {
        let attrs = [1, m, n, k].iter().map(|&v| Attr::Int(v as i64)).collect();
        instr(
            "batched_gemm",
            vec![buf(F32, &[1, m, k]), buf(F32, &[1, k, n])],
            F32,
            &[1, m, n],
            attrs,
        )
    };
    vec![
        instr(
            "add",
            vec![buf(F32, &[4]), buf(F32, &[4])],
            F32,
            &[4],
            vec![],
        ),
        instr(
            "broadcast",
            vec![buf(F32, &[3, 1])],
            F32,
            &[2, 3, 4],
            ints(&[1, 2]),
        ),
        instr(
            "pad",
            vec![buf(F32, &[3]), buf(F32, &[])],
            F32,
            &[5],
            ints(&[1, 1, 0]),
        ),
        instr(
            "concatenate",
            vec![buf(F32, &[2, 3]), buf(F32, &[2, 2])],
            F32,
            &[2, 5],
            ints(&[1]),
        ),
        instr("sum", vec![buf(F32, &[3, 20])], F32, &[3], ints(&[1])),
        instr("sum", vec![buf(F32, &[2, 100_000])], F32, &[2], ints(&[1])),
        instr(
            "cumulative_sum",
            vec![buf(F32, &[2, 600])],
            F32,
            &[2, 600],
            ints(&[1, 0]),
        ),
        instr(
            "sort",
            vec![buf(F32, &[2, 5000])],
            F32,
            &[2, 5000],
            vec![Attr::Int(1), Attr::Atom("asc".to_string())],
        ),
        instr("window_sum", vec![buf(F32, &[6, 8])], F32, &[5, 7], window),
        gemm(37, 41, 29),
        gemm(1, 33, 20),
        instr("fft", vec![buf(C64, &[64])], C64, &[64], ints(&[0])),
        instr("fft", vec![buf(C64, &[30])], C64, &[30], ints(&[0])),
    ]
}

#[test]
fn sample_kernels_build_on_a_device_when_there_is_one() {
    let Some(ctx) = device() else {
        eprintln!("no OpenCL device: skipped");
        return;
    };
    for i in samples() {
        let plan =
            generate_plan(&i, &ctx.caps, &ctx.limits).unwrap_or_else(|e| panic!("{}: {}", i.op, e));
        for pass in &plan.passes {
            CompiledKernel::compile(&ctx, &pass.source, &pass.entry, &pass.build_options)
                .unwrap_or_else(|e| panic!("{} / {}: {}", i.op, pass.entry, e));
        }
    }
}

#[test]
fn per_item_and_per_group_kernels_start_with_their_guard() {
    // The tiled GEMM, `conv` and the FFT passes write their own guards or need none.
    for i in samples()
        .iter()
        .filter(|i| !matches!(i.op.as_str(), "batched_gemm" | "fft"))
    {
        for pass in &plan(i).passes {
            assert!(starts_guarded(&pass.source), "{} / {}", i.op, pass.entry);
        }
    }
}
