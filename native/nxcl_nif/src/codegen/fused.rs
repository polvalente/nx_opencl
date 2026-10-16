//! Fused elementwise kernels. A group of elementwise steps of one output shape
//! becomes one kernel: each step is a static function built from the same
//! templates as the unfused kernel for its op (`elementwise::scalar_step`), so
//! a fused op computes what its unfused form computes. Group inputs are
//! buffers read at the work item's index, buffers read through a broadcast
//! index mapping, or scalar constants written as literals. The kernel has one
//! output buffer, always fresh, and writes it once.

use super::elementwise::{scalar_step, ScalarCode};
use super::index::{check_rank, coords, dims, strides};
use super::types::{type_info, DType, TypeInfo};
use super::{element_count, Attr, Caps, Instr, KernelArg, KernelSpec};

/// How a group input is read.
#[derive(Clone, Debug, PartialEq)]
pub enum InMap {
    /// Element `i` of the buffer, which has the output shape.
    Direct,
    /// The buffer has its own shape. Operand axis `j` fills output axis
    /// `axes[j]`; an axis of size 1 repeats.
    Broadcast(Vec<usize>),
}

/// Where a step reads a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Src {
    /// The k-th group input.
    In(usize),
    /// The result of an earlier step.
    Step(usize),
}

#[derive(Clone, Debug)]
pub struct Step {
    pub op: String,
    pub srcs: Vec<Src>,
    /// The type the step produces.
    pub dtype: DType,
    pub attrs: Vec<Attr>,
}

/// A group: how each operand of the instruction is read, and the steps in
/// order. The last step is the result.
#[derive(Clone, Debug)]
pub struct Fused {
    pub maps: Vec<InMap>,
    pub steps: Vec<Step>,
}

/// Kernel arguments of a group: one per buffer input, the output, and for
/// broadcast reads the output dimensions plus one stride per axis for each
/// broadcast input. Constants are literals and take none.
pub fn arg_count(buffers: usize, rank: usize, broadcasts: usize) -> usize {
    let index = if broadcasts > 0 {
        rank * (1 + broadcasts)
    } else {
        0
    };
    buffers + 1 + index
}

/// Hash of the text, for the entry name: equal groups get equal names.
fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A scalar constant as a C literal of its storage type, written from its
/// bytes so the value is exact (NaN, infinities and every integer width).
fn literal(info: &TypeInfo, bytes: &[u8]) -> Result<String, String> {
    let wrong = || format!("a constant of {} has {} bytes", info.dtype, bytes.len());
    let f32_lit = |b: &[u8]| -> Result<String, String> {
        let a: [u8; 4] = b.try_into().map_err(|_| wrong())?;
        Ok(format!("as_float(0x{:08x}u)", u32::from_ne_bytes(a)))
    };
    let f64_lit = |b: &[u8]| -> Result<String, String> {
        let a: [u8; 8] = b.try_into().map_err(|_| wrong())?;
        Ok(format!("as_double(0x{:016x}UL)", u64::from_ne_bytes(a)))
    };
    match info.storage {
        "float" => f32_lit(bytes),
        "double" => f64_lit(bytes),
        "float2" if bytes.len() == 8 => Ok(format!(
            "(float2)({}, {})",
            f32_lit(&bytes[..4])?,
            f32_lit(&bytes[4..])?
        )),
        "double2" if bytes.len() == 16 => Ok(format!(
            "(double2)({}, {})",
            f64_lit(&bytes[..8])?,
            f64_lit(&bytes[8..])?
        )),
        "half" => {
            let a: [u8; 2] = bytes.try_into().map_err(|_| wrong())?;
            Ok(format!("as_half((ushort)0x{:04x})", u16::from_ne_bytes(a)))
        }
        "float2" | "double2" => Err(wrong()),
        storage => {
            if !matches!(bytes.len(), 1 | 2 | 4 | 8) {
                return Err(wrong());
            }
            let mut raw = [0u8; 8];
            raw[..bytes.len()].copy_from_slice(bytes);
            let unsigned = u64::from_ne_bytes(raw);
            if info.is_signed() {
                let shift = 64 - 8 * bytes.len() as u32;
                let v = ((unsigned << shift) as i64) >> shift;
                if v == i64::MIN {
                    Ok(format!("({storage})(-9223372036854775807L-1L)"))
                } else {
                    Ok(format!("({storage})({v}L)"))
                }
            } else {
                Ok(format!("({storage}){unsigned}UL"))
            }
        }
    }
}

/// Generate the kernel for a fused group. `instr.operands` are the group
/// inputs; `instr.out` and `instr.out_shape` describe the last step.
/// `max_args` is the device's kernel argument limit; a group over it returns
/// an error that starts with `argument limit`.
pub fn generate(
    instr: &Instr,
    fused: &Fused,
    caps: &Caps,
    max_args: usize,
) -> Result<KernelSpec, String> {
    let Fused { maps, steps } = fused;
    let last = steps.last().ok_or("a fused group has no steps")?;
    if last.dtype != instr.out {
        return Err(format!(
            "fused: the last step is {}, the output {}",
            last.dtype, instr.out
        ));
    }
    if maps.len() != instr.operands.len() {
        return Err(format!(
            "fused: {} input maps for {} operands",
            maps.len(),
            instr.operands.len()
        ));
    }
    let rank = instr.out_shape.len();

    // Strides of each broadcast input, per output axis.
    let mut bstrides: Vec<Option<Vec<i64>>> = Vec::new();
    for (k, (operand, map)) in instr.operands.iter().zip(maps).enumerate() {
        bstrides.push(match map {
            _ if operand.constant.is_some() => None,
            InMap::Direct => {
                if operand.shape != instr.out_shape {
                    return Err(format!(
                        "fused: input {} has shape {:?}, the output {:?}",
                        k, operand.shape, instr.out_shape
                    ));
                }
                None
            }
            InMap::Broadcast(axes) => {
                if axes.len() != operand.shape.len() {
                    return Err(format!(
                        "fused: {} axes for input {} of rank {}",
                        axes.len(),
                        k,
                        operand.shape.len()
                    ));
                }
                let own = strides(&operand.shape);
                let mut s = vec![0i64; rank];
                for (j, &a) in axes.iter().enumerate() {
                    if a >= rank {
                        return Err(format!("fused: broadcast axis {} is out of range", a));
                    }
                    let dim = operand.shape[j];
                    if dim != 1 && dim != instr.out_shape[a] {
                        return Err(format!(
                            "fused: dimension {} of input {} does not fit output axis {}",
                            dim, k, a
                        ));
                    }
                    if dim != 1 {
                        s[a] = own[j];
                    }
                }
                Some(s)
            }
        });
    }
    let n_buffers = instr
        .operands
        .iter()
        .filter(|o| o.constant.is_none())
        .count();
    let n_bcast = bstrides.iter().filter(|s| s.is_some()).count();
    if n_bcast > 0 {
        check_rank("fused", rank)?;
    }
    let needed = arg_count(n_buffers, rank, n_bcast);
    if needed > max_args {
        return Err(format!(
            "argument limit: the fused kernel needs {} arguments, the device allows {}",
            needed, max_args
        ));
    }

    // One function per step.
    let mut code = ScalarCode::default();
    let mut dtypes: Vec<DType> = instr.operands.iter().map(|o| o.dtype).collect();
    let mut types: Vec<DType> = Vec::with_capacity(steps.len());
    let mut lines = String::new();
    for (j, step) in steps.iter().enumerate() {
        let mut in_types = Vec::with_capacity(step.srcs.len());
        let mut args = Vec::with_capacity(step.srcs.len());
        for src in &step.srcs {
            match *src {
                Src::In(k) => {
                    let operand = instr
                        .operands
                        .get(k)
                        .ok_or_else(|| format!("fused: step {} reads input {}", j, k))?;
                    in_types.push(operand.dtype);
                    args.push(match &operand.constant {
                        Some(bytes) => literal(&type_info(operand.dtype)?, bytes)?,
                        None => format!("v{}", k),
                    });
                }
                Src::Step(m) if m < j => {
                    in_types.push(types[m]);
                    args.push(format!("t{}", m));
                }
                Src::Step(m) => return Err(format!("fused: step {} reads step {}", j, m)),
            }
        }
        let name = format!("s{}", j);
        let (fn_code, compute) = scalar_step(&step.op, &in_types, step.dtype, &step.attrs, &name)?;
        code.merge(fn_code);
        dtypes.push(step.dtype);
        dtypes.push(compute);
        types.push(step.dtype);
        lines.push_str(&format!(
            "  {} t{} = {}({});\n",
            type_info(step.dtype)?.storage,
            j,
            name,
            args.join(", ")
        ));
    }

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

    // Parameters and arguments, in the order the kernel declares them.
    let mut params = Vec::new();
    let mut args = Vec::new();
    for (k, operand) in instr.operands.iter().enumerate() {
        if operand.constant.is_none() {
            params.push(format!(
                "__global const {}* in{}",
                type_info(operand.dtype)?.storage,
                k
            ));
            args.push(KernelArg::Operand(k));
        }
    }
    params.push(format!("__global {}* out", type_info(instr.out)?.storage));
    args.push(KernelArg::Output);

    let mut loads = String::new();
    let head = if n_bcast > 0 {
        for (d, v) in dims(&instr.out_shape).into_iter().enumerate() {
            params.push(format!("const long o{}", d));
            args.push(KernelArg::Scalar(v.to_ne_bytes().to_vec()));
        }
        coords(rank)
    } else {
        "  size_t i = get_global_id(0);\n".to_string()
    };
    for (k, operand) in instr.operands.iter().enumerate() {
        if operand.constant.is_some() {
            continue;
        }
        let storage = type_info(operand.dtype)?.storage;
        let index = match &bstrides[k] {
            None => "i".to_string(),
            Some(s) => {
                for (d, v) in s.iter().enumerate() {
                    params.push(format!("const long b{}s{}", k, d));
                    args.push(KernelArg::Scalar(v.to_ne_bytes().to_vec()));
                }
                let terms: Vec<String> = (0..rank).map(|d| format!("c{d} * b{k}s{d}")).collect();
                if terms.is_empty() {
                    "0".to_string()
                } else {
                    terms.join(" + ")
                }
            }
        };
        loads.push_str(&format!("  {storage} v{k} = in{k}[{index}];\n"));
    }

    let body = format!("{head}{loads}{lines}  out[i] = t{};\n", steps.len() - 1);
    let helpers = code.render();
    let entry = format!(
        "fz_{}_{:016x}",
        steps.len(),
        fnv(&format!("{helpers}{body}"))
    );
    let source = format!(
        "{pragmas}{helpers}__kernel void {entry}({params}) {{\n{body}}}\n",
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
    use super::super::types::{encode_const, ConstVal, Kind};
    use super::super::Operand;
    use super::*;

    const F32: DType = DType::new(Kind::F, 32);
    const S32: DType = DType::new(Kind::S, 32);
    const U8: DType = DType::new(Kind::U, 8);

    fn buf(dtype: DType, shape: &[usize]) -> Operand {
        Operand {
            dtype,
            shape: shape.to_vec(),
            constant: None,
        }
    }

    fn konst(dtype: DType, value: ConstVal) -> Operand {
        Operand {
            dtype,
            shape: vec![],
            constant: Some(encode_const(dtype, value).unwrap()),
        }
    }

    fn step(op: &str, srcs: &[Src], dtype: DType) -> Step {
        Step {
            op: op.to_string(),
            srcs: srcs.to_vec(),
            dtype,
            attrs: vec![],
        }
    }

    fn group(operands: Vec<Operand>, maps: Vec<InMap>, steps: Vec<Step>) -> (Instr, Fused) {
        let out = steps.last().unwrap().dtype;
        (
            Instr {
                op: "fused".to_string(),
                operands,
                out,
                out_shape: vec![6],
                attrs: vec![],
            },
            Fused { maps, steps },
        )
    }

    /// `x`, then `n` alternating `exp`/`negate` steps.
    fn chain(n: usize) -> (Instr, Fused) {
        let mut steps = vec![step("exp", &[Src::In(0)], F32)];
        for j in 1..n {
            let op = if j % 2 == 0 { "exp" } else { "negate" };
            steps.push(step(op, &[Src::Step(j - 1)], F32));
        }
        group(vec![buf(F32, &[6])], vec![InMap::Direct], steps)
    }

    fn spec(i: &Instr, f: &Fused) -> KernelSpec {
        generate(i, f, &Caps::all(), 64).unwrap()
    }

    #[test]
    fn a_twenty_step_chain_has_one_entry_and_one_store() {
        let (i, f) = chain(20);
        let s = spec(&i, &f);
        assert_eq!(s.source.matches("__kernel").count(), 1);
        assert_eq!(s.source.matches("out[i] =").count(), 1);
        assert_eq!(s.source.matches("= s").count(), 20);
        assert!(s.source.contains("out[i] = t19;"));
        assert_eq!(s.global, vec![6]);
    }

    #[test]
    fn constants_are_literals_and_take_no_argument() {
        let (i, f) = group(
            vec![buf(F32, &[6]), konst(F32, ConstVal::Float(1.0))],
            vec![InMap::Direct, InMap::Direct],
            vec![step("add", &[Src::In(0), Src::In(1)], F32)],
        );
        let s = spec(&i, &f);
        assert!(s.source.contains("as_float(0x3f800000u)"), "{}", s.source);
        assert!(!s.source.contains("* in1"), "{}", s.source);
        assert_eq!(s.args, vec![KernelArg::Operand(0), KernelArg::Output]);
    }

    #[test]
    fn integer_literals_keep_sign_and_width() {
        let m = type_info(S32).unwrap();
        assert_eq!(literal(&m, &(-5i32).to_ne_bytes()).unwrap(), "(int)(-5L)");
        let u = type_info(DType::new(Kind::U, 64)).unwrap();
        assert_eq!(
            literal(&u, &u64::MAX.to_ne_bytes()).unwrap(),
            "(ulong)18446744073709551615UL"
        );
        let l = type_info(DType::new(Kind::S, 64)).unwrap();
        assert!(literal(&l, &i64::MIN.to_ne_bytes())
            .unwrap()
            .contains("9223372036854775807L-1L"));
    }

    #[test]
    fn a_broadcast_input_reads_through_its_index_mapping() {
        // Output {2, 3}; the input {3} fills axis 1.
        let (mut i, f) = group(
            vec![buf(F32, &[2, 3]), buf(F32, &[3])],
            vec![InMap::Direct, InMap::Broadcast(vec![1])],
            vec![step("add", &[Src::In(0), Src::In(1)], F32)],
        );
        i.out_shape = vec![2, 3];
        let s = spec(&i, &f);
        assert!(s.source.contains("v0 = in0[i];"), "{}", s.source);
        assert!(
            s.source.contains("v1 = in1[c0 * b1s0 + c1 * b1s1];"),
            "{}",
            s.source
        );
        let long = |v: i64| KernelArg::Scalar(v.to_ne_bytes().to_vec());
        // Axis 0 repeats (stride 0); axis 1 walks the input.
        assert_eq!(
            s.args,
            vec![
                KernelArg::Operand(0),
                KernelArg::Operand(1),
                KernelArg::Output,
                long(2),
                long(3),
                long(0),
                long(1)
            ]
        );
    }

    #[test]
    fn mixed_types_cast_inside_the_kernel() {
        let (i, f) = group(
            vec![buf(F32, &[6]), buf(S32, &[6])],
            vec![InMap::Direct, InMap::Direct],
            vec![step("add", &[Src::In(0), Src::In(1)], F32)],
        );
        let s = spec(&i, &f);
        assert!(s.source.contains("((float)(in1))"), "{}", s.source);
    }

    #[test]
    fn a_comparison_feeds_a_select() {
        let (i, f) = group(
            vec![buf(F32, &[6]), buf(F32, &[6])],
            vec![InMap::Direct, InMap::Direct],
            vec![
                step("greater", &[Src::In(0), Src::In(1)], U8),
                step("select", &[Src::Step(0), Src::In(0), Src::In(1)], F32),
            ],
        );
        let s = spec(&i, &f);
        assert!(s.source.contains("uchar t0 = s0(v0, v1);"), "{}", s.source);
        assert!(
            s.source.contains("float t1 = s1(t0, v0, v1);"),
            "{}",
            s.source
        );
    }

    #[test]
    fn a_group_over_the_argument_limit_returns_the_limit_error() {
        // Five buffers and the output need 6 arguments.
        let ops = vec![buf(F32, &[6]); 5];
        let mut steps = vec![step("add", &[Src::In(0), Src::In(1)], F32)];
        for k in 2..5 {
            steps.push(step("add", &[Src::Step(k - 2), Src::In(k)], F32));
        }
        let (i, f) = group(ops, vec![InMap::Direct; 5], steps);
        assert!(generate(&i, &f, &Caps::all(), 6).is_ok());
        let err = generate(&i, &f, &Caps::all(), 5).unwrap_err();
        assert!(err.starts_with("argument limit"), "{}", err);
        assert!(
            err.contains("needs 6") && err.contains("allows 5"),
            "{}",
            err
        );
    }

    #[test]
    fn two_equal_groups_give_the_same_source() {
        let (i, f) = chain(8);
        let (j, g) = chain(8);
        let (a, b) = (spec(&i, &f), spec(&j, &g));
        assert_eq!(a.source, b.source);
        assert_eq!(a.entry, b.entry);
        let (k, h) = chain(9);
        assert_ne!(spec(&k, &h).entry, a.entry);
    }

    #[test]
    fn the_output_is_a_fresh_argument_and_each_input_appears_once() {
        // Both steps read input 0; no argument names it twice or names the output twice.
        let (i, f) = group(
            vec![buf(F32, &[6]), buf(F32, &[6])],
            vec![InMap::Direct, InMap::Direct],
            vec![
                step("add", &[Src::In(0), Src::In(1)], F32),
                step("multiply", &[Src::Step(0), Src::In(0)], F32),
            ],
        );
        let s = spec(&i, &f);
        assert_eq!(
            s.args,
            vec![
                KernelArg::Operand(0),
                KernelArg::Operand(1),
                KernelArg::Output
            ]
        );
        assert_eq!(s.source.matches("in0[").count(), 1);
        assert_eq!(s.source.matches("out[").count(), 1);
    }

    #[test]
    fn a_bad_step_reference_is_an_error() {
        let (i, f) = group(
            vec![buf(F32, &[6])],
            vec![InMap::Direct],
            vec![step("exp", &[Src::Step(0)], F32)],
        );
        assert!(generate(&i, &f, &Caps::all(), 64).is_err());
    }
}
