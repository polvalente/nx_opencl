//! N-d index math for kernels. Shapes and strides reach a kernel as `long`
//! arguments, and the flat work-item id is split into coordinates in
//! unrolled source, so the source depends on the rank and not on the sizes.

/// Highest tensor rank a kernel here takes. A higher rank is an error.
pub const MAX_RANK: usize = 8;

/// Error when `rank` is above `MAX_RANK`.
pub fn check_rank(op: &str, rank: usize) -> Result<(), String> {
    if rank > MAX_RANK {
        Err(format!(
            "{}: rank {} is above the limit of {}",
            op, rank, MAX_RANK
        ))
    } else {
        Ok(())
    }
}

/// A padding amount from the wire. Kernels subtract it from `long` coordinates,
/// so it must stay far from the `long` limits; 32 bits is more than any tensor
/// axis needs.
pub fn check_pad(op: &str, v: i64) -> Result<i64, String> {
    i32::try_from(v)
        .map(i64::from)
        .map_err(|_| format!("{}: padding {} does not fit in 32 bits", op, v))
}

/// Row-major element strides of `shape`.
pub fn strides(shape: &[usize]) -> Vec<i64> {
    let mut out = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        // Saturates on a shape with a zero axis and huge ones; such a tensor is
        // empty and no kernel indexes it.
        out[d] = out[d + 1].saturating_mul(shape[d + 1] as i64);
    }
    out
}

pub fn dims(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&d| d as i64).collect()
}

/// Source that sets `base`, the offset of element 0 of line `line`, for a
/// tensor scanned or sorted along an axis of `N` elements with stride `inner`.
/// `line` counts the lines in row-major order of the other axes.
pub const LINE: &str = "  long outer = line / inner;
  long inn = line % inner;
  long base = outer * N * inner + inn;
";

/// Source that splits the flat id `i` into `c0..c{rank-1}`, given the output
/// dimensions `o0..`.
pub fn coords(rank: usize) -> String {
    let mut s = String::from("  size_t i = get_global_id(0);\n");
    if rank > 0 {
        s.push_str("  long r = (long)i;\n");
    }
    for d in (0..rank).rev() {
        s.push_str(&format!("  long c{d} = r % o{d};\n  r /= o{d};\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[5]), vec![1]);
        assert!(strides(&[]).is_empty());
    }

    #[test]
    fn coords_split_the_id_from_the_last_axis_up() {
        let s = coords(2);
        assert!(s.find("long c1 = r % o1;").unwrap() < s.find("long c0 = r % o0;").unwrap());
        assert!(!coords(0).contains("long r"));
    }

    #[test]
    fn the_rank_cap_is_eight() {
        assert!(check_rank("x", 8).is_ok());
        assert!(check_rank("x", 9).unwrap_err().contains("rank 9"));
    }

    #[test]
    fn strides_of_an_empty_but_huge_shape_do_not_wrap() {
        let s = strides(&[0, 1 << 40, 1 << 40]);
        assert_eq!(s[2], 1);
        assert_eq!(s[1], 1 << 40);
        assert_eq!(s[0], i64::MAX);
    }
}
