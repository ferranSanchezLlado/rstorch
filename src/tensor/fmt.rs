//! [`Display`](std::fmt::Display) and [`Debug`](std::fmt::Debug) for
//! [`Tensor`]: a shape/dtype/device summary plus truncated values.
//!
//! Printing a tensor is a **host boundary** — the values are read with
//! `to_vec`, which synchronizes the backend and copies the whole logical
//! tensor to the host (strided views are walked in row-major logical order,
//! so what you see is what the tensor *means*, not how it is laid out). A
//! read failure is never a panic: it degrades to an `<unavailable: …>`
//! placeholder, because a formatting impl that can blow up is a debugging
//! trap.
//!
//! # The two spellings
//!
//! - `Display` (`{}`) is the human view: a one-line header followed by the
//!   values rendered as a nested, PyTorch-ish block, one row per line.
//!   Floats honor the format string's precision (`{:.2}`) and default to
//!   four fractional digits, switching to scientific notation when any value
//!   is too large or too small to read at that precision.
//! - `Debug` (`{:?}`, `{:#?}`) is the developer view: a struct with the
//!   metadata fields — including the recorded layout's contiguity and whether
//!   the tensor carries an autograd graph — and a *flat* value list. Floats
//!   print in their shortest round-trip form, so a Debug print never hides a
//!   value.
//!
//! The `contiguous` field is the only place contiguity reaches outside the
//! crate, and it is a debugging aid rather than a contract: it reports the
//! logical layout as recorded, not a promise about physical storage. There is
//! deliberately no public predicate to branch on — see
//! [`Tensor::contiguous`] for the way to *ask* for a dense operand.
//!
//! # Truncation
//!
//! Both are bounded in size, so printing a model-sized tensor cannot flood a
//! terminal. `Display` elides an axis holding more than [`MAX_AXIS_ITEMS`]
//! entries — or more than `2 * EDGE_ITEMS`, once the tensor as a whole
//! exceeds [`SUMMARY_THRESHOLD`] elements — down to its first and last
//! [`EDGE_ITEMS`] entries around a `...` marker. `Debug` applies the same
//! rule once, to the flat value list.

use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::Tensor;
use std::fmt;

/// Longest axis `Display` renders in full; longer axes are elided.
const MAX_AXIS_ITEMS: usize = 8;

/// Entries kept at each end of an elided axis or value list.
const EDGE_ITEMS: usize = 3;

/// Above this element count every axis longer than `2 * EDGE_ITEMS` is
/// elided, not just those longer than [`MAX_AXIS_ITEMS`]. Without it a
/// rank-4 tensor of 8-long axes would print 512 rows.
const SUMMARY_THRESHOLD: usize = 1000;

/// Fractional digits `Display` uses for floats when the format string does
/// not ask for a precision.
const DEFAULT_PRECISION: usize = 4;

/// Longest flat value list `Debug` renders in full.
const MAX_FLAT_ITEMS: usize = 8;

/// Magnitude at which fixed-point float rendering gets unreadable and
/// `Display` switches the whole tensor to scientific notation.
const LARGE_MAGNITUDE: f64 = 1e6;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Tensor(shape={}, dtype={}, device={})",
            self.shape(),
            self.dtype(),
            self.device()
        )?;
        match Values::read(self) {
            Ok(values) => {
                let precision = f.precision().unwrap_or(DEFAULT_PRECISION);
                let renderer = Renderer {
                    style: FloatStyle::for_values(&values, precision),
                    values: &values,
                    // A big tensor summarizes more aggressively so the block
                    // stays a screenful whatever its rank.
                    cap: if self.num_elements() > SUMMARY_THRESHOLD {
                        2 * EDGE_ITEMS
                    } else {
                        MAX_AXIS_ITEMS
                    },
                };
                renderer.block(f, self.dims(), 0, 0)
            }
            Err(err) => write!(f, "<values unavailable: {err}>"),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values = Values::read(self);
        let mut out = f.debug_struct("Tensor");
        out.field("shape", &format_args!("{}", self.shape()))
            .field("dtype", &format_args!("{}", self.dtype()))
            .field("device", &format_args!("{}", self.device()))
            .field("contiguous", &self.is_contiguous())
            .field("traced", &self.node().is_some());
        match &values {
            Ok(values) => out.field(
                "values",
                &format_args!(
                    "{}",
                    Flat {
                        values,
                        len: self.num_elements()
                    }
                ),
            ),
            Err(err) => out.field("values", &format_args!("<unavailable: {err}>")),
        };
        out.finish()
    }
}

/// The tensor's elements on the host, in row-major logical order, narrowed
/// to the families that print differently.
///
/// `F16`/`BF16` widen into `F32` — both conversions are exact, so nothing is
/// invented and there are four cases to render instead of six.
enum Values {
    /// `f16`, `bf16` and `f32` tensors.
    F32(Vec<f32>),
    /// `f64` tensors.
    F64(Vec<f64>),
    /// `i64` tensors.
    I64(Vec<i64>),
    /// `bool` tensors.
    Bool(Vec<bool>),
}

impl Values {
    /// Copy `t`'s elements to the host. The host boundary of both impls.
    fn read(t: &Tensor) -> Result<Values> {
        Ok(match t.dtype() {
            DType::F16 => Values::F32(
                t.to_vec::<half::f16>()?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
            ),
            DType::BF16 => Values::F32(
                t.to_vec::<half::bf16>()?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
            ),
            DType::F32 => Values::F32(t.to_vec::<f32>()?),
            DType::F64 => Values::F64(t.to_vec::<f64>()?),
            DType::I64 => Values::I64(t.to_vec::<i64>()?),
            DType::Bool => Values::Bool(t.to_vec::<bool>()?),
        })
    }

    /// Render the element at flat index `i`.
    fn cell(&self, i: usize, style: FloatStyle) -> String {
        match self {
            Values::F32(v) => style.render(v[i]),
            Values::F64(v) => style.render(v[i]),
            Values::I64(v) => v[i].to_string(),
            Values::Bool(v) => v[i].to_string(),
        }
    }
}

/// How float elements are spelled.
#[derive(Clone, Copy)]
enum FloatStyle {
    /// Shortest round-trip form (`Debug`): never hides a value.
    Shortest,
    /// Fixed point with `n` fractional digits.
    Fixed(usize),
    /// Scientific notation with `n` fractional digits.
    Exp(usize),
}

impl FloatStyle {
    /// The style `Display` uses for `values` at `precision`: fixed point,
    /// unless some value would be unreadable there — too large to scan, or
    /// small enough that fixed point would round it to a bare `0.0000` and
    /// lie about it.
    fn for_values(values: &Values, precision: usize) -> FloatStyle {
        // The smallest magnitude `precision` fractional digits still show.
        // A zero-precision request has no such floor (every value below 1
        // would otherwise flip the tensor to scientific notation).
        let smallest = match i32::try_from(precision) {
            Ok(0) | Err(_) => 0.0,
            Ok(p) => 10f64.powi(-p),
        };
        let unreadable = |v: f64| {
            let a = v.abs();
            v.is_finite() && v != 0.0 && (a >= LARGE_MAGNITUDE || a < smallest)
        };
        // One outlier switches the whole tensor: a column of mixed
        // notations is harder to read than a column of uniform ones.
        let exp = match values {
            Values::F32(v) => v.iter().any(|&x| unreadable(f64::from(x))),
            Values::F64(v) => v.iter().copied().any(unreadable),
            Values::I64(_) | Values::Bool(_) => false,
        };
        if exp {
            FloatStyle::Exp(precision)
        } else {
            FloatStyle::Fixed(precision)
        }
    }

    /// Spell one float.
    fn render<T>(self, v: T) -> String
    where
        T: fmt::Debug + fmt::Display + fmt::LowerExp,
    {
        match self {
            // `Debug` rather than `Display`: `1.0f32` must not print as `1`
            // in a float tensor.
            FloatStyle::Shortest => format!("{v:?}"),
            FloatStyle::Fixed(p) => format!("{v:.p$}"),
            FloatStyle::Exp(p) => format!("{v:.p$e}"),
        }
    }
}

/// Which entries of an `n`-long axis to render: `Some(index)` for a value,
/// `None` for the `...` elision marker. Axes of at most `cap` entries are
/// rendered whole; longer ones keep `edge` entries at each end.
fn entries(n: usize, cap: usize, edge: usize) -> Vec<Option<usize>> {
    if n <= cap {
        return (0..n).map(Some).collect();
    }
    let mut out: Vec<Option<usize>> = (0..edge).map(Some).collect();
    out.push(None);
    out.extend((n - edge..n).map(Some));
    out
}

/// The nested-block value renderer behind `Display`.
struct Renderer<'a> {
    values: &'a Values,
    style: FloatStyle,
    /// Longest axis rendered in full (see [`SUMMARY_THRESHOLD`]).
    cap: usize,
}

impl Renderer<'_> {
    /// Render the sub-tensor of shape `dims` whose first element sits at
    /// flat index `base`, with its rows indented by `indent` columns.
    ///
    /// A rank-0 block is a bare value; otherwise the innermost axis becomes
    /// a comma-separated line and every outer axis stacks its children one
    /// per line, so a rank-`n` block reads like nested Rust array literals.
    fn block(
        &self,
        f: &mut fmt::Formatter<'_>,
        dims: &[usize],
        base: usize,
        indent: usize,
    ) -> fmt::Result {
        let Some((&n, rest)) = dims.split_first() else {
            return f.write_str(&self.values.cell(base, self.style));
        };
        // Row-major: one step along this axis skips a whole sub-tensor.
        let stride: usize = rest.iter().product();
        let innermost = rest.is_empty();
        f.write_str("[")?;
        for (position, entry) in entries(n, self.cap, EDGE_ITEMS).into_iter().enumerate() {
            if position > 0 {
                if innermost {
                    f.write_str(", ")?;
                } else {
                    write!(f, ",\n{:width$}", "", width = indent + 1)?;
                }
            }
            match entry {
                Some(i) => self.block(f, rest, base + i * stride, indent + 1)?,
                None => f.write_str("...")?,
            }
        }
        f.write_str("]")
    }
}

/// The flat, elided value list behind `Debug`.
struct Flat<'a> {
    values: &'a Values,
    len: usize,
}

impl fmt::Display for Flat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (position, entry) in entries(self.len, MAX_FLAT_ITEMS, EDGE_ITEMS)
            .into_iter()
            .enumerate()
        {
            if position > 0 {
                f.write_str(", ")?;
            }
            match entry {
                Some(i) => f.write_str(&self.values.cell(i, FloatStyle::Shortest))?,
                None => f.write_str("...")?,
            }
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::shape::Shape;

    const CPU: Device = Device::Cpu;

    /// A contiguous f32 tensor on CPU.
    fn t_f32(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    // ------------------------------------------------------------------
    // Display: small tensors, every rank
    // ------------------------------------------------------------------

    #[test]
    fn display_scalar_is_a_bare_value() {
        let t = Tensor::full((), 7.5, DType::F32, &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[], dtype=f32, device=cpu)\n7.5000"
        );
    }

    #[test]
    fn display_vector_is_one_line() {
        let t = t_f32(&[1.0, -2.25, 0.0], [3]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[3], dtype=f32, device=cpu)\n[1.0000, -2.2500, 0.0000]"
        );
    }

    #[test]
    fn display_matrix_is_one_row_per_line() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[2, 3], dtype=f32, device=cpu)\n\
             [[1.0000, 2.0000, 3.0000],\n \
             [4.0000, 5.0000, 6.0000]]"
        );
    }

    #[test]
    fn display_rank3_nests_and_indents() {
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let t = t_f32(&data, [2, 2, 2]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[2, 2, 2], dtype=f32, device=cpu)\n\
             [[[1.0000, 2.0000],\n  \
             [3.0000, 4.0000]],\n \
             [[5.0000, 6.0000],\n  \
             [7.0000, 8.0000]]]"
        );
    }

    #[test]
    fn display_empty_tensors() {
        let t = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[0, 3], dtype=f32, device=cpu)\n[]"
        );
        // An empty *inner* axis still shows the outer structure.
        let t = Tensor::zeros([3, 0], DType::F32, &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[3, 0], dtype=f32, device=cpu)\n[[],\n [],\n []]"
        );
    }

    #[test]
    fn display_reads_strided_views_in_logical_order() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed =
            Tensor::from_parts(t.storage().clone(), t.layout().transpose(0, 1).unwrap());
        assert!(!transposed.is_contiguous());
        assert_eq!(
            transposed.to_string(),
            "Tensor(shape=[3, 2], dtype=f32, device=cpu)\n\
             [[1.0000, 4.0000],\n \
             [2.0000, 5.0000],\n \
             [3.0000, 6.0000]]"
        );
    }

    // ------------------------------------------------------------------
    // Display: truncation
    // ------------------------------------------------------------------

    #[test]
    fn display_elides_a_long_axis() {
        // Ten entries: the first and last three around a marker.
        let t = Tensor::arange(0.0, 10.0, 1.0, DType::I64, &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[10], dtype=i64, device=cpu)\n[0, 1, 2, ..., 7, 8, 9]"
        );
        // Exactly at the cap, nothing is elided.
        let t = Tensor::arange(0.0, 8.0, 1.0, DType::I64, &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[8], dtype=i64, device=cpu)\n[0, 1, 2, 3, 4, 5, 6, 7]"
        );
    }

    #[test]
    fn display_elides_rows_and_columns_of_a_big_matrix() {
        let data: Vec<i64> = (0..90).collect();
        let t = Tensor::from_vec(data, [9, 10], &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[9, 10], dtype=i64, device=cpu)\n\
             [[0, 1, 2, ..., 7, 8, 9],\n \
             [10, 11, 12, ..., 17, 18, 19],\n \
             [20, 21, 22, ..., 27, 28, 29],\n \
             ...,\n \
             [60, 61, 62, ..., 67, 68, 69],\n \
             [70, 71, 72, ..., 77, 78, 79],\n \
             [80, 81, 82, ..., 87, 88, 89]]"
        );
    }

    #[test]
    fn display_summarizes_harder_past_the_element_threshold() {
        // 1400 elements: the 7-long axis is under `MAX_AXIS_ITEMS` but still
        // gets elided, because the tensor as a whole is over
        // `SUMMARY_THRESHOLD`.
        let data: Vec<i64> = (0..1400).collect();
        let t = Tensor::from_vec(data, [7, 200], &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[7, 200], dtype=i64, device=cpu)\n\
             [[0, 1, 2, ..., 197, 198, 199],\n \
             [200, 201, 202, ..., 397, 398, 399],\n \
             [400, 401, 402, ..., 597, 598, 599],\n \
             ...,\n \
             [800, 801, 802, ..., 997, 998, 999],\n \
             [1000, 1001, 1002, ..., 1197, 1198, 1199],\n \
             [1200, 1201, 1202, ..., 1397, 1398, 1399]]"
        );
        // Below the threshold the same 7-long axis prints whole.
        let data: Vec<i64> = (0..7).collect();
        let t = Tensor::from_vec(data, [7], &CPU).unwrap();
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[7], dtype=i64, device=cpu)\n[0, 1, 2, 3, 4, 5, 6]"
        );
    }

    // ------------------------------------------------------------------
    // Display: dtypes and float spelling
    // ------------------------------------------------------------------

    #[test]
    fn display_covers_every_dtype() {
        let f16 = Tensor::full([2], 1.5, DType::F16, &CPU).unwrap();
        assert_eq!(
            f16.to_string(),
            "Tensor(shape=[2], dtype=f16, device=cpu)\n[1.5000, 1.5000]"
        );

        let bf16 = Tensor::full([2], -0.5, DType::BF16, &CPU).unwrap();
        assert_eq!(
            bf16.to_string(),
            "Tensor(shape=[2], dtype=bf16, device=cpu)\n[-0.5000, -0.5000]"
        );

        let f32 = t_f32(&[0.25], [1]);
        assert_eq!(
            f32.to_string(),
            "Tensor(shape=[1], dtype=f32, device=cpu)\n[0.2500]"
        );

        let f64 = Tensor::from_vec(vec![0.125f64, 2.0], [2], &CPU).unwrap();
        assert_eq!(
            f64.to_string(),
            "Tensor(shape=[2], dtype=f64, device=cpu)\n[0.1250, 2.0000]"
        );

        // Integers and bools are never given fractional digits.
        let i64 = Tensor::from_vec(vec![-7i64, 0, 42], [3], &CPU).unwrap();
        assert_eq!(
            i64.to_string(),
            "Tensor(shape=[3], dtype=i64, device=cpu)\n[-7, 0, 42]"
        );

        let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert_eq!(
            b.to_string(),
            "Tensor(shape=[2], dtype=bool, device=cpu)\n[true, false]"
        );
    }

    #[test]
    fn display_honors_the_format_precision() {
        let t = t_f32(&[1.0, 2.5], [2]);
        assert_eq!(
            format!("{t:.1}"),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[1.0, 2.5]"
        );
        assert_eq!(
            format!("{t:.0}"),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[1, 2]"
        );
        // Precision is a float notion; integers ignore it.
        let i = Tensor::from_vec(vec![3i64], [1], &CPU).unwrap();
        assert_eq!(
            format!("{i:.3}"),
            "Tensor(shape=[1], dtype=i64, device=cpu)\n[3]"
        );
    }

    #[test]
    fn display_switches_to_scientific_for_unreadable_magnitudes() {
        // A value fixed point would round away to `0.0000` flips the whole
        // tensor, so the columns stay in one notation.
        let t = t_f32(&[1e-8, 1.0], [2]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[1.0000e-8, 1.0000e0]"
        );
        // ...and so does one that is too long to scan.
        let t = t_f32(&[2.5e7, 0.0], [2]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[2.5000e7, 0.0000e0]"
        );
        // Zero, infinities and NaN are not "unreadable": they do not switch
        // an otherwise ordinary tensor.
        let t = t_f32(&[0.0, f32::INFINITY, f32::NAN, 1.5], [4]);
        assert_eq!(
            t.to_string(),
            "Tensor(shape=[4], dtype=f32, device=cpu)\n[0.0000, inf, NaN, 1.5000]"
        );
        // A smaller precision has a higher floor for "too small to see".
        let t = t_f32(&[0.001, 1.0], [2]);
        assert_eq!(
            format!("{t:.4}"),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[0.0010, 1.0000]"
        );
        assert_eq!(
            format!("{t:.2}"),
            "Tensor(shape=[2], dtype=f32, device=cpu)\n[1.00e-3, 1.00e0]"
        );
    }

    // ------------------------------------------------------------------
    // Debug
    // ------------------------------------------------------------------

    #[test]
    fn debug_is_a_one_line_struct_with_flat_values() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert_eq!(
            format!("{t:?}"),
            "Tensor { shape: [2, 3], dtype: f32, device: cpu, contiguous: true, \
             traced: false, values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }"
        );
    }

    #[test]
    fn debug_alternate_is_multi_line() {
        let t = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        assert_eq!(
            format!("{t:#?}"),
            "Tensor {\n    \
             shape: [2],\n    \
             dtype: i64,\n    \
             device: cpu,\n    \
             contiguous: true,\n    \
             traced: false,\n    \
             values: [1, 2],\n}"
        );
    }

    #[test]
    fn debug_elides_long_value_lists() {
        let t = Tensor::arange(0.0, 100.0, 1.0, DType::I64, &CPU).unwrap();
        assert_eq!(
            format!("{t:?}"),
            "Tensor { shape: [100], dtype: i64, device: cpu, contiguous: true, \
             traced: false, values: [0, 1, 2, ..., 97, 98, 99] }"
        );
    }

    #[test]
    fn debug_reports_a_non_contiguous_layout() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let transposed =
            Tensor::from_parts(t.storage().clone(), t.layout().transpose(0, 1).unwrap());
        assert_eq!(
            format!("{transposed:?}"),
            "Tensor { shape: [2, 2], dtype: f32, device: cpu, contiguous: false, \
             traced: false, values: [1.0, 3.0, 2.0, 4.0] }"
        );
    }

    #[test]
    fn debug_contiguity_field_is_the_only_way_contiguity_leaves_the_crate() {
        // `Tensor::is_contiguous` is `pub(crate)`, so this field is the whole
        // affordance a downstream consumer has. It is reached through the
        // public view ops alone, exactly as such a consumer would reach it.
        let dense = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert!(format!("{dense:?}").contains("contiguous: true"));
        let strided = dense.transpose(0, 1).unwrap();
        assert!(format!("{strided:?}").contains("contiguous: false"));
        assert!(
            format!("{:?}", strided.contiguous().unwrap()).contains("contiguous: true"),
            "asking for a dense operand must show up in the Debug field"
        );
    }

    #[test]
    fn debug_never_hides_a_value_behind_a_rounding() {
        // Display rounds to four digits; Debug shows the shortest form that
        // round-trips, so the two disagree on purpose.
        let t = t_f32(&[0.1, 1e-8], [2]);
        assert_eq!(
            format!("{t:?}"),
            "Tensor { shape: [2], dtype: f32, device: cpu, contiguous: true, \
             traced: false, values: [0.1, 1e-8] }"
        );
    }

    #[test]
    fn debug_covers_every_dtype() {
        let cases = [
            (
                Tensor::full([1], 1.5, DType::F16, &CPU).unwrap(),
                "dtype: f16",
                "values: [1.5]",
            ),
            (
                Tensor::full([1], 1.5, DType::BF16, &CPU).unwrap(),
                "dtype: bf16",
                "values: [1.5]",
            ),
            (t_f32(&[1.5], [1]), "dtype: f32", "values: [1.5]"),
            (
                Tensor::from_vec(vec![1.5f64], [1], &CPU).unwrap(),
                "dtype: f64",
                "values: [1.5]",
            ),
            (
                Tensor::from_vec(vec![-3i64], [1], &CPU).unwrap(),
                "dtype: i64",
                "values: [-3]",
            ),
            (
                Tensor::from_vec(vec![true], [1], &CPU).unwrap(),
                "dtype: bool",
                "values: [true]",
            ),
        ];
        for (t, dtype, values) in cases {
            let rendered = format!("{t:?}");
            assert!(rendered.contains(dtype), "{rendered}");
            assert!(rendered.contains(values), "{rendered}");
        }
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    #[test]
    fn entries_elides_only_past_the_cap() {
        assert_eq!(entries(3, 8, 3), vec![Some(0), Some(1), Some(2)]);
        assert_eq!(entries(0, 8, 3), vec![]);
        assert_eq!(
            entries(9, 8, 3),
            vec![Some(0), Some(1), Some(2), None, Some(6), Some(7), Some(8)]
        );
    }

    #[test]
    fn formatting_a_broadcast_view_repeats_its_elements() {
        // Stride-0 axes are logical elements like any other.
        let row = t_f32(&[1.0, 2.0], [1, 2]);
        let b = Tensor::from_parts(
            row.storage().clone(),
            row.layout().broadcast_to(&Shape::from([3, 2])).unwrap(),
        );
        assert_eq!(
            b.to_string(),
            "Tensor(shape=[3, 2], dtype=f32, device=cpu)\n\
             [[1.0000, 2.0000],\n \
             [1.0000, 2.0000],\n \
             [1.0000, 2.0000]]"
        );
    }
}
