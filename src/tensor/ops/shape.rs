//! Shape and view operations (T21): `reshape`, `transpose`, `permute`,
//! `squeeze`, `unsqueeze`, `narrow`, `broadcast_to`, `cat`, `stack`.
//!
//! # Views and contiguity (exploration §4.2)
//!
//! `transpose`, `permute`, `squeeze`, `unsqueeze`, `narrow` and
//! `broadcast_to` are **always zero-copy**: they re-describe the same storage
//! through a new [`Layout`] (an `Arc` bump, no element copy). `reshape`
//! follows PyTorch semantics — a view when the source layout permits one
//! (`Layout::reshape_view`), a contiguous copy otherwise. `cat`/`stack` are
//! the only ops here that always allocate, because their result cannot be a
//! view of several disjoint buffers.
//!
//! Axes are `isize` with negative indexing throughout, resolved by
//! [`Shape::resolve_axis`] (or `Shape::resolve_insert_axis` for the
//! axis-inserting `unsqueeze`/`stack`), so `-1` always means "last".
//!
//! # Backwards
//!
//! Every op here is differentiable and records through the frozen
//! `crate::autograd::record` seam. The rules are the geometric inverses:
//!
//! | forward | backward |
//! |---|---|
//! | `reshape(to)` | `reshape(from)` |
//! | `transpose(a, b)` | `transpose(a, b)` (an involution) |
//! | `permute(p)` | `permute(p⁻¹)` |
//! | `squeeze(a)` / `unsqueeze(a)` | `unsqueeze(a)` / `squeeze(a)` |
//! | `narrow(a, s, l)` | zero-pad back to the source size |
//! | `broadcast_to(t)` | `sum_to(source dims)` |
//! | `cat` / `stack` | `narrow` the cotangent, one region per input |
//!
//! No backward closure captures a tensor: they close over dimensions, axis
//! indices, the dtype and the device only, which trivially satisfies the
//! detached-output capture rule (exploration §4.3).

use crate::autograd::record;
use crate::backend::dispatch;
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::CpuStorage;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Re-view `t`'s storage through `layout`, producing an **untraced** tensor.
/// The value-level half of every zero-copy op here; the caller wraps the
/// result with [`record`].
fn view_of(t: &Tensor, layout: Layout) -> Tensor {
    Tensor::from_parts(t.storage().clone(), layout)
}

/// Concatenate `parts` along the pre-resolved `axis` into a fresh contiguous
/// tensor of `out_dims`, **untraced**.
///
/// The caller has already validated that every part shares the dtype, the
/// device and the rank, and agrees on every axis other than `axis`, and that
/// For `cat`, `out_dims` is `parts[0].dims()` with `axis` set to the sum of
/// the parts' sizes along it. For `stack`, `insert_axis` makes that axis an
/// implicit size-1 dimension in every part. `parts` must be non-empty.
///
/// Assembly uses the backend's construction-only `copy_into` primitive, so
/// accelerator tensors never cross a host boundary. CPU's implementation
/// retains dense slice-copy fast paths.
fn concat_values(
    parts: &[&Tensor],
    axis: usize,
    out_dims: &[usize],
    insert_axis: bool,
) -> Result<Tensor> {
    let dtype = parts[0].dtype();
    let backend = dispatch::backend(parts[0].device());
    let layout = Layout::contiguous(out_dims.to_vec())?;
    let total = layout.num_elements();
    if total == 0 {
        // No elements to move: allocate the (empty) buffer and be done. This
        // also covers a zero-sized axis anywhere in `out_dims`.
        return Ok(Tensor::from_parts(backend.full(0, dtype, 0.0)?, layout));
    }

    // Dense CPU inputs can be assembled in one allocation. The generic
    // copy-into path below remains necessary to keep accelerator data on-device.
    if parts[0].device() == Device::Cpu {
        let outer: usize = out_dims[..axis].iter().product();
        let inner: usize = out_dims[axis + 1..].iter().product();
        let sizes: Vec<usize> = if insert_axis {
            vec![1; parts.len()]
        } else {
            parts.iter().map(|t| t.dims()[axis]).collect()
        };
        let blocks = parts
            .iter()
            .map(|t| backend.transfer_out(t.view()))
            .collect::<Result<Vec<CpuStorage>>>()?;

        macro_rules! assemble {
            ($variant:ident) => {{
                let mut slices = Vec::with_capacity(blocks.len());
                for block in &blocks {
                    match block {
                        CpuStorage::$variant(data) => slices.push(data.as_slice()),
                        _ => unreachable!("cat: every block carries the validated dtype"),
                    }
                }
                let mut out = Vec::with_capacity(total);
                for group in 0..outer {
                    for (data, &size) in slices.iter().zip(&sizes) {
                        let chunk = size * inner;
                        let start = group * chunk;
                        out.extend_from_slice(&data[start..start + chunk]);
                    }
                }
                CpuStorage::$variant(Arc::new(out))
            }};
        }

        let host = match dtype {
            DType::F16 => assemble!(F16),
            DType::BF16 => assemble!(BF16),
            DType::F32 => assemble!(F32),
            DType::F64 => assemble!(F64),
            DType::I64 => assemble!(I64),
            DType::Bool => assemble!(Bool),
        };
        return Ok(Tensor::from_parts(backend.transfer_in(host)?, layout));
    }

    let mut storage = backend.full(total, dtype, 0.0)?;
    let mut start = 0;
    for part in parts {
        let size = if insert_axis { 1 } else { part.dims()[axis] };
        let region = layout.narrow(axis, start, size)?;
        let region = if insert_axis {
            region.squeeze(axis)?
        } else {
            region
        };
        backend.copy_into(part.view(), &mut storage, &region)?;
        start += size;
    }
    Ok(Tensor::from_parts(storage, layout))
}

/// The backward of [`Tensor::narrow`]: place the cotangent `g` of the
/// narrowed region back into a zero tensor of the source geometry.
///
/// `axis`/`start` are the forward call's (resolved) arguments, `size` the
/// source size along `axis`, and `src_dims` the source dims. When the narrow
/// covered the whole axis this is the identity and `g` passes straight
/// through.
fn pad_with_zeros(
    g: &Tensor,
    axis: usize,
    start: usize,
    size: usize,
    src_dims: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let len = g.dims()[axis];
    if start == 0 && len == size {
        return Ok(g.clone());
    }
    let mut pad_dims = src_dims.to_vec();
    let mut pieces: Vec<Tensor> = Vec::with_capacity(3);
    if start > 0 {
        pad_dims[axis] = start;
        pieces.push(Tensor::zeros(pad_dims.clone(), dtype, device)?);
    }
    pieces.push(g.clone());
    let tail = size - start - len;
    if tail > 0 {
        pad_dims[axis] = tail;
        pieces.push(Tensor::zeros(pad_dims, dtype, device)?);
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, axis as isize)
}

/// Shared operand validation for [`Tensor::cat`]/[`Tensor::stack`]: dtype,
/// device and rank must match `first` exactly.
fn check_operand(op: &'static str, first: &Tensor, other: &Tensor) -> Result<()> {
    if other.dtype() != first.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: first.dtype(),
            got: other.dtype(),
        });
    }
    if other.device() != first.device() {
        return Err(Error::DeviceMismatch {
            op,
            expected: first.device(),
            got: other.device(),
        });
    }
    if other.rank() != first.rank() {
        return Err(Error::RankMismatch {
            op,
            expected: first.rank(),
            got: other.rank(),
        });
    }
    Ok(())
}

/// The first tensor of a `cat`/`stack` operand list, or a loud error for an
/// empty list (there is no shape to invent).
fn first_operand<'a>(op: &'static str, tensors: &'a [&'a Tensor]) -> Result<&'a Tensor> {
    tensors.first().copied().ok_or_else(|| Error::InvalidArg {
        op,
        msg: format!("{op} requires at least one tensor"),
    })
}

impl Tensor {
    /// The same elements under a new shape, with the same total element
    /// count (PyTorch `reshape` semantics).
    ///
    /// Returns a zero-copy view whenever the source layout admits one — always
    /// for a contiguous tensor, and for strided ones when the merged/split
    /// axes line up with contiguous runs. Otherwise the source is materialized
    /// into a fresh contiguous buffer first. Use
    /// [`is_contiguous`](Tensor::is_contiguous) if you need to know which
    /// happened.
    ///
    /// # Errors
    ///
    /// [`Error::ReshapeMismatch`] if the target element count differs from
    /// the source's.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?;
    /// let y = x.reshape([3, 2])?;
    /// assert_eq!(y.dims(), &[3, 2]);
    /// assert_eq!(y.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let target = shape.into();
        if target.num_elements() != self.num_elements() {
            return Err(Error::ReshapeMismatch {
                op: "reshape",
                from: self.shape().clone(),
                to: target,
            });
        }
        let out = match self.layout().reshape_view(target.clone()) {
            Some(layout) => view_of(self, layout),
            None => {
                // Not expressible as a view: materialize row-major, then the
                // target shape is trivially a contiguous view of the copy.
                let storage = dispatch::backend(self.device()).copy_strided(self.view())?;
                Tensor::from_parts(storage, Layout::contiguous(target)?)
            }
        };
        let src_dims: Vec<usize> = self.dims().to_vec();
        Ok(record(
            "reshape",
            out,
            &[self],
            Box::new(move |g| vec![g.reshape(src_dims.clone()).ok()]),
        ))
    }

    /// Swap two axes (negative indexing allowed). Zero-copy.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidAxis`] if either axis is outside `[-rank, rank)`.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?;
    /// let t = x.transpose(0, -1)?;
    /// assert_eq!(t.dims(), &[3, 2]);
    /// assert_eq!(t.to_vec::<f32>()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn transpose(&self, a: isize, b: isize) -> Result<Tensor> {
        let lhs = self.shape().resolve_axis(a, "transpose")?;
        let rhs = self.shape().resolve_axis(b, "transpose")?;
        let out = view_of(self, self.layout().transpose(lhs, rhs)?);
        Ok(record(
            "transpose",
            out,
            &[self],
            // A transposition is its own inverse.
            Box::new(move |g| vec![g.transpose(lhs as isize, rhs as isize).ok()]),
        ))
    }

    /// Reorder every axis: `perm[i]` names the source axis that becomes
    /// output axis `i` (negative indexing allowed). Zero-copy.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] if `perm` is not a permutation of the axes (wrong
    /// length or a repeated axis), [`Error::InvalidAxis`] if an entry is out
    /// of range.
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    /// let x = Tensor::zeros([2, 3, 4], DType::F32, &Device::Cpu)?;
    /// assert_eq!(x.permute(&[2, 0, 1])?.dims(), &[4, 2, 3]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn permute(&self, perm: &[isize]) -> Result<Tensor> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(Error::InvalidArg {
                op: "permute",
                msg: format!(
                    "permutation of length {} for a rank-{rank} tensor",
                    perm.len()
                ),
            });
        }
        let mut resolved = Vec::with_capacity(rank);
        for &axis in perm {
            resolved.push(self.shape().resolve_axis(axis, "permute")?);
        }
        // `Layout::permute` is the validation point for repeated axes; only
        // build the inverse once it has accepted `resolved`.
        let out = view_of(self, self.layout().permute(&resolved)?);
        let mut inverse = vec![0isize; rank];
        for (new_axis, &old_axis) in resolved.iter().enumerate() {
            inverse[old_axis] = new_axis as isize;
        }
        Ok(record(
            "permute",
            out,
            &[self],
            Box::new(move |g| vec![g.permute(&inverse).ok()]),
        ))
    }

    /// Drop a size-1 axis (negative indexing allowed). Zero-copy.
    ///
    /// There is one spelling: the axis is always explicit, so a shape never
    /// changes rank behind your back the way an argument-less `squeeze()`
    /// would.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidAxis`] if the axis is out of range, [`Error::InvalidArg`]
    /// if it is not of size 1.
    pub fn squeeze(&self, axis: isize) -> Result<Tensor> {
        let ax = self.shape().resolve_axis(axis, "squeeze")?;
        let out = view_of(self, self.layout().squeeze(ax)?);
        Ok(record(
            "squeeze",
            out,
            &[self],
            Box::new(move |g| vec![g.unsqueeze(ax as isize).ok()]),
        ))
    }

    /// Insert a size-1 axis at `axis`, an insertion position in `[-rank-1,
    /// rank]` (so `rank` appends and `-1` inserts before the last axis).
    /// Zero-copy.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidAxis`] if the insertion position is out of range.
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    /// let x = Tensor::zeros([2, 3], DType::F32, &Device::Cpu)?;
    /// assert_eq!(x.unsqueeze(1)?.dims(), &[2, 1, 3]);
    /// assert_eq!(x.unsqueeze(2)?.dims(), &[2, 3, 1]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn unsqueeze(&self, axis: isize) -> Result<Tensor> {
        let ax = self.shape().resolve_insert_axis(axis, "unsqueeze")?;
        let out = view_of(self, self.layout().unsqueeze(ax)?);
        Ok(record(
            "unsqueeze",
            out,
            &[self],
            Box::new(move |g| vec![g.squeeze(ax as isize).ok()]),
        ))
    }

    /// Restrict `axis` to the `len` positions starting at `start` (negative
    /// axis indexing allowed). Zero-copy; `len == 0` yields an empty tensor.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidAxis`] if the axis is out of range,
    /// [`Error::IndexOutOfBounds`] if `start + len` exceeds the axis size.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &Device::Cpu)?;
    /// assert_eq!(x.narrow(0, 1, 2)?.to_vec::<f32>()?, vec![2.0, 3.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn narrow(&self, axis: isize, start: usize, len: usize) -> Result<Tensor> {
        let ax = self.shape().resolve_axis(axis, "narrow")?;
        let out = view_of(self, self.layout().narrow(ax, start, len)?);
        let size = self.dims()[ax];
        let src_dims: Vec<usize> = self.dims().to_vec();
        let dtype = self.dtype();
        let device = self.device();
        Ok(record(
            "narrow",
            out,
            &[self],
            Box::new(move |g| {
                vec![pad_with_zeros(g, ax, start, size, &src_dims, dtype, &device).ok()]
            }),
        ))
    }

    /// Expand to `shape` under the NumPy/PyTorch broadcast rules: align to
    /// the right, and every axis of `self` must be equal to the target's or
    /// of size 1. Zero-copy — expanded axes get stride 0, so no elements are
    /// duplicated in memory.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `shape` is not a valid broadcast of the
    /// current shape (including a lower-rank target).
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let row = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [1, 3], &Device::Cpu)?;
    /// let b = row.broadcast_to([2, 3])?;
    /// assert_eq!(b.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let target = shape.into();
        let out = view_of(self, self.layout().broadcast_to(&target)?);
        let src_dims: Vec<usize> = self.dims().to_vec();
        Ok(record(
            "broadcast_to",
            out,
            &[self],
            // The transpose of a broadcast is a sum over the axes it expanded.
            Box::new(move |g| vec![g.sum_to(&src_dims).ok()]),
        ))
    }

    /// Concatenate `tensors` along an existing `axis` (negative indexing
    /// allowed). Every input must share the dtype, the device, the rank and
    /// every dimension other than `axis`.
    ///
    /// The result is a fresh contiguous tensor — it cannot be a view of
    /// several disjoint buffers.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] on an empty operand list (there is no shape to
    /// infer) or an element count that overflows `usize`;
    /// [`Error::DTypeMismatch`], [`Error::DeviceMismatch`],
    /// [`Error::RankMismatch`] or [`Error::ShapeMismatch`] when an operand
    /// disagrees with the first one; [`Error::InvalidAxis`] if `axis` is out
    /// of range.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0], [1, 2], &Device::Cpu)?;
    /// let b = Tensor::from_vec(vec![3.0f32, 4.0], [1, 2], &Device::Cpu)?;
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.dims(), &[2, 2]);
    /// assert_eq!(c.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
        let first = first_operand("cat", tensors)?;
        let ax = first.shape().resolve_axis(axis, "cat")?;
        let mut total = 0usize;
        for t in tensors {
            check_operand("cat", first, t)?;
            if t.dims()
                .iter()
                .enumerate()
                .any(|(a, &d)| a != ax && d != first.dims()[a])
            {
                return Err(Error::ShapeMismatch {
                    op: "cat",
                    lhs: first.shape().clone(),
                    rhs: t.shape().clone(),
                });
            }
            total = total
                .checked_add(t.dims()[ax])
                .ok_or_else(|| Error::InvalidArg {
                    op: "cat",
                    msg: format!("concatenated size along axis {ax} overflows usize"),
                })?;
        }

        let mut out_dims = first.dims().to_vec();
        out_dims[ax] = total;
        let out = concat_values(tensors, ax, &out_dims, false)?;

        let sizes: Vec<usize> = tensors.iter().map(|t| t.dims()[ax]).collect();
        Ok(record(
            "cat",
            out,
            tensors,
            Box::new(move |g| {
                // Each input owns one contiguous run of the output along `ax`.
                let mut start = 0usize;
                let mut grads = Vec::with_capacity(sizes.len());
                for &size in &sizes {
                    grads.push(g.narrow(ax as isize, start, size).ok());
                    start += size;
                }
                grads
            }),
        ))
    }

    /// Stack `tensors` along a **new** axis inserted at `axis`, an insertion
    /// position in `[-rank-1, rank]`. Every input must have exactly the same
    /// shape, dtype and device; the result has rank `rank + 1` with
    /// `tensors.len()` entries along the new axis.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] on an empty operand list;
    /// [`Error::DTypeMismatch`], [`Error::DeviceMismatch`],
    /// [`Error::RankMismatch`] or [`Error::ShapeMismatch`] when an operand
    /// disagrees with the first one; [`Error::InvalidAxis`] if `axis` is out
    /// of range.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?;
    /// let b = Tensor::from_vec(vec![3.0f32, 4.0], [2], &Device::Cpu)?;
    /// let s = Tensor::stack(&[&a, &b], 0)?;
    /// assert_eq!(s.dims(), &[2, 2]);
    /// assert_eq!(Tensor::stack(&[&a, &b], 1)?.to_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
        let first = first_operand("stack", tensors)?;
        let ax = first.shape().resolve_insert_axis(axis, "stack")?;
        for t in tensors {
            check_operand("stack", first, t)?;
            if t.dims() != first.dims() {
                return Err(Error::ShapeMismatch {
                    op: "stack",
                    lhs: first.shape().clone(),
                    rhs: t.shape().clone(),
                });
            }
        }

        let mut out_dims = first.dims().to_vec();
        out_dims.insert(ax, tensors.len());
        let out = concat_values(tensors, ax, &out_dims, true)?;

        let count = tensors.len();
        Ok(record(
            "stack",
            out,
            tensors,
            Box::new(move |g| {
                (0..count)
                    .map(|i| {
                        g.narrow(ax as isize, i, 1)
                            .and_then(|slice| slice.squeeze(ax as isize))
                            .ok()
                    })
                    .collect()
            }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{CpuStorage, Storage};
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    /// A contiguous f32 tensor on CPU.
    fn t_f32(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    /// `0.0, 1.0, ...` of length `n`, viewed as `shape`.
    fn iota(shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        let data: Vec<f32> = (0..shape.num_elements()).map(|v| v as f32).collect();
        Tensor::from_vec(data, shape, &CPU).unwrap()
    }

    /// The address of the f32 buffer behind `t`, for "view or copy?" checks.
    fn f32_buf_ptr(t: &Tensor) -> *const f32 {
        match t.storage() {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ptr(),
            _ => panic!("expected an f32 CPU tensor"),
        }
    }

    /// Assert `out` shares `src`'s buffer (the zero-copy contract).
    fn assert_shares(src: &Tensor, out: &Tensor) {
        assert_eq!(
            f32_buf_ptr(src),
            f32_buf_ptr(out),
            "expected a zero-copy view"
        );
    }

    // ------------------------------------------------------------------
    // reshape
    // ------------------------------------------------------------------

    #[test]
    fn reshape_contiguous_is_a_view() {
        let x = iota([2, 3, 4]);
        for target in [vec![24], vec![6, 4], vec![2, 12], vec![2, 2, 2, 3]] {
            let r = x.reshape(target.clone()).unwrap();
            assert_eq!(r.dims(), target.as_slice());
            assert!(r.is_contiguous());
            assert_shares(&x, &r);
            assert_eq!(r.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        }
        // Scalar round trip.
        let s = t_f32(&[7.0], [1]);
        let scalar = s.reshape(()).unwrap();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.to_scalar::<f32>().unwrap(), 7.0);
    }

    #[test]
    fn reshape_strided_views_when_possible_and_copies_otherwise() {
        // Merging the axes of a transposed tensor is not expressible as a
        // stride view: the copy branch must run.
        let x = iota([2, 3]);
        let tr = x.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        let merged = tr.reshape([6]).unwrap();
        assert!(merged.is_contiguous());
        assert_ne!(f32_buf_ptr(&tr), f32_buf_ptr(&merged));
        assert_eq!(
            merged.to_vec::<f32>().unwrap(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );

        // Splitting an axis that is itself a contiguous run stays a view.
        let x = iota([4, 6]);
        let tr = x.transpose(0, 1).unwrap(); // dims [6, 4], strides [1, 6]
        let split = tr.reshape([6, 2, 2]).unwrap();
        assert_shares(&tr, &split);
        assert_eq!(split.to_vec::<f32>().unwrap(), tr.to_vec::<f32>().unwrap());
    }

    #[test]
    fn reshape_of_a_narrowed_view_is_dense() {
        let x = iota([3, 4]);
        let n = x.narrow(1, 1, 2).unwrap();
        let r = n.reshape([6]).unwrap();
        assert!(r.is_contiguous());
        assert_eq!(r.storage().len(), 6);
        assert_eq!(
            r.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
        );
    }

    #[test]
    fn reshape_element_count_must_match() {
        let x = iota([2, 3]);
        assert!(matches!(
            x.reshape([4, 2]),
            Err(Error::ReshapeMismatch { op: "reshape", .. })
        ));
        match x.reshape([5]) {
            Err(Error::ReshapeMismatch { from, to, .. }) => {
                assert_eq!(from, Shape::from([2, 3]));
                assert_eq!(to, Shape::from([5]));
            }
            _ => panic!("expected a ReshapeMismatch"),
        }
        // An empty tensor can be reshaped to any other empty shape.
        let e = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        assert_eq!(e.reshape([3, 0]).unwrap().dims(), &[3, 0]);
        assert!(e.reshape([1]).is_err());
    }

    // ------------------------------------------------------------------
    // transpose / permute
    // ------------------------------------------------------------------

    #[test]
    fn transpose_swaps_axes_zero_copy() {
        let x = iota([2, 3]);
        let t = x.transpose(0, 1).unwrap();
        assert_eq!(t.dims(), &[3, 2]);
        assert_shares(&x, &t);
        assert_eq!(
            t.to_vec::<f32>().unwrap(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
        // Negative axes, and the involution property.
        assert_eq!(
            x.transpose(-2, -1).unwrap().to_vec::<f32>().unwrap(),
            t.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            t.transpose(0, 1).unwrap().to_vec::<f32>().unwrap(),
            x.to_vec::<f32>().unwrap()
        );
        // Swapping an axis with itself is the identity.
        assert_eq!(x.transpose(1, 1).unwrap().dims(), &[2, 3]);
    }

    #[test]
    fn transpose_rejects_out_of_range_axes() {
        let x = iota([2, 3]);
        assert!(matches!(
            x.transpose(0, 2),
            Err(Error::InvalidAxis {
                op: "transpose",
                ..
            })
        ));
        assert!(matches!(
            x.transpose(-3, 0),
            Err(Error::InvalidAxis {
                op: "transpose",
                ..
            })
        ));
        // A scalar has no axes at all.
        let s = t_f32(&[1.0], ());
        assert!(matches!(s.transpose(0, 0), Err(Error::InvalidAxis { .. })));
    }

    #[test]
    fn permute_reorders_every_axis_zero_copy() {
        let x = iota([2, 3, 4]);
        let p = x.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.dims(), &[4, 2, 3]);
        assert_shares(&x, &p);
        // Compare against the explicit gather of the permuted coordinates.
        let src = x.to_vec::<f32>().unwrap();
        let mut expected = Vec::new();
        for k in 0..4 {
            for i in 0..2 {
                for j in 0..3 {
                    expected.push(src[i * 12 + j * 4 + k]);
                }
            }
        }
        assert_eq!(p.to_vec::<f32>().unwrap(), expected);

        // Negative entries resolve like everywhere else; the identity
        // permutation is a no-op; rank-0 accepts the empty permutation.
        assert_eq!(x.permute(&[-1, -3, -2]).unwrap().dims(), &[4, 2, 3]);
        assert_eq!(x.permute(&[0, 1, 2]).unwrap().dims(), &[2, 3, 4]);
        assert_eq!(t_f32(&[1.0], ()).permute(&[]).unwrap().rank(), 0);
    }

    #[test]
    fn permute_rejects_non_permutations() {
        let x = iota([2, 3, 4]);
        assert!(matches!(
            x.permute(&[0, 1]),
            Err(Error::InvalidArg { op: "permute", .. })
        ));
        assert!(matches!(
            x.permute(&[0, 0, 1]),
            Err(Error::InvalidArg { op: "permute", .. })
        ));
        assert!(matches!(
            x.permute(&[0, 1, 3]),
            Err(Error::InvalidAxis { op: "permute", .. })
        ));
        // A negative entry aliasing an earlier positive one is still a repeat.
        assert!(matches!(
            x.permute(&[0, 1, -2]),
            Err(Error::InvalidArg { op: "permute", .. })
        ));
    }

    // ------------------------------------------------------------------
    // squeeze / unsqueeze
    // ------------------------------------------------------------------

    #[test]
    fn squeeze_and_unsqueeze_are_inverse_views() {
        let x = iota([2, 1, 3]);
        let s = x.squeeze(1).unwrap();
        assert_eq!(s.dims(), &[2, 3]);
        assert_shares(&x, &s);
        assert_eq!(s.unsqueeze(1).unwrap().dims(), &[2, 1, 3]);

        // Negative axes on both.
        assert_eq!(x.squeeze(-2).unwrap().dims(), &[2, 3]);
        assert_eq!(s.unsqueeze(-1).unwrap().dims(), &[2, 3, 1]);
        assert_eq!(s.unsqueeze(-3).unwrap().dims(), &[1, 2, 3]);
        // The append position is `rank`, which is *not* a valid `resolve_axis`
        // index — insertion axes go up to and including the rank.
        assert_eq!(s.unsqueeze(2).unwrap().dims(), &[2, 3, 1]);
        // Rank-0 gains its first axis.
        assert_eq!(t_f32(&[5.0], ()).unsqueeze(0).unwrap().dims(), &[1]);

        // Values are untouched by either.
        assert_eq!(
            s.unsqueeze(0).unwrap().to_vec::<f32>().unwrap(),
            x.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn squeeze_and_unsqueeze_errors() {
        let x = iota([2, 1, 3]);
        assert!(matches!(
            x.squeeze(0),
            Err(Error::InvalidArg { op: "squeeze", .. })
        ));
        assert!(matches!(
            x.squeeze(3),
            Err(Error::InvalidAxis { op: "squeeze", .. })
        ));
        assert!(matches!(
            x.unsqueeze(4),
            Err(Error::InvalidAxis {
                op: "unsqueeze",
                ..
            })
        ));
        assert!(matches!(
            x.unsqueeze(-5),
            Err(Error::InvalidAxis {
                op: "unsqueeze",
                ..
            })
        ));
    }

    // ------------------------------------------------------------------
    // narrow
    // ------------------------------------------------------------------

    #[test]
    fn narrow_restricts_an_axis_zero_copy() {
        let x = iota([3, 4]);
        let n = x.narrow(1, 1, 2).unwrap();
        assert_eq!(n.dims(), &[3, 2]);
        assert_shares(&x, &n);
        assert_eq!(
            n.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
        );
        // Negative axis, whole-axis narrow, and an empty narrow.
        assert_eq!(x.narrow(-1, 0, 4).unwrap().dims(), &[3, 4]);
        let empty = x.narrow(0, 2, 0).unwrap();
        assert_eq!(empty.dims(), &[0, 4]);
        assert!(empty.to_vec::<f32>().unwrap().is_empty());
        // Narrowing twice composes.
        assert_eq!(
            x.narrow(0, 1, 2)
                .unwrap()
                .narrow(1, 3, 1)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![7.0, 11.0]
        );
    }

    #[test]
    fn narrow_bounds_are_loud() {
        let x = iota([3, 4]);
        assert!(matches!(
            x.narrow(0, 2, 2),
            Err(Error::IndexOutOfBounds { op: "narrow", .. })
        ));
        assert!(matches!(
            x.narrow(0, 4, 0),
            Err(Error::IndexOutOfBounds { op: "narrow", .. })
        ));
        assert!(matches!(
            x.narrow(2, 0, 1),
            Err(Error::InvalidAxis { op: "narrow", .. })
        ));
        assert!(matches!(
            x.narrow(0, usize::MAX, 1),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }

    // ------------------------------------------------------------------
    // broadcast_to
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_to_expands_without_copying() {
        let row = t_f32(&[1.0, 2.0, 3.0], [1, 3]);
        let b = row.broadcast_to([2, 3]).unwrap();
        assert_eq!(b.dims(), &[2, 3]);
        assert_shares(&row, &b);
        assert_eq!(
            b.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );

        // New leading axes are free, and a scalar broadcasts to anything.
        let b = row.broadcast_to([4, 2, 3]).unwrap();
        assert_eq!(b.dims(), &[4, 2, 3]);
        assert_eq!(b.num_elements(), 24);
        let s = t_f32(&[9.0], ());
        assert_eq!(
            s.broadcast_to([2, 2]).unwrap().to_vec::<f32>().unwrap(),
            vec![9.0; 4]
        );
        // The identity broadcast is accepted.
        assert_eq!(row.broadcast_to([1, 3]).unwrap().dims(), &[1, 3]);
    }

    #[test]
    fn broadcast_to_rejects_incompatible_targets() {
        let x = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
        assert!(matches!(
            x.broadcast_to([4, 2]),
            Err(Error::ShapeMismatch {
                op: "broadcast_to",
                ..
            })
        ));
        // Lower rank is not a broadcast.
        assert!(matches!(
            x.broadcast_to([2]),
            Err(Error::ShapeMismatch {
                op: "broadcast_to",
                ..
            })
        ));
    }

    // ------------------------------------------------------------------
    // cat
    // ------------------------------------------------------------------

    #[test]
    fn cat_along_each_axis() {
        let a = t_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = t_f32(&[5.0, 6.0], [1, 2]);
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.dims(), &[3, 2]);
        assert!(c.is_contiguous());
        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Along the last axis the regions interleave per row.
        let d = t_f32(&[7.0, 8.0], [2, 1]);
        let c = Tensor::cat(&[&a, &d], -1).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 7.0, 3.0, 4.0, 8.0]
        );

        // A middle axis of a rank-3 tensor, three operands, uneven sizes.
        let x = iota([2, 1, 3]);
        let y = iota([2, 2, 3]);
        let c = Tensor::cat(&[&x, &y, &x], 1).unwrap();
        assert_eq!(c.dims(), &[2, 4, 3]);
        let (xs, ys) = (x.to_vec::<f32>().unwrap(), y.to_vec::<f32>().unwrap());
        let mut expected = Vec::new();
        for group in 0..2 {
            expected.extend_from_slice(&xs[group * 3..group * 3 + 3]);
            expected.extend_from_slice(&ys[group * 6..group * 6 + 6]);
            expected.extend_from_slice(&xs[group * 3..group * 3 + 3]);
        }
        assert_eq!(c.to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn cat_handles_single_operands_views_and_empties() {
        // One operand: a contiguous copy of the input's values.
        let x = iota([2, 3]);
        let one = Tensor::cat(&[&x], 0).unwrap();
        assert_eq!(one.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());

        // Strided operands are materialized in logical order, not raw order.
        let tr = x.transpose(0, 1).unwrap();
        let c = Tensor::cat(&[&tr, &tr], 1).unwrap();
        assert_eq!(c.dims(), &[3, 4]);
        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![0.0, 3.0, 0.0, 3.0, 1.0, 4.0, 1.0, 4.0, 2.0, 5.0, 2.0, 5.0]
        );

        // A zero-length operand contributes nothing but is legal.
        let empty = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        let c = Tensor::cat(&[&empty, &x, &empty], 0).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
        assert_eq!(c.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        // All-empty stays empty.
        let c = Tensor::cat(&[&empty, &empty], 0).unwrap();
        assert_eq!(c.dims(), &[0, 3]);
        assert_eq!(c.num_elements(), 0);
    }

    #[test]
    fn cat_preserves_non_float_dtypes() {
        let a = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        let b = Tensor::from_vec(vec![3i64], [1], &CPU).unwrap();
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.dtype(), DType::I64);
        assert_eq!(c.to_vec::<i64>().unwrap(), vec![1, 2, 3]);

        let a = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        let c = Tensor::cat(&[&a, &a], 0).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.to_vec::<bool>().unwrap(), vec![true, false, true, false]);
    }

    #[test]
    fn cat_operand_mismatches_are_loud() {
        let a = iota([2, 3]);
        assert!(matches!(
            Tensor::cat(&[], 0),
            Err(Error::InvalidArg { op: "cat", .. })
        ));
        // Mismatched non-cat axis.
        let bad = iota([2, 4]);
        assert!(matches!(
            Tensor::cat(&[&a, &bad], 0),
            Err(Error::ShapeMismatch { op: "cat", .. })
        ));
        // ...but the same shapes are fine when that axis *is* the cat axis.
        assert_eq!(Tensor::cat(&[&a, &bad], 1).unwrap().dims(), &[2, 7]);
        // Rank.
        let r3 = iota([2, 3, 1]);
        assert!(matches!(
            Tensor::cat(&[&a, &r3], 0),
            Err(Error::RankMismatch { op: "cat", .. })
        ));
        // Dtype.
        let ints = Tensor::from_vec(vec![0i64; 6], [2, 3], &CPU).unwrap();
        assert!(matches!(
            Tensor::cat(&[&a, &ints], 0),
            Err(Error::DTypeMismatch { op: "cat", .. })
        ));
        // Axis.
        assert!(matches!(
            Tensor::cat(&[&a], 2),
            Err(Error::InvalidAxis { op: "cat", .. })
        ));
        // Scalars have no axis to concatenate along.
        let s = t_f32(&[1.0], ());
        assert!(matches!(
            Tensor::cat(&[&s, &s], 0),
            Err(Error::InvalidAxis { op: "cat", .. })
        ));
    }

    // ------------------------------------------------------------------
    // stack
    // ------------------------------------------------------------------

    #[test]
    fn stack_inserts_a_new_axis() {
        let a = t_f32(&[1.0, 2.0], [2]);
        let b = t_f32(&[3.0, 4.0], [2]);

        let s = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(s.dims(), &[2, 2]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let s = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(s.dims(), &[2, 2]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
        // -1 is the same insertion position as `rank`.
        assert_eq!(
            Tensor::stack(&[&a, &b], -1)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0, 3.0, 2.0, 4.0]
        );

        // A middle axis on rank-2 inputs, three operands.
        let x = iota([2, 3]);
        let s = Tensor::stack(&[&x, &x, &x], 1).unwrap();
        assert_eq!(s.dims(), &[2, 3, 3]);
        let xs = x.to_vec::<f32>().unwrap();
        let mut expected = Vec::new();
        for row in 0..2 {
            for _ in 0..3 {
                expected.extend_from_slice(&xs[row * 3..row * 3 + 3]);
            }
        }
        assert_eq!(s.to_vec::<f32>().unwrap(), expected);

        // Stacking scalars is the canonical way to build a rank-1 tensor.
        let z = t_f32(&[9.0], ());
        assert_eq!(Tensor::stack(&[&z, &z], 0).unwrap().dims(), &[2]);
    }

    #[test]
    fn stack_reads_strided_operands_in_logical_order() {
        let x = iota([2, 3]);
        let tr = x.transpose(0, 1).unwrap();
        let s = Tensor::stack(&[&tr, &tr], 0).unwrap();
        assert_eq!(s.dims(), &[2, 3, 2]);
        let expected: Vec<f32> = tr
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .copied()
            .chain(tr.to_vec::<f32>().unwrap())
            .collect();
        assert_eq!(s.to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn stack_operand_mismatches_are_loud() {
        let a = iota([2, 3]);
        assert!(matches!(
            Tensor::stack(&[], 0),
            Err(Error::InvalidArg { op: "stack", .. })
        ));
        // stack requires *identical* shapes, unlike cat.
        let b = iota([2, 4]);
        assert!(matches!(
            Tensor::stack(&[&a, &b], 0),
            Err(Error::ShapeMismatch { op: "stack", .. })
        ));
        let r3 = iota([2, 3, 1]);
        assert!(matches!(
            Tensor::stack(&[&a, &r3], 0),
            Err(Error::RankMismatch { op: "stack", .. })
        ));
        let ints = Tensor::from_vec(vec![0i64; 6], [2, 3], &CPU).unwrap();
        assert!(matches!(
            Tensor::stack(&[&a, &ints], 0),
            Err(Error::DTypeMismatch { op: "stack", .. })
        ));
        // Insertion axis range is [-rank-1, rank].
        assert!(matches!(
            Tensor::stack(&[&a], 3),
            Err(Error::InvalidAxis { op: "stack", .. })
        ));
        assert!(matches!(
            Tensor::stack(&[&a], -4),
            Err(Error::InvalidAxis { op: "stack", .. })
        ));
        assert_eq!(Tensor::stack(&[&a], 2).unwrap().dims(), &[2, 3, 1]);
    }

    // ------------------------------------------------------------------
    // pad_with_zeros — the value-level half of the `narrow` backward.
    //
    // The FD case below covers the closure end to end; this test covers the
    // same placement property directly on the helper, where a failure names
    // the padding maths rather than the whole gradient chain.
    // ------------------------------------------------------------------

    #[test]
    fn pad_with_zeros_places_the_region_and_zeroes_the_rest() {
        // Source [3, 4]; the forward narrowed axis 1 to [1, 3).
        let g = iota([3, 2]);
        let padded = pad_with_zeros(&g, 1, 1, 4, &[3, 4], DType::F32, &CPU).unwrap();
        assert_eq!(padded.dims(), &[3, 4]);
        assert!(padded.is_contiguous());
        assert_eq!(
            padded.to_vec::<f32>().unwrap(),
            vec![
                0.0, 0.0, 1.0, 0.0, // row 0
                0.0, 2.0, 3.0, 0.0, // row 1
                0.0, 4.0, 5.0, 0.0, // row 2
            ]
        );

        // Leading slice: only a trailing pad.
        let g = iota([1, 4]);
        let padded = pad_with_zeros(&g, 0, 0, 3, &[3, 4], DType::F32, &CPU).unwrap();
        let mut expected = vec![0.0, 1.0, 2.0, 3.0];
        expected.extend(std::iter::repeat_n(0.0f32, 8));
        assert_eq!(padded.to_vec::<f32>().unwrap(), expected);

        // Trailing slice: only a leading pad.
        let padded = pad_with_zeros(&g, 0, 2, 3, &[3, 4], DType::F32, &CPU).unwrap();
        let mut expected = vec![0.0f32; 8];
        expected.extend([0.0, 1.0, 2.0, 3.0]);
        assert_eq!(padded.to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn pad_with_zeros_degenerate_cases() {
        // A whole-axis narrow is the identity: the same buffer flows through,
        // with no allocation at all.
        let g = iota([3, 4]);
        let same = pad_with_zeros(&g, 1, 0, 4, &[3, 4], DType::F32, &CPU).unwrap();
        assert_eq!(f32_buf_ptr(&g), f32_buf_ptr(&same));

        // An empty narrow contributes nothing: an all-zero source-shaped
        // gradient, whatever the start offset was.
        for start in 0..=3usize {
            let g = Tensor::zeros([0, 4], DType::F32, &CPU).unwrap();
            let padded = pad_with_zeros(&g, 0, start, 3, &[3, 4], DType::F32, &CPU).unwrap();
            assert_eq!(padded.dims(), &[3, 4]);
            assert_eq!(padded.to_vec::<f32>().unwrap(), vec![0.0; 12]);
        }

        // Non-float dtypes survive the round trip (the helper is dtype-generic
        // even though only float tensors carry gradients today).
        let g = Tensor::from_vec(vec![7i64], [1], &CPU).unwrap();
        let padded = pad_with_zeros(&g, 0, 1, 3, &[3], DType::I64, &CPU).unwrap();
        assert_eq!(padded.to_vec::<i64>().unwrap(), vec![0, 7, 0]);
    }

    // ------------------------------------------------------------------
    // Composition sanity: these ops chain the way the layer zoo will use
    // them (flatten-for-linear, head split/merge in attention).
    // ------------------------------------------------------------------

    #[test]
    fn attention_style_head_split_and_merge_round_trips() {
        // [batch, time, embed] -> [batch, heads, time, head_dim] -> back.
        let (b, t, e, h) = (2usize, 3usize, 4usize, 2usize);
        let x = iota([b, t, e]);
        let heads = x
            .reshape([b, t, h, e / h])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        assert_eq!(heads.dims(), &[b, h, t, e / h]);
        let back = heads.transpose(1, 2).unwrap().reshape([b, t, e]).unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
    }

    #[test]
    fn cat_is_the_inverse_of_narrowing_at_the_split_points() {
        let x = iota([4, 5]);
        for axis in 0..2usize {
            let size = x.dims()[axis];
            for split in 0..=size {
                let head = x.narrow(axis as isize, 0, split).unwrap();
                let tail = x.narrow(axis as isize, split, size - split).unwrap();
                let joined = Tensor::cat(&[&head, &tail], axis as isize).unwrap();
                assert_eq!(joined.dims(), x.dims());
                assert_eq!(
                    joined.to_vec::<f32>().unwrap(),
                    x.to_vec::<f32>().unwrap(),
                    "axis {axis} split {split}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Backward: finite-difference cases against the single `check_grad`
    // harness, activated by **T31** now that T30's engine is live.
    //
    // `check_grad` needs a scalar-valued `f`, and the reduction ops that
    // would supply one live in T23 (not in this task's layer). Every case
    // therefore scalarizes with `pick`, which selects one output element
    // using only this file's own ops. `check_grad` perturbs *every* input
    // element, so the full gradient tensor is still checked — against a
    // one-hot cotangent rather than an all-ones one.
    // ------------------------------------------------------------------

    /// The element of `t` at row-major position `flat`, as a rank-0 tensor.
    fn pick(t: &Tensor, flat: usize) -> Result<Tensor> {
        let mut coords = vec![0usize; t.rank()];
        let mut rest = flat;
        for axis in (0..t.rank()).rev() {
            coords[axis] = rest % t.dims()[axis];
            rest /= t.dims()[axis];
        }
        let mut cur = t.clone();
        for (axis, &c) in coords.iter().enumerate() {
            cur = cur.narrow(axis as isize, c, 1)?;
        }
        cur.reshape(())
    }

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-4;

    #[test]
    fn grad_reshape() {
        let x = iota([2, 3]);
        for flat in 0..6 {
            check_grad(
                move |xs| pick(&xs[0].reshape([3, 2])?, flat),
                std::slice::from_ref(&x),
                EPS,
                TOL,
            )
            .unwrap();
        }
        // The copy branch (a transposed source merged into one axis).
        let x = iota([2, 3]);
        check_grad(
            |xs| pick(&xs[0].transpose(0, 1)?.reshape([6])?, 4),
            &[x],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_transpose_and_permute() {
        let x = iota([2, 3]);
        check_grad(|xs| pick(&xs[0].transpose(0, 1)?, 3), &[x], EPS, TOL).unwrap();

        let x = iota([2, 3, 4]);
        check_grad(|xs| pick(&xs[0].permute(&[2, 0, 1])?, 17), &[x], EPS, TOL).unwrap();
    }

    #[test]
    fn grad_squeeze_and_unsqueeze() {
        let x = iota([2, 1, 3]);
        check_grad(|xs| pick(&xs[0].squeeze(1)?, 4), &[x], EPS, TOL).unwrap();

        let x = iota([2, 3]);
        check_grad(|xs| pick(&xs[0].unsqueeze(-1)?, 5), &[x], EPS, TOL).unwrap();
    }

    #[test]
    fn grad_narrow_pads_with_zeros() {
        let x = iota([3, 4]);
        // Interior slice: the gradient must be zero outside [1, 3).
        for flat in 0..6 {
            check_grad(
                move |xs| pick(&xs[0].narrow(1, 1, 2)?, flat),
                std::slice::from_ref(&x),
                EPS,
                TOL,
            )
            .unwrap();
        }
        // Leading and trailing slices exercise the one-sided pads.
        check_grad(
            |xs| pick(&xs[0].narrow(0, 0, 1)?, 2),
            std::slice::from_ref(&x),
            EPS,
            TOL,
        )
        .unwrap();
        check_grad(|xs| pick(&xs[0].narrow(0, 2, 1)?, 2), &[x], EPS, TOL).unwrap();
    }

    #[test]
    fn grad_broadcast_to_sums_the_expanded_axes() {
        let x = iota([1, 3]);
        for flat in 0..6 {
            check_grad(
                move |xs| pick(&xs[0].broadcast_to([2, 3])?, flat),
                std::slice::from_ref(&x),
                EPS,
                TOL,
            )
            .unwrap();
        }
        // A new leading axis as well as an expanded one.
        let x = iota([3]);
        check_grad(
            |xs| pick(&xs[0].broadcast_to([2, 2, 3])?, 7),
            &[x],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_cat_splits_the_cotangent_per_input() {
        let a = iota([2, 2]);
        let b = iota([1, 2]);
        for flat in 0..6 {
            check_grad(
                move |xs| pick(&Tensor::cat(&[&xs[0], &xs[1]], 0)?, flat),
                &[a.clone(), b.clone()],
                EPS,
                TOL,
            )
            .unwrap();
        }
        // The last-axis case, where the regions interleave.
        let a = iota([2, 2]);
        let b = iota([2, 1]);
        check_grad(
            |xs| pick(&Tensor::cat(&[&xs[0], &xs[1]], -1)?, 4),
            &[a, b],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_stack_splits_the_cotangent_per_input() {
        let a = iota([2, 3]);
        let b = iota([2, 3]);
        check_grad(
            |xs| pick(&Tensor::stack(&[&xs[0], &xs[1]], 0)?, 9),
            &[a.clone(), b.clone()],
            EPS,
            TOL,
        )
        .unwrap();
        check_grad(
            |xs| pick(&Tensor::stack(&[&xs[0], &xs[1]], 1)?, 5),
            &[a, b],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_flows_through_a_chain_of_view_ops() {
        // The attention-style reshape/transpose chain, differentiated end to
        // end: a regression net for composing the inverses in the right order.
        let x = iota([2, 3, 4]);
        check_grad(
            |xs| {
                let heads = xs[0].reshape([2, 3, 2, 2])?.transpose(1, 2)?;
                let merged = heads.transpose(1, 2)?.reshape([2, 3, 4])?;
                pick(&merged.narrow(2, 1, 2)?, 7)
            },
            &[x],
            EPS,
            TOL,
        )
        .unwrap();
    }
}
