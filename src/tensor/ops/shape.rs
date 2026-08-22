//! Shape and view operations: `reshape`, `transpose`, `permute`,
//! `squeeze`, `unsqueeze`, `narrow`, `broadcast_to`, `cat`, `stack`,
//! `repeat`, `split` and `chunk`.
//!
//! # Views and contiguity
//!
//! `transpose`, `permute`, `squeeze`, `unsqueeze`, `narrow` and
//! `broadcast_to` are **always zero-copy**: they re-describe the same storage
//! through a new [`Layout`] (an `Arc` bump, no element copy). `reshape`
//! follows `PyTorch` semantics — a view when the source layout permits one
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
//! | `repeat` | none of its own — it is `cat` of the same tensor, and the engine sums the one input's several parent slots, so the cotangent adds over the repeats |
//! | `split` / `chunk` | none of their own — each piece is a `narrow`, and the engine sums the pieces' zero-padded cotangents back onto the source |
//!
//! No backward closure captures a tensor: they close over dimensions, axis
//! indices, the dtype and the device only, which trivially satisfies the
//! detached-output capture rule.

use super::{same_device, same_dtype, same_rank};
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
    same_dtype(op, first, other)?;
    same_device(op, first, other)?;
    same_rank(op, first, other)
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
    /// count (`PyTorch` `reshape` semantics).
    ///
    /// Returns a zero-copy view whenever the source layout admits one — always
    /// for a row-major source, and for strided ones when the merged/split axes
    /// line up with dense runs. Otherwise the source is materialized into a
    /// fresh buffer first. Which of the two happened is deliberately not
    /// observable: it is an allocation choice, not part of the contract.
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
        // Checked, because `target` is caller-supplied: the plain product panics
        // on overflow in debug and wraps in release, and a wrapped count could
        // *match* this tensor's real element count and pass the guard below.
        let Some(target_elements) = target.checked_num_elements() else {
            return Err(Error::ReshapeMismatch {
                op: "reshape",
                from: self.shape().clone(),
                to: target,
            });
        };
        if target_elements != self.num_elements() {
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
            Box::new(move |g| Ok(vec![Some(g.reshape(src_dims.clone())?)])),
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
            Box::new(move |g| Ok(vec![Some(g.transpose(lhs as isize, rhs as isize)?)])),
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
            Box::new(move |g| Ok(vec![Some(g.permute(&inverse)?)])),
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
            Box::new(move |g| Ok(vec![Some(g.unsqueeze(ax as isize)?)])),
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
            Box::new(move |g| Ok(vec![Some(g.squeeze(ax as isize)?)])),
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
                Ok(vec![Some(pad_with_zeros(
                    g, ax, start, size, &src_dims, dtype, &device,
                )?)])
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
            Box::new(move |g| Ok(vec![Some(g.sum_to(&src_dims)?)])),
        ))
    }

    /// Concatenate `tensors` along an existing `axis` (negative indexing
    /// allowed). Every input must share the dtype, the device, the rank and
    /// every dimension other than `axis`.
    ///
    /// The result is a fresh allocation — it cannot be a view of several
    /// disjoint buffers.
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
                    grads.push(Some(g.narrow(ax as isize, start, size)?));
                    start += size;
                }
                Ok(grads)
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
                        let slice = g.narrow(ax as isize, i, 1)?;
                        Ok(Some(slice.squeeze(ax as isize)?))
                    })
                    .collect()
            }),
        ))
    }

    // ---- repeat / split / chunk ------------------------------------------

    /// Tile the whole tensor: axis `a` of the result has length
    /// `dims()[a] * reps[a]` (`PyTorch`'s `repeat`, restricted to the same
    /// rank — unlike `PyTorch`, this does not prepend axes for a shorter
    /// `reps`, so a rank mismatch is a loud error rather than an implicit
    /// reshape).
    ///
    /// Built as repeated [`cat`](Tensor::cat) of `self` with itself, one axis
    /// at a time. That is also what makes the gradient free and correct
    /// without a dedicated backward: `cat` records one edge per occurrence of
    /// the *same* input node, and [`backward`](Tensor::backward) already sums
    /// every edge that targets one node (the same mechanism weight tying
    /// relies on) — so the cotangent naturally adds back the `reps[a]` tiled
    /// copies onto `self`.
    ///
    /// # Errors
    /// [`Error::RankMismatch`] if `reps.len()` is not this tensor's rank.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?;
    /// assert_eq!(x.repeat(&[3])?.to_vec::<f32>()?, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn repeat(&self, reps: &[usize]) -> Result<Tensor> {
        const OP: &str = "repeat";
        if reps.len() != self.rank() {
            return Err(Error::RankMismatch {
                op: OP,
                expected: self.rank(),
                got: reps.len(),
            });
        }
        let mut cur = self.clone();
        for (axis, &r) in reps.iter().enumerate() {
            if r == 1 {
                continue;
            }
            if r == 0 {
                cur = cur.narrow(axis as isize, 0, 0)?;
                continue;
            }
            let copies: Vec<&Tensor> = std::iter::repeat_n(&cur, r).collect();
            cur = Tensor::cat(&copies, axis as isize)?;
        }
        Ok(cur)
    }

    /// Split into consecutive, exactly-sized pieces along `axis`: the `i`-th
    /// output has length `sizes[i]`, and the lengths must sum to the axis
    /// size exactly (`PyTorch`'s `split` with an explicit size list).
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] if `axis` is out of range,
    /// [`Error::InvalidArg`] if `sizes` does not sum to the axis length.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], [5], &Device::Cpu)?;
    /// let parts = x.split(&[2, 3], 0)?;
    /// assert_eq!(parts[0].to_vec::<f32>()?, vec![1.0, 2.0]);
    /// assert_eq!(parts[1].to_vec::<f32>()?, vec![3.0, 4.0, 5.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn split(&self, sizes: &[usize], axis: isize) -> Result<Vec<Tensor>> {
        const OP: &str = "split";
        let ax = self.shape().resolve_axis(axis, OP)?;
        let axis_len = self.dims()[ax];
        let total: usize = sizes.iter().sum();
        if total != axis_len {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("split sizes sum to {total}, but axis {ax} has length {axis_len}"),
            });
        }
        let mut start = 0;
        let mut out = Vec::with_capacity(sizes.len());
        for &len in sizes {
            out.push(self.narrow(ax as isize, start, len)?);
            start += len;
        }
        Ok(out)
    }

    /// Split into at most `n` pieces along `axis`, each of size
    /// `ceil(axis_len / n)` except the last, which holds the remainder
    /// (`PyTorch`'s `chunk`) — the non-dividing case yields *fewer than `n`*
    /// pieces rather than an empty trailing one. An empty axis is the one
    /// edge case: it yields a single empty piece rather than zero pieces, so
    /// callers never have to special-case an empty result `Vec`.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] if `axis` is out of range,
    /// [`Error::InvalidArg`] if `n` is zero.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], [5], &Device::Cpu)?;
    /// let parts = x.chunk(2, 0)?;
    /// assert_eq!(parts[0].to_vec::<f32>()?, vec![1.0, 2.0, 3.0]);
    /// assert_eq!(parts[1].to_vec::<f32>()?, vec![4.0, 5.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn chunk(&self, n: usize, axis: isize) -> Result<Vec<Tensor>> {
        const OP: &str = "chunk";
        if n == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: "chunk count must be non-zero".to_owned(),
            });
        }
        let ax = self.shape().resolve_axis(axis, OP)?;
        let axis_len = self.dims()[ax];
        let chunk_size = axis_len.div_ceil(n).max(1);
        let mut out = Vec::new();
        let mut start = 0;
        while start < axis_len {
            let len = chunk_size.min(axis_len - start);
            out.push(self.narrow(ax as isize, start, len)?);
            start += len;
        }
        if out.is_empty() {
            out.push(self.narrow(ax as isize, 0, 0)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
