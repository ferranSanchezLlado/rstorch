//! Indexing ops: the epoch-1/2 vocabulary behind embeddings, KV caches, and
//! label lookups (the "indexing set").
//!
//! Two indexed reads, each with its scatter-accumulate backward:
//!
//! | forward | index shape | backward |
//! |---|---|---|
//! | [`index_select`](Tensor::index_select) | 1-D, whole slices | `index_add` |
//! | [`gather`](Tensor::gather) | same rank as the source, per element | `scatter_add` |
//!
//! plus the index/mask builders that feed them:
//! [`index_range`](Tensor::index_range) (the `arange`-built `[0, len)` index
//! vector), [`index_vec`](Tensor::index_vec) (host positions → an `I64` index
//! tensor), and [`causal_mask`](Tensor::causal_mask) (the comparison-built
//! `[t, t]` attention mask).
//!
//! Indices are always [`I64`](crate::DType::I64) tensors that live on the
//! same device as the data — masks and index sets stay on-device, and a
//! negative index is a loud
//! [`IndexOutOfBounds`](crate::Error::IndexOutOfBounds) rather than a
//! Python-style wrap.

use super::{require_dtype, same_device};
use crate::autograd::{self, BackwardFn};
use crate::backend::{CmpOp, View, dispatch};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Validate an index operand: `I64` dtype, same device as the data it
/// indexes. Rank is checked per op (1-D for `index_select`, same-rank for
/// `gather`).
fn check_index_operand(op: &'static str, data: &Tensor, indices: &Tensor) -> Result<()> {
    require_dtype(op, indices, DType::I64)?;
    same_device(op, data, indices)
}

/// The cotangent of an [`index_select`](Tensor::index_select) input: a fresh
/// zero tensor of `shape` with the upstream gradient's slices accumulated
/// back at `indices` (backend `index_add`).
///
/// Not a differentiable op — it runs *inside* a backward closure, on an
/// already-computed cotangent.
fn index_add_into_zeros(
    shape: &Shape,
    dtype: DType,
    device: Device,
    axis: usize,
    indices: &Tensor,
    src: &Tensor,
) -> Result<Tensor> {
    let zeros = Tensor::zeros(shape.clone(), dtype, &device)?;
    let storage =
        dispatch::backend(device).index_add(zeros.view(), axis, indices.view(), src.view())?;
    Ok(Tensor::from_parts(
        storage,
        Layout::contiguous(shape.clone())?,
    ))
}

/// The cotangent of a [`gather`](Tensor::gather) input: a fresh zero tensor
/// of `shape` with the upstream gradient scattered back element-wise through
/// `indices` (backend `scatter_add`); duplicate destinations accumulate.
///
/// Not a differentiable op — see [`index_add_into_zeros`].
fn scatter_add_into_zeros(
    shape: &Shape,
    dtype: DType,
    device: Device,
    axis: usize,
    indices: &Tensor,
    src: &Tensor,
) -> Result<Tensor> {
    let zeros = Tensor::zeros(shape.clone(), dtype, &device)?;
    let storage =
        dispatch::backend(device).scatter_add(zeros.view(), axis, indices.view(), src.view())?;
    Ok(Tensor::from_parts(
        storage,
        Layout::contiguous(shape.clone())?,
    ))
}

impl Tensor {
    // ---- indexed reads ---------------------------------------------------

    /// Select whole slices along `axis` at the positions in the 1-D
    /// [`I64`](crate::DType::I64) tensor `indices` (`PyTorch` `index_select`).
    ///
    /// The result has this tensor's shape with `axis` resized to
    /// `indices.num_elements()`, is contiguous, and keeps the dtype and
    /// device. This is the embedding lookup: `weight.index_select(0, &ids)`
    /// on a `[vocab, embed]` table with `[n]` ids yields `[n, embed]`.
    /// Repeated indices are fine (and the reason the backward accumulates).
    ///
    /// `axis` is an `isize` with negative indexing (`-1` is the last axis).
    ///
    /// # Errors
    ///
    /// - [`InvalidAxis`](crate::Error::InvalidAxis) if `axis` is outside
    ///   `[-rank, rank)`.
    /// - [`DTypeMismatch`](crate::Error::DTypeMismatch) if `indices` is not
    ///   `I64`, [`DeviceMismatch`](crate::Error::DeviceMismatch) if it lives
    ///   on another device.
    /// - [`RankMismatch`](crate::Error::RankMismatch) if `indices` is not
    ///   rank 1 (use [`gather`](Tensor::gather) for a same-rank index grid).
    /// - [`IndexOutOfBounds`](crate::Error::IndexOutOfBounds) for a negative
    ///   index or one at/past the axis size. Indices are validated before
    ///   any element is read.
    ///
    /// # Gradient
    ///
    /// Slices flow straight back to the positions they came from; repeated
    /// indices sum. `indices` is integral and takes no gradient.
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let dev = Device::Cpu;
    /// let table = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], &dev)?;
    /// let ids = Tensor::index_vec(&[2, 0], &dev)?;
    /// let rows = table.index_select(0, &ids)?;
    /// assert_eq!(rows.dims(), &[2, 2]);
    /// assert_eq!(rows.to_vec::<f32>()?, vec![5.0, 6.0, 1.0, 2.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn index_select(&self, axis: isize, indices: &Tensor) -> Result<Tensor> {
        const OP: &str = "index_select";
        let axis = self.shape().resolve_axis(axis, OP)?;
        check_index_operand(OP, self, indices)?;
        if indices.rank() != 1 {
            return Err(Error::RankMismatch {
                op: OP,
                expected: 1,
                got: indices.rank(),
            });
        }

        let mut out_dims = self.dims().to_vec();
        out_dims[axis] = indices.num_elements();
        let storage =
            dispatch::backend(self.device()).index_select(self.view(), axis, indices.view())?;
        let out = Tensor::from_parts(storage, Layout::contiguous(out_dims)?);

        let shape = self.shape().clone();
        let dtype = self.dtype();
        let device = self.device();
        // The index set is a constant of the backward: capture it detached so
        // the closure never holds a graph.
        let idx = indices.detach();
        let backward: BackwardFn = Box::new(move |g| {
            let grad = index_add_into_zeros(&shape, dtype, device, axis, &idx, g)?;
            Ok(vec![Some(grad)])
        });
        Ok(autograd::record(OP, out, &[self], backward))
    }

    /// Gather individual elements along `axis` through a **same-rank**
    /// [`I64`](crate::DType::I64) index grid (`PyTorch` `gather`).
    ///
    /// `out[c0, .., c_axis, ..] = self[c0, .., indices[c0, .., c_axis, ..], ..]`:
    /// every coordinate except `axis` is taken from the output position, and
    /// the `axis` coordinate comes from the index grid. The result has
    /// `indices`' shape, is contiguous, and keeps this tensor's dtype and
    /// device. Picking one logit per row — `logits.gather(1, &targets)` on
    /// `[n, classes]` with an `[n, 1]` index — is the canonical use.
    ///
    /// `axis` is an `isize` with negative indexing.
    ///
    /// # Errors
    ///
    /// - [`InvalidAxis`](crate::Error::InvalidAxis),
    ///   [`DTypeMismatch`](crate::Error::DTypeMismatch),
    ///   [`DeviceMismatch`](crate::Error::DeviceMismatch) as for
    ///   [`index_select`](Tensor::index_select).
    /// - [`RankMismatch`](crate::Error::RankMismatch) if `indices` does not
    ///   have this tensor's rank.
    /// - [`ShapeMismatch`](crate::Error::ShapeMismatch) if the index grid is
    ///   larger than the source on any axis other than `axis` (it may be
    ///   smaller, as `PyTorch` allows).
    /// - [`IndexOutOfBounds`](crate::Error::IndexOutOfBounds) for a negative
    ///   index or one at/past `dims()[axis]`.
    ///
    /// # Gradient
    ///
    /// Each gathered element's cotangent is scattered back to the position it
    /// was read from; elements read more than once accumulate.
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let dev = Device::Cpu;
    /// let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &dev)?;
    /// // One target class per row, as an [n, 1] index grid.
    /// let targets = Tensor::from_vec(vec![2i64, 0], [2, 1], &dev)?;
    /// let picked = logits.gather(1, &targets)?;
    /// assert_eq!(picked.dims(), &[2, 1]);
    /// assert_eq!(picked.to_vec::<f32>()?, vec![3.0, 4.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn gather(&self, axis: isize, indices: &Tensor) -> Result<Tensor> {
        const OP: &str = "gather";
        let axis = self.shape().resolve_axis(axis, OP)?;
        check_index_operand(OP, self, indices)?;
        if indices.rank() != self.rank() {
            return Err(Error::RankMismatch {
                op: OP,
                expected: self.rank(),
                got: indices.rank(),
            });
        }
        for (a, (&i, &s)) in indices.dims().iter().zip(self.dims()).enumerate() {
            if a != axis && i > s {
                return Err(Error::ShapeMismatch {
                    op: OP,
                    lhs: self.shape().clone(),
                    rhs: indices.shape().clone(),
                });
            }
        }

        let storage = dispatch::backend(self.device()).gather(self.view(), axis, indices.view())?;
        let out = Tensor::from_parts(storage, Layout::contiguous(indices.shape().clone())?);

        let shape = self.shape().clone();
        let dtype = self.dtype();
        let device = self.device();
        let idx = indices.detach();
        let backward: BackwardFn = Box::new(move |g| {
            let grad = scatter_add_into_zeros(&shape, dtype, device, axis, &idx, g)?;
            Ok(vec![Some(grad)])
        });
        Ok(autograd::record(OP, out, &[self], backward))
    }

    // ---- index and mask builders ----------------------------------------

    /// The 1-D [`I64`](crate::DType::I64) index vector `[0, 1, ..., len-1]`
    /// on `device` — `arange` specialized to the indexing dtype.
    ///
    /// This is the "select everything, in order" index and the building
    /// block of [`causal_mask`](Tensor::causal_mask) (positions compared
    /// against positions).
    ///
    /// # Errors
    ///
    /// [`InvalidArg`](crate::Error::InvalidArg) if `len` cannot be
    /// represented exactly as the `f64` bound `arange` counts with (only
    /// reachable above 2⁵³, far past any allocatable index vector).
    ///
    /// ```
    /// # use rstorch::{DType, Device, Tensor};
    /// let r = Tensor::index_range(4, &Device::Cpu)?;
    /// assert_eq!(r.dtype(), DType::I64);
    /// assert_eq!(r.to_vec::<i64>()?, vec![0, 1, 2, 3]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn index_range(len: usize, device: &Device) -> Result<Tensor> {
        let end = len as f64;
        if end as usize != len {
            return Err(Error::InvalidArg {
                op: "index_range",
                msg: format!("length {len} is not exactly representable as a range bound"),
            });
        }
        Tensor::arange(0.0, end, 1.0, DType::I64, device)
    }

    /// A 1-D [`I64`](crate::DType::I64) index tensor from host `positions` —
    /// the bridge from a `&[usize]` of row numbers (a data loader's batch, a
    /// permutation) to a device-side index for
    /// [`index_select`](Tensor::index_select).
    ///
    /// # Errors
    ///
    /// [`InvalidArg`](crate::Error::InvalidArg) if a position exceeds
    /// [`i64::MAX`], which no addressable axis can reach.
    ///
    /// ```
    /// # use rstorch::{DType, Device, Tensor};
    /// let ids = Tensor::index_vec(&[3, 0, 3], &Device::Cpu)?;
    /// assert_eq!(ids.dims(), &[3]);
    /// assert_eq!(ids.dtype(), DType::I64);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn index_vec(positions: &[usize], device: &Device) -> Result<Tensor> {
        let values = positions
            .iter()
            .map(|&p| {
                i64::try_from(p).map_err(|_| Error::InvalidArg {
                    op: "index_vec",
                    msg: format!("position {p} does not fit in i64"),
                })
            })
            .collect::<Result<Vec<i64>>>()?;
        Tensor::from_vec(values, [positions.len()], device)
    }

    /// The `[t, t]` causal (autoregressive) attention mask on `device`:
    /// [`Bool`](crate::DType::Bool), **`true` where a position must be
    /// masked out** — i.e. `mask[q][k] == (k > q)`, key `k` strictly in
    /// query `q`'s future.
    ///
    /// The `true`-means-blocked polarity is chosen so the mask drops straight
    /// into `masked_fill(&mask, f64::NEG_INFINITY)` before the softmax, which
    /// is the one place it is used. The mask is built on-device by comparing
    /// [`index_range`](Tensor::index_range) against itself (a broadcast
    /// `Gt`), never by a host loop: no round-trip, and the cost is one `[t]`
    /// index vector plus the `[t, t]` result.
    ///
    /// Masks are plain tensors, so combining a causal mask with a padding
    /// mask is ordinary boolean tensor arithmetic in the attention layer.
    ///
    /// ```
    /// # use rstorch::{DType, Device, Tensor};
    /// let m = Tensor::causal_mask(3, &Device::Cpu)?;
    /// assert_eq!(m.dims(), &[3, 3]);
    /// assert_eq!(m.dtype(), DType::Bool);
    /// // Row q may attend to keys 0..=q; everything past the diagonal is masked.
    /// assert_eq!(
    ///     m.to_vec::<bool>()?,
    ///     vec![false, true, true, false, false, true, false, false, false]
    /// );
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`index_range`](Self::index_range): [`Error::InvalidArg`] if `t`
    /// is not exactly representable as an `f64` range bound.
    pub fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
        let positions = Tensor::index_range(t, device)?;
        let square = Shape::from([t, t]);
        // Query index down the rows, key index across the columns: two
        // zero-copy stride-0 views of the same `[t]` buffer.
        let queries = positions.layout().unsqueeze(1)?.broadcast_to(&square)?;
        let keys = positions.layout().unsqueeze(0)?.broadcast_to(&square)?;
        let storage = dispatch::backend(*device).compare(
            CmpOp::Gt,
            View::new(positions.storage(), &keys),
            View::new(positions.storage(), &queries),
        )?;
        Ok(Tensor::from_parts(storage, Layout::contiguous(square)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: Device = Device::Cpu;

    fn t_f32(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn ids(values: &[i64], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape, &CPU).unwrap()
    }

    // ------------------------------------------------------------------
    // index_select
    // ------------------------------------------------------------------

    #[test]
    fn index_select_is_the_embedding_lookup() {
        // A [4, 3] embedding table, three ids (one repeated).
        let table = t_f32(
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
            [4, 3],
        );
        let out = table.index_select(0, &ids(&[3, 0, 3], [3])).unwrap();
        assert_eq!(out.dims(), &[3, 3]);
        assert!(out.is_contiguous());
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.device(), CPU);
        assert_eq!(
            out.to_vec::<f32>().unwrap(),
            vec![3.0, 3.1, 3.2, 0.0, 0.1, 0.2, 3.0, 3.1, 3.2]
        );
    }

    #[test]
    fn index_select_handles_inner_and_negative_axes() {
        // [[1,2,3],[4,5,6]]
        let x = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let picked = x.index_select(1, &ids(&[2, 0], [2])).unwrap();
        assert_eq!(picked.dims(), &[2, 2]);
        assert_eq!(picked.to_vec::<f32>().unwrap(), vec![3.0, 1.0, 6.0, 4.0]);

        // -1 is the last axis: the same selection.
        let same = x.index_select(-1, &ids(&[2, 0], [2])).unwrap();
        assert_eq!(
            same.to_vec::<f32>().unwrap(),
            picked.to_vec::<f32>().unwrap()
        );

        // -2 is axis 0.
        let rows = x.index_select(-2, &ids(&[1], [1])).unwrap();
        assert_eq!(rows.dims(), &[1, 3]);
        assert_eq!(rows.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn index_select_accepts_an_empty_index_set() {
        let x = t_f32(&[1.0, 2.0, 3.0], [3]);
        let out = x.index_select(0, &ids(&[], [0])).unwrap();
        assert_eq!(out.dims(), &[0]);
        assert!(out.to_vec::<f32>().unwrap().is_empty());
    }

    #[test]
    fn index_select_index_range_is_the_identity_selection() {
        let x = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
        let all = Tensor::index_range(3, &CPU).unwrap();
        let out = x.index_select(0, &all).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
        assert_eq!(out.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
    }

    #[test]
    fn index_select_rejects_bad_indices_and_operands() {
        let x = t_f32(&[1.0, 2.0, 3.0], [3]);

        assert!(matches!(
            x.index_select(0, &ids(&[3], [1])),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: 3,
                axis: 0,
                size: 3
            })
        ));
        // Negative indices do not wrap.
        assert!(matches!(
            x.index_select(0, &ids(&[-1], [1])),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: -1,
                ..
            })
        ));
        // Index tensors are I64, never silently cast.
        assert!(matches!(
            x.index_select(0, &t_f32(&[0.0], [1])),
            Err(Error::DTypeMismatch {
                op: "index_select",
                expected: DType::I64,
                got: DType::F32
            })
        ));
        // A same-rank grid is `gather`'s job, not `index_select`'s.
        assert!(matches!(
            x.index_select(0, &ids(&[0, 1], [1, 2])),
            Err(Error::RankMismatch {
                op: "index_select",
                expected: 1,
                got: 2
            })
        ));
        assert!(matches!(
            x.index_select(3, &ids(&[0], [1])),
            Err(Error::InvalidAxis {
                op: "index_select",
                axis: 3,
                rank: 1
            })
        ));
        assert!(matches!(
            x.index_select(-2, &ids(&[0], [1])),
            Err(Error::InvalidAxis {
                op: "index_select",
                axis: -2,
                rank: 1
            })
        ));
        // Indexing a scalar has no axis at all.
        assert!(matches!(
            t_f32(&[1.0], ()).index_select(0, &ids(&[0], [1])),
            Err(Error::InvalidAxis { rank: 0, .. })
        ));
    }

    // ------------------------------------------------------------------
    // gather
    // ------------------------------------------------------------------

    #[test]
    fn gather_picks_one_element_per_row() {
        // [[1,2,3],[4,5,6]] with per-row targets [2, 0].
        let logits = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let picked = logits.gather(1, &ids(&[2, 0], [2, 1])).unwrap();
        assert_eq!(picked.dims(), &[2, 1]);
        assert!(picked.is_contiguous());
        assert_eq!(picked.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);

        // A full-width grid keeps the source shape.
        let all = logits
            .gather(-1, &ids(&[0, 1, 2, 2, 1, 0], [2, 3]))
            .unwrap();
        assert_eq!(all.dims(), &[2, 3]);
        assert_eq!(
            all.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 6.0, 5.0, 4.0]
        );
    }

    #[test]
    fn gather_along_the_leading_axis() {
        let x = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let out = x.gather(0, &ids(&[1, 0, 1], [1, 3])).unwrap();
        assert_eq!(out.dims(), &[1, 3]);
        assert_eq!(out.to_vec::<f32>().unwrap(), vec![4.0, 2.0, 6.0]);
    }

    #[test]
    fn gather_rejects_bad_indices_and_shapes() {
        let x = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        assert!(matches!(
            x.gather(1, &ids(&[0, 3], [2, 1])),
            Err(Error::IndexOutOfBounds {
                op: "gather",
                index: 3,
                axis: 1,
                size: 3
            })
        ));
        assert!(matches!(
            x.gather(1, &ids(&[-1, 0], [2, 1])),
            Err(Error::IndexOutOfBounds {
                op: "gather",
                index: -1,
                ..
            })
        ));
        // Same-rank index grid required.
        assert!(matches!(
            x.gather(1, &ids(&[0, 0], [2])),
            Err(Error::RankMismatch {
                op: "gather",
                expected: 2,
                got: 1
            })
        ));
        // Larger than the source on a non-gathered axis.
        assert!(matches!(
            x.gather(1, &ids(&[0; 9], [3, 3])),
            Err(Error::ShapeMismatch { op: "gather", .. })
        ));
        assert!(matches!(
            x.gather(1, &t_f32(&[0.0, 0.0], [2, 1])),
            Err(Error::DTypeMismatch { op: "gather", .. })
        ));
        assert!(matches!(
            x.gather(2, &ids(&[0, 0], [2, 1])),
            Err(Error::InvalidAxis { op: "gather", .. })
        ));
    }

    // ------------------------------------------------------------------
    // Index and mask builders
    // ------------------------------------------------------------------

    #[test]
    fn index_range_and_index_vec_build_i64_vectors() {
        let r = Tensor::index_range(5, &CPU).unwrap();
        assert_eq!(r.dims(), &[5]);
        assert_eq!(r.dtype(), DType::I64);
        assert_eq!(r.to_vec::<i64>().unwrap(), vec![0, 1, 2, 3, 4]);

        // Zero length is an empty index vector, not an error.
        let r = Tensor::index_range(0, &CPU).unwrap();
        assert_eq!(r.dims(), &[0]);
        assert!(r.to_vec::<i64>().unwrap().is_empty());

        let v = Tensor::index_vec(&[7, 0, 7], &CPU).unwrap();
        assert_eq!(v.dims(), &[3]);
        assert_eq!(v.dtype(), DType::I64);
        assert_eq!(v.to_vec::<i64>().unwrap(), vec![7, 0, 7]);

        let empty = Tensor::index_vec(&[], &CPU).unwrap();
        assert_eq!(empty.dims(), &[0]);
    }

    #[test]
    fn causal_mask_blocks_exactly_the_future() {
        let m = Tensor::causal_mask(4, &CPU).unwrap();
        assert_eq!(m.dims(), &[4, 4]);
        assert_eq!(m.dtype(), DType::Bool);
        assert_eq!(m.device(), CPU);
        assert!(m.is_contiguous());

        let values = m.to_vec::<bool>().unwrap();
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(values[q * 4 + k], k > q, "mask[{q}][{k}]");
            }
        }
    }

    #[test]
    fn causal_mask_degenerate_sizes() {
        let m = Tensor::causal_mask(1, &CPU).unwrap();
        assert_eq!(m.dims(), &[1, 1]);
        assert_eq!(m.to_vec::<bool>().unwrap(), vec![false]);

        let m = Tensor::causal_mask(0, &CPU).unwrap();
        assert_eq!(m.dims(), &[0, 0]);
        assert!(m.to_vec::<bool>().unwrap().is_empty());
    }

    // ------------------------------------------------------------------
    // Backward — finite differences.
    //
    // `check_grad` requires a scalar, obtained here with a *weighted*
    // `sum_all`. A one-hot
    // selection does catch a scatter that overwrites (the dropped contribution
    // is missing from the total), but it reaches one gathered element out of
    // however many the index names, and an *unweighted* sum hands every
    // gathered element the same cotangent — neither can distinguish a
    // scatter-add landing in the right slot from one landing in the wrong one.
    // Distinct weights can, and the accumulation case below then pins the sum
    // of two *different* contributions rather than the presence of one.
    // ------------------------------------------------------------------

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-4;

    /// `Σ w ⊙ x` with pairwise-distinct constant weights: a scalar objective
    /// whose cotangent on `x` is `w` rather than a constant.
    fn wsum(x: &Tensor) -> Result<Tensor> {
        let w: Vec<f32> = (0..x.num_elements())
            .map(|i| 0.25 + 0.5 * (i as f32))
            .collect();
        x.mul(&Tensor::from_vec(w, x.dims().to_vec(), &CPU)?)?
            .sum_all()
    }

    #[test]
    fn index_select_backward_matches_finite_differences() {
        // Out of order, and row 1 is never selected — its gradient is checked
        // against zero, not skipped.
        let x = t_f32(&[1.0, -2.0, 3.0, 0.5, -1.5, 2.5], [3, 2]);
        let idx = ids(&[2, 0], [2]);
        crate::testing::check_grad(
            |inputs| wsum(&inputs[0].index_select(0, &idx)?),
            &[x],
            EPS,
            TOL,
        )
        .unwrap();

        // Selecting along a trailing axis, where the gathered stride is 1.
        let x = t_f32(&[1.0, -2.0, 3.0, 0.5, -1.5, 2.5], [2, 3]);
        let idx = ids(&[2, 0, 2], [3]);
        crate::testing::check_grad(
            |inputs| wsum(&inputs[0].index_select(1, &idx)?),
            &[x],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn index_select_backward_accumulates_repeated_indices() {
        // Row 1 twice, under *different* weights (0.25 and 0.75): its
        // gradient is their sum, 1.0. A backward that overwrote instead of
        // accumulating would report 0.25 or 0.75 — both wrong, and both
        // invisible to an unweighted objective.
        let x = t_f32(&[1.0, -2.0, 3.0], [3]);
        let idx = ids(&[1, 1, 2], [3]);
        crate::testing::check_grad(
            |inputs| wsum(&inputs[0].index_select(0, &idx)?),
            std::slice::from_ref(&x),
            EPS,
            TOL,
        )
        .unwrap();

        // The analytic gradient itself, so the accumulation is pinned as a
        // value and not only as an agreement with finite differences.
        let traced = x.traced().unwrap();
        let grads = wsum(&traced.index_select(0, &idx).unwrap())
            .unwrap()
            .backward()
            .unwrap();
        let g = grads.wrt_input(&traced).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(g, vec![0.0, 1.0, 1.25]);
    }

    #[test]
    fn gather_backward_matches_finite_differences() {
        // Repeated column indices inside a row: the scatter-add has to sum
        // two cotangents into the same source element.
        let x = t_f32(&[1.0, -2.0, 3.0, 0.5, -1.5, 2.5], [2, 3]);
        let idx = ids(&[2, 0, 2, 1, 1, 1], [2, 3]);
        crate::testing::check_grad(|inputs| wsum(&inputs[0].gather(1, &idx)?), &[x], EPS, TOL)
            .unwrap();

        // …and along the leading axis, with a narrower index than the source.
        let x = t_f32(&[1.0, -2.0, 3.0, 0.5, -1.5, 2.5], [3, 2]);
        let idx = ids(&[2, 0, 2, 2], [2, 2]);
        crate::testing::check_grad(|inputs| wsum(&inputs[0].gather(0, &idx)?), &[x], EPS, TOL)
            .unwrap();
    }
}
