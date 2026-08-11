//! [`Embedding`] — a learned lookup table indexed by `I64` token ids.
//!
//! The forward pass is one [`index_select`](crate::Tensor::index_select) on
//! the `[num_embeddings, embedding_dim]` table, so the gradient is the
//! `index_add` that op already owns: every id's row receives the cotangent of
//! the position it was read into, and a repeated id **accumulates** (the whole
//! reason the backward scatters instead of assigning).
//!
//! Ids are ordinary on-device [`I64`](crate::DType::I64) tensors of any rank
//! (`Bool` and `I64` are ordinary dtypes) — a `[batch, seq]`
//! id batch yields `[batch, seq, embedding_dim]` without the caller ever
//! flattening or naming a rank.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Forward, Mode, Param};
use crate::rng::Rng;
use crate::tensor::Tensor;

/// A learned embedding table: `[num_embeddings, embedding_dim]` rows selected
/// by token id.
///
/// The single parameter is named `weight`, so its `state_dict` path is
/// `weight` under this module's own prefix (`tok_emb.weight` for a field named
/// `tok_emb`). That name is the persistence contract.
///
/// Positional embeddings need no separate type: an `Embedding` over
/// `max_seq_len` rows looked up with
/// [`Tensor::index_range`](crate::Tensor::index_range) *is* a learned
/// positional table.
///
/// ```
/// # use rstorch::nn::{Embedding, Mode};
/// # use rstorch::{Device, Rng, Tensor};
/// # fn main() -> rstorch::Result<()> {
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(7);
/// let emb = Embedding::new(100, 8, &dev, &mut rng)?;
///
/// // A [2, 3] batch of ids becomes a [2, 3, 8] batch of vectors.
/// let ids = Tensor::from_vec(vec![5i64, 0, 99, 1, 1, 42], [2, 3], &dev)?;
/// assert_eq!(emb.lookup(&ids, Mode::EVAL)?.dims(), &[2, 3, 8]);
///
/// // Learned positional embeddings: the same table, looked up by position.
/// let positions = Tensor::index_range(3, &dev)?;
/// assert_eq!(emb.lookup(&positions, Mode::EVAL)?.dims(), &[3, 8]);
/// # Ok(())
/// # }
/// ```
#[derive(rstorch::Module)]
pub struct Embedding {
    /// The table. Shape `[num_embeddings, embedding_dim]`, and fixed for the
    /// module's lifetime ([`Param::set`] preserves shape).
    weight: Param,
}

impl Embedding {
    /// A fresh table of `num_embeddings` rows of `embedding_dim`
    /// [`F32`](crate::DType::F32) values on `device`, initialized from the
    /// standard normal (PyTorch's `nn.Embedding` default) using `rng`.
    ///
    /// Tables are created in `F32` and converted afterwards
    /// ([`nn::to_dtype`](crate::nn::to_dtype)), like every other layer.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Embedding::new"`) if either dimension is
    /// zero: a table with no rows can never be indexed and a row with no
    /// values carries no information, so both are caller bugs rather than
    /// degenerate-but-valid shapes.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Embedding> {
        const OP: &str = "Embedding::new";
        if num_embeddings == 0 || embedding_dim == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "num_embeddings and embedding_dim must both be non-zero, \
                     got {num_embeddings} and {embedding_dim}"
                ),
            });
        }
        let weight = Tensor::randn([num_embeddings, embedding_dim], DType::F32, device, rng)?;
        Ok(Embedding {
            weight: Param::new(weight),
        })
    }

    /// Wrap an existing `[num_embeddings, embedding_dim]` table (a pretrained
    /// matrix, or one built by hand in a test) as a fresh parameter.
    ///
    /// The tensor is used as given — dtype and device come from it — and any
    /// graph it carries is dropped: a parameter is a leaf, not an interior
    /// value.
    ///
    /// # Errors
    ///
    /// - [`Error::RankMismatch`] (`op: "Embedding::from_weight"`) unless
    ///   `weight` is rank 2.
    /// - [`Error::InvalidArg`] if it is not a float tensor (an integer table
    ///   could never receive a gradient) or if either dimension is zero.
    pub fn from_weight(weight: Tensor) -> Result<Embedding> {
        const OP: &str = "Embedding::from_weight";
        let (rows, cols) = weight.dims2().map_err(|_| Error::RankMismatch {
            op: OP,
            expected: 2,
            got: weight.rank(),
        })?;
        if !weight.dtype().is_float() {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "an embedding table must be a float tensor, got {}",
                    weight.dtype()
                ),
            });
        }
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "an embedding table must have no zero dimension, got [{rows}, {cols}]"
                ),
            });
        }
        Ok(Embedding {
            weight: Param::new(weight.detach()),
        })
    }

    /// Number of rows in the table (the vocabulary size).
    pub fn num_embeddings(&self) -> usize {
        self.weight.value().dims()[0]
    }

    /// Width of one embedding vector.
    pub fn embedding_dim(&self) -> usize {
        self.weight.value().dims()[1]
    }

    /// The table parameter — the handle for weight tying (the parent module
    /// reads it for the output head) and for
    /// [`Grads::wrt`](crate::Grads::wrt).
    pub fn weight(&self) -> &Param {
        &self.weight
    }

    /// Look up `ids`, appending the embedding axis: ids of shape `[d0, .., dn]`
    /// give `[d0, .., dn, embedding_dim]`.
    ///
    /// `ids` may have any rank, including 0 (a single id yields a
    /// `[embedding_dim]` vector). This is the `&self` sibling of
    /// [`Forward::forward`] — an embedding has no mutable state, so it is
    /// usable from a shared reference (and inside a `Fn` closure such as a
    /// finite-difference gradient check's objective).
    ///
    /// `mode`'s recording axis decides whether the lookup is traced; its
    /// behavior axis is irrelevant here (a table behaves identically in train
    /// and eval).
    ///
    /// # Errors
    ///
    /// - [`Error::DTypeMismatch`] (`op: "Embedding::lookup"`) unless `ids` is
    ///   [`I64`](crate::DType::I64), and [`Error::DeviceMismatch`] if it lives
    ///   on another device than the table.
    /// - [`Error::IndexOutOfBounds`] for a negative id or one at/past
    ///   [`num_embeddings`](Embedding::num_embeddings) — ids are validated
    ///   before any row is read.
    pub fn lookup(&self, ids: &Tensor, mode: Mode) -> Result<Tensor> {
        const OP: &str = "Embedding::lookup";
        if ids.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                op: OP,
                expected: DType::I64,
                got: ids.dtype(),
            });
        }
        let table = self.weight.get(mode);
        if ids.device() != table.device() {
            return Err(Error::DeviceMismatch {
                op: OP,
                expected: table.device(),
                got: ids.device(),
            });
        }
        // One flat select, then restore the id shape with the embedding axis
        // appended. `reshape` is a view whenever the ids are contiguous.
        let flat = ids.reshape([ids.num_elements()])?;
        let rows = table.index_select(0, &flat)?;
        let mut dims = ids.dims().to_vec();
        dims.push(self.embedding_dim());
        rows.reshape(dims)
    }
}

impl Forward for Embedding {
    /// [`lookup`](Embedding::lookup) — `x` is the [`I64`](crate::DType::I64)
    /// id tensor, so an `Embedding` is the legal first layer of a
    /// [`Sequential`](crate::nn::Sequential).
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        self.lookup(x, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::nn;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    /// A `[rows, cols]` table whose row `r` is `[r.0, r.1, r.2, ..]` — every
    /// element identifies the row and column it came from.
    fn table(rows: usize, cols: usize) -> Tensor {
        let values: Vec<f32> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| r as f32 + c as f32 / 10.0))
            .collect();
        Tensor::from_vec(values, [rows, cols], &CPU).unwrap()
    }

    fn ids(values: &[i64], shape: impl Into<crate::shape::Shape>) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape, &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    // ------------------------------------------------------------------
    // Shapes and values
    // ------------------------------------------------------------------

    #[test]
    fn lookup_appends_the_embedding_axis_at_every_id_rank() {
        let emb = Embedding::from_weight(table(4, 3)).unwrap();
        assert_eq!(emb.num_embeddings(), 4);
        assert_eq!(emb.embedding_dim(), 3);

        // [batch, seq] -> [batch, seq, dim]
        let batched = emb
            .lookup(&ids(&[3, 0, 1, 1, 2, 3], [2, 3]), Mode::EVAL)
            .unwrap();
        assert_eq!(batched.dims(), &[2, 3, 3]);
        assert_eq!(&v(&batched)[0..3], &[3.0, 3.1, 3.2]);
        assert_eq!(&v(&batched)[3..6], &[0.0, 0.1, 0.2]);

        // [seq] -> [seq, dim]
        let flat = emb.lookup(&ids(&[2, 2], [2]), Mode::EVAL).unwrap();
        assert_eq!(flat.dims(), &[2, 3]);
        assert_eq!(v(&flat), vec![2.0, 2.1, 2.2, 2.0, 2.1, 2.2]);

        // rank-0 id -> [dim]
        let single = emb.lookup(&ids(&[1], ()), Mode::EVAL).unwrap();
        assert_eq!(single.dims(), &[3]);
        assert_eq!(v(&single), vec![1.0, 1.1, 1.2]);

        // rank-3 ids -> [.., dim]
        let deep = emb
            .lookup(&ids(&[0, 1, 2, 3], [2, 1, 2]), Mode::EVAL)
            .unwrap();
        assert_eq!(deep.dims(), &[2, 1, 2, 3]);
    }

    #[test]
    fn lookup_of_non_contiguous_ids_reads_them_in_logical_order() {
        let emb = Embedding::from_weight(table(4, 2)).unwrap();
        // A transposed (non-contiguous) id view: logical order is 0, 2, 1, 3.
        let grid = ids(&[0, 1, 2, 3], [2, 2]).transpose(0, 1).unwrap();
        assert!(!grid.is_contiguous());
        let out = emb.lookup(&grid, Mode::EVAL).unwrap();
        assert_eq!(out.dims(), &[2, 2, 2]);
        assert_eq!(v(&out), vec![0.0, 0.1, 2.0, 2.1, 1.0, 1.1, 3.0, 3.1]);
    }

    #[test]
    fn forward_is_lookup_so_embedding_is_a_layer() {
        let mut emb = Embedding::from_weight(table(4, 2)).unwrap();
        let batch = ids(&[1, 2], [1, 2]);
        let direct = emb.lookup(&batch, Mode::EVAL).unwrap();
        let through_trait = emb.forward(&batch, Mode::EVAL).unwrap();
        assert_eq!(v(&direct), v(&through_trait));
    }

    // ------------------------------------------------------------------
    // Module wiring
    // ------------------------------------------------------------------

    #[test]
    fn the_table_is_the_only_parameter_and_is_named_weight() {
        let emb = Embedding::from_weight(table(5, 4)).unwrap();
        assert_eq!(nn::state_dict(&emb).keys().collect::<Vec<_>>(), ["weight"]);
        assert_eq!(nn::num_params(&emb), 20);
    }

    #[test]
    fn new_initializes_f32_on_the_requested_device_from_the_rng() {
        let mut rng = Rng::seed(11);
        let emb = Embedding::new(6, 3, &CPU, &mut rng).unwrap();
        assert_eq!(emb.weight().value().dims(), &[6, 3]);
        assert_eq!(emb.weight().value().dtype(), DType::F32);
        assert_eq!(emb.weight().value().device(), CPU);

        // Same seed, same table; a different seed, a different one.
        let same = Embedding::new(6, 3, &CPU, &mut Rng::seed(11)).unwrap();
        assert_eq!(v(emb.weight().value()), v(same.weight().value()));
        let other = Embedding::new(6, 3, &CPU, &mut Rng::seed(12)).unwrap();
        assert_ne!(v(emb.weight().value()), v(other.weight().value()));
    }

    // ------------------------------------------------------------------
    // Gradients
    // ------------------------------------------------------------------

    /// The objective the gradient tests differentiate: a fixed non-uniform
    /// weighting of the looked-up rows, so no row's contribution cancels.
    fn coefficients(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.25).collect();
        Tensor::from_vec(values, shape.to_vec(), &CPU).unwrap()
    }

    #[test]
    fn gradient_accumulates_repeated_ids_and_matches_finite_differences() {
        // Row 1 is used three times, row 3 once, rows 0 and 2 never.
        let id_batch = ids(&[1, 3, 1, 1], [2, 2]);
        let coef = coefficients(&[2, 2, 3]);

        let emb = Embedding::from_weight(table(4, 3)).unwrap();
        let out = emb.lookup(&id_batch, Mode::TRAIN).unwrap();
        let loss = out.mul(&coef).unwrap().sum_all().unwrap();
        let analytic = loss.backward().unwrap().wrt(emb.weight()).unwrap();
        assert_eq!(analytic.dims(), &[4, 3]);

        // Hand-rolled expectation: row r accumulates the coefficient blocks of
        // every position that selected it; unselected rows stay zero.
        let coef_values = v(&coef);
        let mut expected = vec![0.0f32; 4 * 3];
        for (position, id) in id_batch.to_vec::<i64>().unwrap().iter().enumerate() {
            for c in 0..3 {
                expected[*id as usize * 3 + c] += coef_values[position * 3 + c];
            }
        }
        assert_eq!(v(&analytic), expected);

        // …and the same objective, with the table as a *traced input*, agrees
        // with central finite differences.
        let f = |xs: &[Tensor]| {
            let rows = xs[0].index_select(0, &id_batch.reshape([4]).unwrap())?;
            rows.reshape([2, 2, 3])?.mul(&coef)?.sum_all()
        };
        check_grad(f, &[table(4, 3)], 1e-2, 1e-3).unwrap();
    }

    #[test]
    fn lookup_records_only_under_a_recording_mode() {
        let emb = Embedding::from_weight(table(4, 2)).unwrap();
        let id_batch = ids(&[0, 2], [2]);

        let traced = emb.lookup(&id_batch, Mode::TRAIN).unwrap();
        assert!(traced.sum_all().unwrap().backward().is_ok());

        let plain = emb.lookup(&id_batch, Mode::EVAL).unwrap();
        assert!(matches!(
            plain.sum_all().unwrap().backward(),
            Err(Error::NotTraced { .. })
        ));
        // Eval and train agree on the values.
        assert_eq!(v(&traced), v(&plain));
    }

    #[test]
    fn a_frozen_table_produces_no_gradient() {
        let mut emb = Embedding::from_weight(table(4, 2)).unwrap();
        emb.weight.freeze();
        let out = emb.lookup(&ids(&[0, 2], [2]), Mode::TRAIN).unwrap();
        assert!(matches!(
            out.sum_all().unwrap().backward(),
            Err(Error::NotTraced { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Loud errors
    // ------------------------------------------------------------------

    #[test]
    fn float_ids_are_a_dtype_error_not_a_silent_truncation() {
        let emb = Embedding::from_weight(table(4, 2)).unwrap();
        let float_ids = Tensor::from_vec(vec![1.0f32, 2.0], [2], &CPU).unwrap();
        assert!(matches!(
            emb.lookup(&float_ids, Mode::EVAL),
            Err(Error::DTypeMismatch {
                op: "Embedding::lookup",
                expected: DType::I64,
                got: DType::F32
            })
        ));
    }

    #[test]
    fn an_id_outside_the_table_is_loud() {
        let emb = Embedding::from_weight(table(4, 2)).unwrap();
        assert!(matches!(
            emb.lookup(&ids(&[4], [1]), Mode::EVAL),
            Err(Error::IndexOutOfBounds { index: 4, .. })
        ));
        assert!(matches!(
            emb.lookup(&ids(&[-1], [1]), Mode::EVAL),
            Err(Error::IndexOutOfBounds { index: -1, .. })
        ));
    }

    #[test]
    fn constructors_reject_degenerate_and_wrongly_shaped_tables() {
        let mut rng = Rng::seed(1);
        assert!(matches!(
            Embedding::new(0, 4, &CPU, &mut rng),
            Err(Error::InvalidArg {
                op: "Embedding::new",
                ..
            })
        ));
        assert!(matches!(
            Embedding::new(4, 0, &CPU, &mut rng),
            Err(Error::InvalidArg {
                op: "Embedding::new",
                ..
            })
        ));
        assert!(matches!(
            Embedding::from_weight(Tensor::zeros([4], DType::F32, &CPU).unwrap()),
            Err(Error::RankMismatch {
                op: "Embedding::from_weight",
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            Embedding::from_weight(Tensor::zeros([4, 2], DType::I64, &CPU).unwrap()),
            Err(Error::InvalidArg {
                op: "Embedding::from_weight",
                ..
            })
        ));
    }

    #[test]
    fn from_weight_drops_an_incoming_graph() {
        // A parameter is a leaf: the value's history is not the model's.
        let traced = table(4, 2).traced().unwrap();
        let emb = Embedding::from_weight(traced.mul_scalar(2.0).unwrap()).unwrap();
        assert!(emb.weight().value().backward().is_err());
        // Row 0 of the table is [0.0, 0.1], doubled.
        assert_eq!(&v(emb.weight().value())[0..2], &[0.0, 0.2]);
    }
}
