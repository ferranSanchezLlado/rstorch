#![allow(clippy::type_complexity)]

use super::parameter::Parameter;
use crate::backend::{Backend, Cpu};
use crate::data::Batch;
use crate::dtype::FloatDType;
use crate::error::{DataError, DeviceError, Result, const_check};
use crate::random::SmallRng;
use crate::shape::{AnyDim, C, D2, D3, Sym};
use crate::tensor::Tensor;

/// Learned token embedding table.
///
/// Parameter names are part of the persistence contract: `weight` has shape
/// `[VOCAB, DIM]`.
pub struct Embedding<const VOCAB: usize, const DIM: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D2<C<VOCAB>, C<DIM>>, E, B>,
}

impl<const VOCAB: usize, const DIM: usize, E, B> Embedding<VOCAB, DIM, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn from_weight(weight: Tensor<D2<C<VOCAB>, C<DIM>>, E, B>) -> Self {
        Self {
            weight: Parameter::new(weight),
        }
    }

    pub fn uniform(rng: &mut SmallRng, low: E, high: E) -> Result<Self> {
        const { const_check::mul_fits(VOCAB, DIM, "embedding_uniform", "VOCAB", "DIM") };

        let values = (0..VOCAB * DIM).map(|_| rng.uniform(low, high)).collect();
        Ok(Self::from_weight(Tensor::from_vec(values)?))
    }

    pub fn normal(rng: &mut SmallRng, std: E) -> Result<Self> {
        const { const_check::mul_fits(VOCAB, DIM, "embedding_normal", "VOCAB", "DIM") };

        let std_f64 = std.to_f64();
        let values = (0..VOCAB * DIM)
            .map(|_| E::from_f64(rng.normal::<f64>() * std_f64))
            .collect();
        Ok(Self::from_weight(Tensor::from_vec(values)?))
    }

    pub fn weight(&self) -> &Parameter<D2<C<VOCAB>, C<DIM>>, E, B> {
        &self.weight
    }

    pub fn forward<const SEQ: usize>(
        &self,
        ids: &[[usize; SEQ]],
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<DIM>>, E, B>> {
        let batch = ids.len();
        let flat: Vec<_> = ids.iter().flat_map(|row| row.iter().copied()).collect();
        self.weight
            .tensor()
            .index_select_rows::<AnyDim>(&flat)?
            .reshape_with_shape([batch, SEQ, DIM])
    }

    /// Tensor-id sibling of [`forward`](Self::forward).
    ///
    /// This is available only when the backend can store i64 tensors. The
    /// `&[[usize; SEQ]]` method remains the universal every-backend path.
    /// Negative ids are rejected rather than wrapped. i64 tensors are data
    /// containers only: no integer autograd, arithmetic, matmul, or pre-1.0 GPU
    /// i64 backend is provided.
    pub fn forward_ids<const SEQ: usize>(
        &self,
        ids: &Tensor<D2<Sym<Batch>, C<SEQ>>, i64, B>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<DIM>>, E, B>>
    where
        B: Backend<i64, Device = <B as Backend<E>>::Device>,
    {
        if self.weight.tensor().device() != ids.device() {
            return Err(DeviceError::Mismatch {
                op: "embedding_forward_ids",
                lhs: format!("{:?}", self.weight.tensor().device()),
                rhs: format!("{:?}", ids.device()),
            }
            .into());
        }
        let batch = ids.shape().dims()[0];
        let flat = ids
            .to_vec()?
            .into_iter()
            .map(|id| {
                if id < 0 {
                    return Err(DataError::NegativeIndex { index: id }.into());
                }
                usize::try_from(id).map_err(|_| {
                    DataError::IndexOutOfBounds {
                        index: usize::MAX,
                        len: VOCAB,
                    }
                    .into()
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.weight
            .tensor()
            .index_select_rows::<AnyDim>(&flat)?
            .reshape_with_shape([batch, SEQ, DIM])
    }
}

crate::nn::has_parameters! {
    impl[const VOCAB: usize, const DIM: usize, E, B] Embedding<VOCAB, DIM, E, B>
    where { }
    {
        params { weight }
        children { }
        transparent_children { }
    }
}

/// Learned positional embedding table.
///
/// Parameter names are part of the persistence contract: `weight` has shape
/// `[SEQ, DIM]`.
pub struct PositionalEmbedding<const SEQ: usize, const DIM: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D2<C<SEQ>, C<DIM>>, E, B>,
}

impl<const SEQ: usize, const DIM: usize, E, B> PositionalEmbedding<SEQ, DIM, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn uniform(rng: &mut SmallRng, low: E, high: E) -> Result<Self> {
        const { const_check::mul_fits(SEQ, DIM, "positional_embedding_uniform", "SEQ", "DIM") };

        let values = (0..SEQ * DIM).map(|_| rng.uniform(low, high)).collect();
        Ok(Self {
            weight: Parameter::new(Tensor::from_vec(values)?),
        })
    }

    pub fn normal(rng: &mut SmallRng, std: E) -> Result<Self> {
        const { const_check::mul_fits(SEQ, DIM, "positional_embedding_normal", "SEQ", "DIM") };

        let std_f64 = std.to_f64();
        let values = (0..SEQ * DIM)
            .map(|_| E::from_f64(rng.normal::<f64>() * std_f64))
            .collect();
        Ok(Self {
            weight: Parameter::new(Tensor::from_vec(values)?),
        })
    }

    pub fn forward(&self, batch: usize) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<DIM>>, E, B>> {
        let positions: Vec<_> = (0..batch).flat_map(|_| 0..SEQ).collect();
        self.weight
            .tensor()
            .index_select_rows::<AnyDim>(&positions)?
            .reshape_with_shape([batch, SEQ, DIM])
    }
}

crate::nn::has_parameters! {
    impl[const SEQ: usize, const DIM: usize, E, B] PositionalEmbedding<SEQ, DIM, E, B>
    where { }
    {
        params { weight }
        children { }
        transparent_children { }
    }
}
