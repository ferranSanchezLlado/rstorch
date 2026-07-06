#![allow(clippy::type_complexity)]

use super::parameter::Parameter;
use crate::backend::{Backend, Cpu};
use crate::data::Batch;
use crate::dtype::FloatDType;
use crate::error::{Result, const_check};
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
