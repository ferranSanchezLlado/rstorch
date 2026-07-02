use super::linear::Linear;
use super::parameter::{HasParameters, Module, ParameterRef, ParameterRefMut};
use crate::backend::{Backend, Cpu};
use crate::data::Batch;
use crate::dtype::FloatDType;
use crate::error::{Result, ShapeError, const_check};
use crate::random::SmallRng;
use crate::shape::{AnyDim, C, D2, D3, D4, DimSpec, Sym};
use crate::tensor::{Mask, Tensor};

pub fn causal_attention_mask<const SEQ: usize>(
    batch_heads: usize,
) -> Result<Mask<D3<AnyDim, C<SEQ>, C<SEQ>>>> {
    const { const_check::mul_fits(SEQ, SEQ, "causal_attention_mask", "SEQ", "SEQ") };

    let mut values = Vec::with_capacity(batch_heads * SEQ * SEQ);
    for _ in 0..batch_heads {
        for query in 0..SEQ {
            for key in 0..SEQ {
                values.push(key > query);
            }
        }
    }
    Mask::from_vec_with_shape(values, [batch_heads, SEQ, SEQ])
}

pub fn scaled_dot_product_attention<BatchDim, const SEQ: usize, const HEAD_DIM: usize, E, B>(
    q: &Tensor<D3<BatchDim, C<SEQ>, C<HEAD_DIM>>, E, B>,
    k: &Tensor<D3<BatchDim, C<SEQ>, C<HEAD_DIM>>, E, B>,
    v: &Tensor<D3<BatchDim, C<SEQ>, C<HEAD_DIM>>, E, B>,
    mask: Option<&Mask<D3<BatchDim, C<SEQ>, C<SEQ>>>>,
) -> Result<Tensor<D3<BatchDim, C<SEQ>, C<HEAD_DIM>>, E, B>>
where
    BatchDim: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    const { const_check::nonzero(HEAD_DIM, "scaled_dot_product_attention", "HEAD_DIM") };

    let scale = E::from_f64((HEAD_DIM as f64).sqrt());
    let mut scores = q.bmm(&k.transpose_last2()?)?.div_scalar(scale)?;
    if let Some(mask) = mask {
        scores = scores.masked_fill(mask, E::from_f64(-1.0e9))?;
    }
    scores.softmax_axis2()?.bmm(v)
}

pub struct MultiHeadAttention<
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    E = f32,
    B = Cpu,
> where
    E: FloatDType,
    B: Backend<E>,
{
    q_proj: Linear<EMBED, EMBED, E, B>,
    k_proj: Linear<EMBED, EMBED, E, B>,
    v_proj: Linear<EMBED, EMBED, E, B>,
    out_proj: Linear<EMBED, EMBED, E, B>,
}

impl<const SEQ: usize, const EMBED: usize, const HEADS: usize, const HEAD_DIM: usize, E, B>
    MultiHeadAttention<SEQ, EMBED, HEADS, HEAD_DIM, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn xavier_uniform(rng: &mut SmallRng) -> Result<Self> {
        ensure_head_shape::<EMBED, HEADS, HEAD_DIM>()?;

        Ok(Self {
            q_proj: Linear::xavier_uniform(rng)?,
            k_proj: Linear::xavier_uniform(rng)?,
            v_proj: Linear::xavier_uniform(rng)?,
            out_proj: Linear::xavier_uniform(rng)?,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>> {
        self.forward_causal(input)
    }

    pub fn forward_causal(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>> {
        let batch = input.shape().dims()[0];
        let mask = causal_attention_mask::<SEQ>(batch * HEADS)?;
        self.forward_with_mask(input, Some(&mask))
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
        mask: Option<&Mask<D3<AnyDim, C<SEQ>, C<SEQ>>>>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>> {
        ensure_head_shape::<EMBED, HEADS, HEAD_DIM>()?;
        let q = self.split_heads(&project_3d(&self.q_proj, input)?)?;
        let k = self.split_heads(&project_3d(&self.k_proj, input)?)?;
        let v = self.split_heads(&project_3d(&self.v_proj, input)?)?;
        let heads = scaled_dot_product_attention(&q, &k, &v, mask)?;
        let merged = self.merge_heads(input.shape().dims()[0], &heads)?;
        project_3d(&self.out_proj, &merged)
    }

    fn split_heads(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
    ) -> Result<Tensor<D3<AnyDim, C<SEQ>, C<HEAD_DIM>>, E, B>> {
        let batch = input.shape().dims()[0];
        input
            .reshape_with_shape::<D4<Sym<Batch>, C<SEQ>, C<HEADS>, C<HEAD_DIM>>>([
                batch, SEQ, HEADS, HEAD_DIM,
            ])?
            .transpose_axes12()?
            .reshape_with_shape([batch * HEADS, SEQ, HEAD_DIM])
    }

    fn merge_heads(
        &self,
        batch: usize,
        input: &Tensor<D3<AnyDim, C<SEQ>, C<HEAD_DIM>>, E, B>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>> {
        input
            .reshape_with_shape::<D4<Sym<Batch>, C<HEADS>, C<SEQ>, C<HEAD_DIM>>>([
                batch, HEADS, SEQ, HEAD_DIM,
            ])?
            .transpose_axes12()?
            .reshape_with_shape([batch, SEQ, EMBED])
    }
}

impl<const SEQ: usize, const EMBED: usize, const HEADS: usize, const HEAD_DIM: usize, E, B>
    HasParameters<E, B> for MultiHeadAttention<SEQ, EMBED, HEADS, HEAD_DIM, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn parameters<'a>(&'a self, out: &mut Vec<ParameterRef<'a, E, B>>) {
        self.q_proj.parameters(out);
        self.k_proj.parameters(out);
        self.v_proj.parameters(out);
        self.out_proj.parameters(out);
    }

    fn parameters_mut<'a>(&'a mut self, out: &mut Vec<ParameterRefMut<'a, E, B>>) {
        self.q_proj.parameters_mut(out);
        self.k_proj.parameters_mut(out);
        self.v_proj.parameters_mut(out);
        self.out_proj.parameters_mut(out);
    }
}

pub(crate) fn ensure_head_shape<const EMBED: usize, const HEADS: usize, const HEAD_DIM: usize>()
-> Result<()> {
    const { const_check::nonzero(EMBED, "multi_head_attention", "EMBED") };
    const { const_check::nonzero(HEADS, "multi_head_attention", "HEADS") };
    const { const_check::nonzero(HEAD_DIM, "multi_head_attention", "HEAD_DIM") };
    const {
        const_check::mul_eq(
            HEADS,
            HEAD_DIM,
            EMBED,
            "multi_head_attention",
            "HEADS",
            "HEAD_DIM",
            "EMBED",
        )
    };

    if HEADS == 0 || HEADS * HEAD_DIM != EMBED {
        return Err(ShapeError::LengthMismatch {
            expected: EMBED,
            found: HEADS * HEAD_DIM,
        }
        .into());
    }
    Ok(())
}

fn project_3d<const SEQ: usize, const IN: usize, const OUT: usize, E, B>(
    layer: &Linear<IN, OUT, E, B>,
    input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<IN>>, E, B>,
) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<OUT>>, E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    let batch = input.shape().dims()[0];
    let flat = input.reshape_with_shape::<D2<AnyDim, C<IN>>>([batch * SEQ, IN])?;
    let mut ctx = ();
    layer
        .forward(&flat, &mut ctx)?
        .reshape_with_shape([batch, SEQ, OUT])
}
