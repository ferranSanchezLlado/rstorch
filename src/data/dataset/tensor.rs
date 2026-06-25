use super::Dataset;
use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::{DataError, Result};
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct TensorDataset<Item, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    data: Vec<Vec<E>>,
    sample_lens: Vec<usize>,
    len: usize,
    _item: PhantomData<(Item, B)>,
}

pub trait IntoTensorDataset<Item, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn into_tensor_dataset_parts(self) -> Result<(Vec<Vec<E>>, Vec<usize>, usize)>;
}

impl<Item, E, B> TensorDataset<Item, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new<Tensors>(tensors: Tensors) -> Result<Self>
    where
        Tensors: IntoTensorDataset<Item, E, B>,
    {
        let (data, sample_lens, len) = tensors.into_tensor_dataset_parts()?;
        Ok(Self {
            data,
            sample_lens,
            len,
            _item: PhantomData,
        })
    }
}

impl<S, E, B> IntoTensorDataset<Vec<Vec<E>>, E, B> for Vec<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn into_tensor_dataset_parts(self) -> Result<(Vec<Vec<E>>, Vec<usize>, usize)> {
        let first = self.first().ok_or(DataError::InvalidTensorDataset {
            reason: "at least one tensor is required",
        })?;
        let len = *first
            .shape()
            .dims()
            .first()
            .ok_or(DataError::InvalidTensorDataset {
                reason: "tensors must have a leading dimension",
            })?;

        let mut data = Vec::with_capacity(self.len());
        let mut sample_lens = Vec::with_capacity(self.len());

        for tensor in &self {
            if tensor.shape().dims().first().copied() != Some(len) {
                return Err(DataError::InvalidTensorDataset {
                    reason: "all tensors must share the same leading dimension",
                }
                .into());
            }

            let sample_len = tensor.numel().checked_div(len).unwrap_or(0);
            sample_lens.push(sample_len);
            data.push(tensor.to_vec()?);
        }

        Ok((data, sample_lens, len))
    }
}

impl<S0, S1, E, B> IntoTensorDataset<(Vec<E>, Vec<E>), E, B>
    for (Tensor<S0, E, B>, Tensor<S1, E, B>)
where
    S0: ShapeSpec,
    S1: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn into_tensor_dataset_parts(self) -> Result<(Vec<Vec<E>>, Vec<usize>, usize)> {
        let len = leading_len(&self.0)?;
        if self.1.shape().dims().first().copied() != Some(len) {
            return Err(DataError::InvalidTensorDataset {
                reason: "all tensors must share the same leading dimension",
            }
            .into());
        }

        Ok((
            vec![self.0.to_vec()?, self.1.to_vec()?],
            vec![sample_len(&self.0, len), sample_len(&self.1, len)],
            len,
        ))
    }
}

impl<E, B> Dataset for TensorDataset<Vec<Vec<E>>, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Item = Vec<Vec<E>>;
    type Error = DataError;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        if index >= self.len {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }

        self.data
            .iter()
            .zip(&self.sample_lens)
            .map(|(data, &sample_len)| {
                let start = index * sample_len;
                let end = start + sample_len;
                Ok(data[start..end].to_vec())
            })
            .collect()
    }
}

impl<E, B> Dataset for TensorDataset<(Vec<E>, Vec<E>), E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Item = (Vec<E>, Vec<E>);
    type Error = DataError;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        if index >= self.len {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }

        Ok((
            sample_at(&self.data[0], self.sample_lens[0], index),
            sample_at(&self.data[1], self.sample_lens[1], index),
        ))
    }
}

fn leading_len<S, E, B>(tensor: &Tensor<S, E, B>) -> Result<usize>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    tensor
        .shape()
        .dims()
        .first()
        .copied()
        .ok_or(DataError::InvalidTensorDataset {
            reason: "tensors must have a leading dimension",
        })
        .map_err(Into::into)
}

fn sample_len<S, E, B>(tensor: &Tensor<S, E, B>, len: usize) -> usize
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    tensor.numel().checked_div(len).unwrap_or(0)
}

fn sample_at<E: Clone>(data: &[E], sample_len: usize, index: usize) -> Vec<E> {
    let start = index * sample_len;
    let end = start + sample_len;
    data[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Batch;
    use crate::error::Error;
    use crate::shape::{C, D1, D2, Sym};

    #[test]
    fn tensor_dataset_validates_shared_leading_dimension() {
        let a = Tensor::<D2<C<2>, C<2>>>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::<D2<C<3>, C<2>>>::from_vec(vec![1.0; 6]).unwrap();

        assert!(TensorDataset::new(vec![a.clone()]).is_ok());
        assert!(matches!(
            TensorDataset::new(vec![
                a.reshape_with_shape::<D2<Sym<Batch>, C<2>>>([2, 2])
                    .unwrap(),
                b.reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
                    .unwrap(),
            ]),
            Err(Error::Data(DataError::InvalidTensorDataset { .. }))
        ));
    }

    #[test]
    fn tensor_dataset_accepts_heterogeneous_pair_shapes() {
        let x = Tensor::<D2<C<2>, C<2>>>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Tensor::<D1<C<2>>>::from_vec(vec![10.0, 20.0]).unwrap();
        let dataset = TensorDataset::new((x, y)).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).unwrap(), (vec![1.0, 2.0], vec![10.0]));
        assert_eq!(dataset.get(1).unwrap(), (vec![3.0, 4.0], vec![20.0]));
    }
}
