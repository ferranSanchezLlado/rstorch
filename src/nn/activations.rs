use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;

pub fn relu<S, E, B>(input: &Tensor<S, E, B>) -> Result<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    input.relu()
}
