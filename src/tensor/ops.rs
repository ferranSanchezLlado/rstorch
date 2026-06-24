use super::autograd::{
    AnyTensor, raw_div, raw_div_scalar, raw_from_vec_like, raw_full_like, raw_mul, raw_mul_scalar,
    raw_neg,
};
use super::{RawTensor, Scalar, Tensor, Tensor1D, Tensor4D};
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::{DeviceError, Error, Result, ShapeError};
use crate::shape::{
    C, D0, D1, D2, D3, D4, DimEntry, DimSpec, Shape, ShapeSpec, StaticShape, bind_and_check,
};

type BinaryKernel<E, B> =
    fn(
        &<B as Backend<E>>::Device,
        &<B as Backend<E>>::Storage,
        &<B as Backend<E>>::Storage,
        usize,
    ) -> std::result::Result<<B as Backend<E>>::Storage, <B as Backend<E>>::Error>;

type ScalarKernel<E, B> =
    fn(
        &<B as Backend<E>>::Device,
        &<B as Backend<E>>::Storage,
        E,
        usize,
    ) -> std::result::Result<<B as Backend<E>>::Storage, <B as Backend<E>>::Error>;

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn reshape<T>(&self) -> Result<Tensor<T, E, B>>
    where
        T: StaticShape,
    {
        let shape = T::static_shape();
        let expected = shape.numel()?;
        if expected != self.numel() {
            return Err(ShapeError::LengthMismatch {
                expected,
                found: self.numel(),
            }
            .into());
        }

        let source = self.contiguous()?;
        let layout = source.raw().layout().reshape_contiguous(shape)?;
        let raw = source.raw().view_with_layout(layout)?;
        let input_raw = self.raw().clone();
        Tensor::<T, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let layout = grad
                .layout()
                .reshape_contiguous(input_raw.shape().clone())?;
            Ok(vec![Some(grad.view_with_layout(layout)?)])
        })
    }

    pub fn reshape_with_shape<T>(&self, shape: impl Into<Shape>) -> Result<Tensor<T, E, B>>
    where
        T: ShapeSpec,
    {
        let shape = shape.into();
        let expected = shape.numel()?;
        if expected != self.numel() {
            return Err(ShapeError::LengthMismatch {
                expected,
                found: self.numel(),
            }
            .into());
        }

        let source = self.contiguous()?;
        let layout = source.raw().layout().reshape_contiguous(shape)?;
        let raw = source.raw().view_with_layout(layout)?;
        let input_raw = self.raw().clone();
        Tensor::<T, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let layout = grad
                .layout()
                .reshape_contiguous(input_raw.shape().clone())?;
            Ok(vec![Some(grad.view_with_layout(layout)?)])
        })
    }

    pub fn reshape0(&self) -> Result<Tensor<D0, E, B>> {
        self.reshape::<D0>()
    }

    pub fn reshape1<const N: usize>(&self) -> Result<Tensor<D1<C<N>>, E, B>> {
        self.reshape::<D1<C<N>>>()
    }

    pub fn reshape2<const M: usize, const N: usize>(&self) -> Result<Tensor<D2<C<M>, C<N>>, E, B>> {
        self.reshape::<D2<C<M>, C<N>>>()
    }

    pub fn reshape3<const X: usize, const Y: usize, const Z: usize>(
        &self,
    ) -> Result<Tensor<D3<C<X>, C<Y>, C<Z>>, E, B>> {
        self.reshape::<D3<C<X>, C<Y>, C<Z>>>()
    }

    pub fn reshape4<const N: usize, const CH: usize, const H: usize, const W: usize>(
        &self,
    ) -> Result<Tensor4D<N, CH, H, W, E, B>> {
        self.reshape::<D4<C<N>, C<CH>, C<H>, C<W>>>()
    }

    pub fn flatten<const N: usize>(&self) -> Result<Tensor<D1<C<N>>, E, B>> {
        self.reshape1::<N>()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "add", B::add, |grad, _lhs, _rhs| {
            Ok((grad.clone(), grad.clone()))
        })
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "sub", B::sub, |grad, _lhs, _rhs| {
            Ok((grad.clone(), raw_neg(grad)?))
        })
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "mul", B::mul, |grad, lhs, rhs| {
            Ok((raw_mul(grad, rhs)?, raw_mul(grad, lhs)?))
        })
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "div", B::div, |grad, lhs, rhs| {
            let dx = raw_div(grad, rhs)?;
            let rhs_sq = raw_mul(rhs, rhs)?;
            let dy = raw_neg(&raw_div(&raw_mul(grad, lhs)?, &rhs_sq)?)?;
            Ok((dx, dy))
        })
    }

    pub fn add_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::add_scalar, |grad| Ok(grad.clone()))
    }

    pub fn sub_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::sub_scalar, |grad| Ok(grad.clone()))
    }

    pub fn mul_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::mul_scalar, move |grad| raw_mul_scalar(grad, rhs))
    }

    pub fn div_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::div_scalar, move |grad| raw_div_scalar(grad, rhs))
    }

    pub fn sum(&self) -> Result<Scalar<E, B>> {
        let input = self.contiguous()?;
        let storage =
            B::sum(input.device(), input.raw().storage(), input.numel()).map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, Shape::known([]))?;
        let input_raw = self.raw().clone();
        Tensor::<D0, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            Ok(vec![Some(raw_full_like(&input_raw, seed)?)])
        })
    }

    pub fn relu(&self) -> Result<Self> {
        let input = self.contiguous()?;
        let zero = E::zero();
        let values = input
            .to_vec()?
            .into_iter()
            .map(|value| if value > zero { value } else { zero })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let mask = input_raw
                .to_vec()?
                .into_iter()
                .map(|value| if value > zero { E::one() } else { zero });
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(mask)
                .map(|(g, m)| g * m)
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    fn ensure_same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)
    }

    fn binary_same_shape(
        &self,
        rhs: &Self,
        op: &'static str,
        kernel: BinaryKernel<E, B>,
        backward: impl Fn(
            &RawTensor<E, B>,
            &RawTensor<E, B>,
            &RawTensor<E, B>,
        ) -> Result<(RawTensor<E, B>, RawTensor<E, B>)>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, op)?;

        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        bind_and_check(
            op,
            S::dim_entries(0)
                .into_iter()
                .zip(lhs_dims.iter().copied())
                .chain(S::dim_entries(1).into_iter().zip(rhs_dims.iter().copied())),
        )?;

        for (axis, (&lhs, &rhs)) in lhs_dims.iter().zip(rhs_dims.iter()).enumerate() {
            if lhs != rhs {
                return Err(ShapeError::DimMismatch {
                    op,
                    operand: 1,
                    axis,
                    expected: lhs,
                    found: rhs,
                }
                .into());
            }
        }

        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let storage = kernel(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            lhs_input.numel(),
        )
        .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let (lhs_grad, rhs_grad) = backward(grad, &lhs_raw, &rhs_raw)?;
                Ok(vec![Some(lhs_grad), Some(rhs_grad)])
            },
        )
    }

    fn unary_scalar(
        &self,
        rhs: E,
        kernel: ScalarKernel<E, B>,
        backward: impl Fn(&RawTensor<E, B>) -> Result<RawTensor<E, B>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let input = self.contiguous()?;
        let storage = kernel(input.device(), input.raw().storage(), rhs, input.numel())
            .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?;
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            Ok(vec![Some(backward(grad)?)])
        })
    }
}

impl<A, E, B> Tensor<D1<A>, E, B>
where
    A: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn cat1<R, const N: usize>(&self, rhs: &Tensor<D1<R>, E, B>) -> Result<Tensor1D<N, E, B>>
    where
        R: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "cat1")?;
        let found =
            self.numel()
                .checked_add(rhs.numel())
                .ok_or_else(|| ShapeError::NumelOverflow {
                    dims: vec![self.numel(), rhs.numel()].into_boxed_slice(),
                })?;
        if found != N {
            return Err(ShapeError::LengthMismatch { expected: N, found }.into());
        }

        let mut data = self.to_vec()?;
        data.extend(rhs.to_vec()?);
        let raw = RawTensor::from_vec_on(self.device().clone(), data, Shape::known([N]))?;
        let left_len = self.numel();
        let lhs_shape = self.shape().clone();
        let rhs_shape = rhs.shape().clone();
        Tensor::<D1<C<N>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let data = grad.to_vec()?;
                let lhs = RawTensor::from_vec_on(
                    grad.device().clone(),
                    data[..left_len].to_vec(),
                    lhs_shape.clone(),
                )?;
                let rhs = RawTensor::from_vec_on(
                    grad.device().clone(),
                    data[left_len..].to_vec(),
                    rhs_shape.clone(),
                )?;
                Ok(vec![Some(lhs), Some(rhs)])
            },
        )
    }
}

impl<A, K, E, B> Tensor<D2<A, K>, E, B>
where
    A: DimSpec,
    K: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn transpose(&self) -> Result<Tensor<D2<K, A>, E, B>> {
        let raw = self.raw().view_with_layout(self.layout().transpose2()?)?;
        Tensor::<D2<K, A>, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], |grad| {
            Ok(vec![Some(
                grad.view_with_layout(grad.layout().transpose2()?)?,
            )])
        })
    }

    pub fn matmul<Cc>(&self, rhs: &Tensor<D2<K, Cc>, E, B>) -> Result<Tensor<D2<A, Cc>, E, B>>
    where
        Cc: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "matmul")?;

        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let m = lhs_dims[0];
        let k = lhs_dims[1];
        let rhs_k = rhs_dims[0];
        let n = rhs_dims[1];

        bind_and_check(
            "matmul",
            [
                (DimEntry::of::<A>(0, 0), m),
                (DimEntry::of::<K>(0, 1), k),
                (DimEntry::of::<K>(1, 0), rhs_k),
                (DimEntry::of::<Cc>(1, 1), n),
            ],
        )?;

        if k != rhs_k {
            return Err(ShapeError::DimMismatch {
                op: "matmul",
                operand: 1,
                axis: 0,
                expected: k,
                found: rhs_k,
            }
            .into());
        }

        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let storage = B::matmul(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            m,
            k,
            n,
        )
        .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, Shape::known([m, n]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<A, Cc>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let rhs_t = rhs_raw.view_with_layout(rhs_raw.layout().transpose2()?)?;
                let lhs_t = lhs_raw.view_with_layout(lhs_raw.layout().transpose2()?)?;
                Ok(vec![
                    Some(raw_matmul(grad, &rhs_t)?),
                    Some(raw_matmul(&lhs_t, grad)?),
                ])
            },
        )
    }
}

impl<A, N, E, B> Tensor<D2<A, N>, E, B>
where
    A: DimSpec,
    N: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn add_row(&self, rhs: &Tensor<D1<N>, E, B>) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "add_row")?;
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        bind_and_check(
            "add_row",
            [
                (DimEntry::of::<A>(0, 0), rows),
                (DimEntry::of::<N>(0, 1), cols),
                (DimEntry::of::<N>(1, 0), rhs.shape().dims()[0]),
            ],
        )?;
        if rhs.shape().dims()[0] != cols {
            return Err(ShapeError::DimMismatch {
                op: "add_row",
                operand: 1,
                axis: 0,
                expected: cols,
                found: rhs.shape().dims()[0],
            }
            .into());
        }

        let lhs_values = self.to_vec()?;
        let rhs_values = rhs.to_vec()?;
        let values = lhs_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| value + rhs_values[idx % cols])
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let mut bias_grad = vec![E::zero(); cols];
                for row in 0..rows {
                    for col in 0..cols {
                        bias_grad[col] = bias_grad[col] + grad_values[row * cols + col];
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, grad_values)?),
                    Some(raw_from_vec_like(&rhs_raw, bias_grad)?),
                ])
            },
        )
    }
}

fn ensure_same_device<E, B>(lhs: &B::Device, rhs: &B::Device, op: &'static str) -> Result<()>
where
    E: FloatDType,
    B: Backend<E>,
{
    if lhs != rhs {
        return Err(DeviceError::Mismatch {
            op,
            lhs: format!("{lhs:?}"),
            rhs: format!("{rhs:?}"),
        }
        .into());
    }
    Ok(())
}

fn raw_matmul<E, B>(lhs: &RawTensor<E, B>, rhs: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    let lhs_input: RawTensor<E, B> =
        RawTensor::from_vec_on(lhs.device().clone(), lhs.to_vec()?, lhs.shape().clone())?;
    let rhs_input: RawTensor<E, B> =
        RawTensor::from_vec_on(rhs.device().clone(), rhs.to_vec()?, rhs.shape().clone())?;
    let m = lhs.shape().dims()[0];
    let k = lhs.shape().dims()[1];
    let n = rhs.shape().dims()[1];
    let storage = B::matmul(
        lhs.device(),
        lhs_input.storage(),
        rhs_input.storage(),
        m,
        k,
        n,
    )
    .map_err(Error::backend)?;
    RawTensor::from_storage_on(lhs.device().clone(), storage, Shape::known([m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, CpuDevice};
    use crate::dtype::DTypeId;
    use crate::error::{DeviceError, Error, ShapeError};
    use crate::shape::{AnyDim, Sym};
    use crate::tensor::test_support::{Batch, Hidden, TestBackend, TestDevice};
    use crate::{Scalar, Tensor2D};

    #[test]
    fn reshape_views_to_static_target() {
        let tensor = Tensor2D::<2, 6>::from_vec((0..12).map(|x| x as f32).collect()).unwrap();
        let reshaped = tensor.reshape1::<12>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[12]);
        assert!(reshaped.shares_storage_with(&tensor));

        let reshaped = tensor.reshape2::<3, 4>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 4]);
        assert!(reshaped.is_contiguous());
        assert!(reshaped.is_view());
        assert_eq!(
            reshaped.to_vec().unwrap(),
            (0..12).map(|x| x as f32).collect::<Vec<_>>()
        );

        let reshaped = tensor.reshape3::<2, 3, 2>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[2, 3, 2]);
        assert!(reshaped.shares_storage_with(&tensor));

        let reshaped = tensor.reshape4::<1, 2, 2, 3>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[1, 2, 2, 3]);
        assert!(reshaped.shares_storage_with(&tensor));

        let err = tensor.reshape2::<5, 2>().unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 10,
                found: 12,
            })
        ));
    }

    #[test]
    fn matmul_computes_and_preserves_types() {
        let lhs = Tensor::<D2<C<2>, C<3>>>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs =
            Tensor::<D2<C<3>, C<2>>>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let out: Tensor<D2<C<2>, C<2>>> = lhs.matmul(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[2, 2]);
        assert_eq!(out.to_vec().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_anydim_still_checks_structure() {
        let lhs = Tensor::<D2<C<2>, AnyDim>>::from_vec_with_shape(vec![1.0; 6], [2, 3]).unwrap();
        let rhs = Tensor::<D2<AnyDim, C<2>>>::from_vec_with_shape(vec![1.0; 8], [4, 2]).unwrap();
        let err = lhs.matmul(&rhs).unwrap_err();

        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "matmul",
                operand: 1,
                axis: 0,
                expected: 3,
                found: 4,
            })
        ));
    }

    #[test]
    fn matmul_shared_sym_reports_symbol_mismatch() {
        let lhs =
            Tensor::<D2<C<2>, Sym<Hidden>>>::from_vec_with_shape(vec![1.0; 6], [2, 3]).unwrap();
        let rhs =
            Tensor::<D2<Sym<Hidden>, C<2>>>::from_vec_with_shape(vec![1.0; 8], [4, 2]).unwrap();
        let err = lhs.matmul(&rhs).unwrap_err();

        assert!(matches!(
            err,
            Error::Shape(ShapeError::SymbolMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn add_and_mul_compute_elementwise() {
        let lhs = Tensor2D::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor2D::<2, 2>::from_vec(vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(
            lhs.add(&rhs).unwrap().to_vec().unwrap(),
            vec![6.0, 8.0, 10.0, 12.0]
        );
        assert_eq!(
            lhs.mul(&rhs).unwrap().to_vec().unwrap(),
            vec![5.0, 12.0, 21.0, 32.0]
        );
    }

    #[test]
    fn same_shape_ops_reject_runtime_mismatches_for_anydim() {
        let lhs = Tensor::<D1<AnyDim>>::from_vec_with_shape(vec![1.0, 2.0], [2]).unwrap();
        let rhs = Tensor::<D1<AnyDim>>::from_vec_with_shape(vec![3.0, 4.0, 5.0], [3]).unwrap();

        let err = lhs.add(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "add",
                operand: 1,
                axis: 0,
                expected: 2,
                found: 3,
            })
        ));
    }

    #[test]
    fn sub_and_div_compute_elementwise() {
        let lhs = Tensor2D::<2, 2>::from_vec(vec![8.0, 9.0, 10.0, 12.0]).unwrap();
        let rhs = Tensor2D::<2, 2>::from_vec(vec![2.0, 3.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            lhs.sub(&rhs).unwrap().to_vec().unwrap(),
            vec![6.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(
            lhs.div(&rhs).unwrap().to_vec().unwrap(),
            vec![4.0, 3.0, 2.0, 2.0]
        );
    }

    #[test]
    fn sum_reduces_to_scalar() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = tensor.sum().unwrap();

        assert_eq!(out.shape().dims(), &[]);
        assert_eq!(out.rank(), 0);
        assert_eq!(out.numel(), 1);
        assert_eq!(out.to_vec().unwrap(), vec![21.0]);
    }

    #[test]
    fn reshape_with_shape_supports_symbolic_targets() {
        let tensor = Tensor1D::<6>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = tensor
            .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
            .unwrap();

        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert!(reshaped.shares_storage_with(&tensor));
        assert_eq!(
            reshaped.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let err = tensor
            .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([2, 3])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "validate",
                operand: 0,
                axis: 1,
                expected: 2,
                found: 3,
            })
        ));
    }

    #[test]
    fn flatten_uses_caller_named_static_target() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flattened = tensor.flatten::<6>().unwrap();

        assert_eq!(flattened.shape().dims(), &[6]);
        assert!(flattened.shares_storage_with(&tensor));
        assert_eq!(
            flattened.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let err = tensor.flatten::<5>().unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 5,
                found: 6,
            })
        ));
    }

    #[test]
    fn transpose_swaps_2d_shape_and_values() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.layout().strides(), &[1, 3]);
        assert!(transposed.shares_storage_with(&tensor));
        assert!(!transposed.is_contiguous());
        assert_eq!(
            transposed.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn contiguous_materializes_non_contiguous_views() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();
        let contiguous = transposed.contiguous().unwrap();

        assert!(contiguous.is_contiguous());
        assert!(!contiguous.shares_storage_with(&tensor));
        assert_eq!(
            contiguous.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn ops_handle_non_contiguous_views() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = tensor.transpose().unwrap();

        assert_eq!(
            view.add(&view).unwrap().to_vec().unwrap(),
            vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
        );
        assert_eq!(
            view.mul_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
        );
        assert_eq!(view.sum().unwrap().to_vec().unwrap(), vec![21.0]);
    }

    #[test]
    fn matmul_handles_transposed_views() {
        let base = Tensor2D::<2, 3>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let lhs = base.transpose().unwrap();
        let rhs = Tensor2D::<2, 1>::from_vec(vec![10.0, 1.0]).unwrap();
        let out = lhs.matmul(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[3, 1]);
        assert_eq!(out.to_vec().unwrap(), vec![15.0, 43.0, 26.0]);
    }

    #[test]
    fn cat1_concatenates_1d_tensors_with_static_target() {
        let lhs = Tensor::<D1<Sym<Batch>>>::from_vec_with_shape(vec![1.0, 2.0], [2]).unwrap();
        let rhs = Tensor1D::<3>::from_vec(vec![3.0, 4.0, 5.0]).unwrap();
        let out = lhs.cat1::<C<3>, 5>(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[5]);
        assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let err = lhs.cat1::<C<3>, 4>(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 4,
                found: 5,
            })
        ));
    }

    #[test]
    fn scalar_ops_compute_for_cpu_f32_and_preserve_metadata() {
        let tensor = Tensor2D::<2, 2>::from_vec(vec![2.0, 4.0, 6.0, 8.0]).unwrap();

        let add = tensor.add_scalar(1.5).unwrap();
        assert_eq!(add.to_vec().unwrap(), vec![3.5, 5.5, 7.5, 9.5]);
        assert_eq!(add.shape(), tensor.shape());
        assert_eq!(add.rank(), 2);
        assert_eq!(add.numel(), 4);
        assert_eq!(add.dtype(), DTypeId::F32);
        assert_eq!(add.device(), &CpuDevice::Cpu);
        assert_eq!(add.layout(), tensor.layout());
        assert_eq!(tensor.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

        assert_eq!(
            tensor.sub_scalar(1.0).unwrap().to_vec().unwrap(),
            vec![1.0, 3.0, 5.0, 7.0]
        );
        assert_eq!(
            tensor.mul_scalar(0.5).unwrap().to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn scalar_ops_compute_for_cpu_f64() {
        let tensor = Tensor::<D1<C<3>>, f64, Cpu>::from_vec(vec![1.0, 2.0, 4.0]).unwrap();

        assert_eq!(
            tensor.add_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![3.0, 4.0, 6.0]
        );
        assert_eq!(
            tensor.sub_scalar(0.5).unwrap().to_vec().unwrap(),
            vec![0.5, 1.5, 3.5]
        );
        assert_eq!(
            tensor.mul_scalar(3.0).unwrap().to_vec().unwrap(),
            vec![3.0, 6.0, 12.0]
        );
        assert_eq!(
            tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![0.5, 1.0, 2.0]
        );
    }

    #[test]
    fn scalar_ops_work_for_scalar_tensor() {
        let tensor = Scalar::<f32, Cpu>::from_vec(vec![3.0]).unwrap();
        let out = tensor.mul_scalar(2.0).unwrap();

        assert_eq!(out.shape().dims(), &[]);
        assert_eq!(out.rank(), 0);
        assert_eq!(out.numel(), 1);
        assert_eq!(out.to_vec().unwrap(), vec![6.0]);
    }

    #[test]
    fn scalar_ops_work_for_symbolic_shape() {
        let tensor = Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        )
        .unwrap();
        let out = tensor.add_scalar(10.0).unwrap();

        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(
            out.to_vec().unwrap(),
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        );
    }

    #[test]
    fn multi_input_ops_reject_device_mismatch() {
        type TestTensor = Tensor<D1<C<2>>, f32, TestBackend>;

        let lhs = TestTensor::from_raw(
            RawTensor::from_vec_on(TestDevice::A, vec![1.0, 2.0], Shape::known([2])).unwrap(),
        )
        .unwrap();
        let rhs = TestTensor::from_raw(
            RawTensor::from_vec_on(TestDevice::B, vec![3.0, 4.0], Shape::known([2])).unwrap(),
        )
        .unwrap();

        let err = lhs.add(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Device(DeviceError::Mismatch {
                op: "add",
                lhs,
                rhs,
            }) if lhs == "A" && rhs == "B"
        ));
    }

    #[test]
    fn scalar_backend_errors_are_boxed_sources() {
        let raw = RawTensor::<f32, crate::tensor::test_support::FailingBackend>::from_storage_on(
            TestDevice::A,
            vec![1.0, 2.0],
            Shape::known([2]),
        )
        .unwrap();
        let tensor =
            Tensor::<D1<C<2>>, f32, crate::tensor::test_support::FailingBackend>::from_raw(raw)
                .unwrap();

        let err = tensor.add_scalar(1.0).unwrap_err();
        assert!(matches!(err, Error::Backend(_)));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod metal_tests {
        use super::*;
        use crate::backend::{Metal, MetalDevice};

        fn device() -> Option<MetalDevice> {
            <Metal as Backend<f32>>::default_device().ok()
        }

        fn tensor_1d(device: MetalDevice, values: Vec<f32>) -> Tensor<D1<C<4>>, f32, Metal> {
            let raw =
                RawTensor::<f32, Metal>::from_vec_on(device, values, Shape::known([4])).unwrap();
            Tensor::<D1<C<4>>, f32, Metal>::from_raw(raw).unwrap()
        }

        #[test]
        fn metal_scalar_add_round_trips() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::known([2, 2]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<2>>, f32, Metal>::from_raw(raw).unwrap();

            let out = tensor.add_scalar(1.0).unwrap();
            assert_eq!(out.to_vec().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
        }

        #[test]
        fn metal_scalar_sub_mul_and_div_round_trip() {
            let Some(device) = device() else {
                return;
            };
            let tensor = tensor_1d(device, vec![8.0, 9.0, 10.0, 12.0]);

            assert_eq!(
                tensor.sub_scalar(2.0).unwrap().to_vec().unwrap(),
                vec![6.0, 7.0, 8.0, 10.0]
            );
            assert_eq!(
                tensor.mul_scalar(3.0).unwrap().to_vec().unwrap(),
                vec![24.0, 27.0, 30.0, 36.0]
            );
            assert_eq!(
                tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
                vec![4.0, 4.5, 5.0, 6.0]
            );
        }

        #[test]
        fn metal_binary_ops_match_cpu() {
            let Some(device) = device() else {
                return;
            };
            let lhs_values = vec![8.0, 9.0, 10.0, 12.0];
            let rhs_values = vec![2.0, 3.0, 5.0, 6.0];
            let lhs = tensor_1d(device.clone(), lhs_values.clone());
            let rhs = tensor_1d(device.clone(), rhs_values.clone());
            let cpu_lhs = Tensor::<D1<C<4>>>::from_vec(lhs_values).unwrap();
            let cpu_rhs = Tensor::<D1<C<4>>>::from_vec(rhs_values).unwrap();

            assert_eq!(
                lhs.add(&rhs).unwrap().to_vec().unwrap(),
                cpu_lhs.add(&cpu_rhs).unwrap().to_vec().unwrap()
            );
            assert_eq!(
                lhs.mul(&rhs).unwrap().to_vec().unwrap(),
                cpu_lhs.mul(&cpu_rhs).unwrap().to_vec().unwrap()
            );

            let sub_storage = <Metal as Backend<f32>>::sub(
                lhs.device(),
                lhs.raw().storage(),
                rhs.raw().storage(),
                lhs.numel(),
            )
            .unwrap();
            let sub_raw = RawTensor::<f32, Metal>::from_storage_on(
                device.clone(),
                sub_storage,
                Shape::known([4]),
            )
            .unwrap();
            assert_eq!(sub_raw.to_vec().unwrap(), vec![6.0, 6.0, 5.0, 6.0]);

            let div_storage = <Metal as Backend<f32>>::div(
                lhs.device(),
                lhs.raw().storage(),
                rhs.raw().storage(),
                lhs.numel(),
            )
            .unwrap();
            let div_raw =
                RawTensor::<f32, Metal>::from_storage_on(device, div_storage, Shape::known([4]))
                    .unwrap();
            assert_eq!(div_raw.to_vec().unwrap(), vec![4.0, 3.0, 2.0, 2.0]);
        }

        #[test]
        fn metal_sum_uses_backend_reduction() {
            let Some(device) = device() else {
                return;
            };
            let tensor = tensor_1d(device, vec![1.0, 2.0, 3.0, 4.0]);

            let out = tensor.sum().unwrap();
            assert_eq!(out.shape().dims(), &[]);
            assert_eq!(out.to_vec().unwrap(), vec![10.0]);
        }

        #[test]
        fn metal_empty_sum_returns_zero_scalar() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(device, Vec::new(), Shape::known([0]))
                .unwrap();
            let tensor = Tensor::<D1<C<0>>, f32, Metal>::from_raw(raw).unwrap();

            let out = tensor.sum().unwrap();
            assert_eq!(out.shape().dims(), &[]);
            assert_eq!(out.to_vec().unwrap(), vec![0.0]);
        }

        #[test]
        fn metal_transpose_and_reshape_with_shape_round_trip() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Shape::known([2, 3]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<3>>, f32, Metal>::from_raw(raw).unwrap();

            assert_eq!(
                tensor.transpose().unwrap().to_vec().unwrap(),
                vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
            );

            let reshaped = tensor
                .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
                .unwrap();
            assert_eq!(reshaped.shape().dims(), &[3, 2]);
            assert_eq!(
                reshaped.to_vec().unwrap(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
        }

        #[test]
        fn metal_matmul_matches_cpu() {
            let Some(device) = device() else {
                return;
            };
            let lhs_raw = RawTensor::<f32, Metal>::from_vec_on(
                device.clone(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Shape::known([2, 3]),
            )
            .unwrap();
            let rhs_raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                Shape::known([3, 2]),
            )
            .unwrap();
            let lhs = Tensor::<D2<C<2>, C<3>>, f32, Metal>::from_raw(lhs_raw).unwrap();
            let rhs = Tensor::<D2<C<3>, C<2>>, f32, Metal>::from_raw(rhs_raw).unwrap();

            let out = lhs.matmul(&rhs).unwrap();
            assert_eq!(out.to_vec().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
        }

        #[test]
        fn metal_preserves_typed_shape_facade() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device.clone(),
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::known([2, 2]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<2>>, f32, Metal>::from_raw(raw).unwrap();
            let out = tensor.mul_scalar(2.0).unwrap();

            assert_eq!(out.shape().dims(), &[2, 2]);
            assert_eq!(out.rank(), 2);
            assert_eq!(out.numel(), 4);
            assert_eq!(out.dtype(), DTypeId::F32);
            assert_eq!(out.device(), &device);
        }
    }
}
