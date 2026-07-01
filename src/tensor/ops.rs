use super::autograd::{
    AnyTensor, raw_div, raw_div_scalar, raw_from_vec_like, raw_full_like, raw_mul, raw_mul_scalar,
    raw_neg,
};
use super::{Mask, RawTensor, Scalar, Tensor, Tensor1D, Tensor4D};
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::{DataError, DeviceError, Error, Result, ShapeError};
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
            let grad = RawTensor::from_vec_on(
                grad.device().clone(),
                grad.to_vec()?,
                grad.shape().clone(),
            )?;
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
            let grad = RawTensor::from_vec_on(
                grad.device().clone(),
                grad.to_vec()?,
                grad.shape().clone(),
            )?;
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
        let values = input
            .to_vec()?
            .into_iter()
            .map(|value| if value > E::ZERO { value } else { E::ZERO })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let mask = input_raw
                .to_vec()?
                .into_iter()
                .map(|value| if value > E::ZERO { E::ONE } else { E::ZERO });
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(mask)
                .map(|(g, m)| g * m)
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    pub fn neg(&self) -> Result<Self> {
        self.unary_map(|x| -x, |_x, _y| -E::ONE)
    }

    pub fn exp(&self) -> Result<Self> {
        self.unary_map(|x| x.exp(), |_x, y| y)
    }

    pub fn ln(&self) -> Result<Self> {
        self.unary_map(|x| x.ln(), |x, _y| E::ONE / x)
    }

    pub fn tanh(&self) -> Result<Self> {
        self.unary_map(|x| x.tanh(), |_x, y| E::ONE - y * y)
    }

    pub fn sigmoid(&self) -> Result<Self> {
        self.unary_map(|x| E::ONE / (E::ONE + (-x).exp()), |_x, y| y * (E::ONE - y))
    }

    /// Raises each element to `exponent`. Gradients follow the mathematical
    /// derivative and may produce infinities or NaNs at values outside the
    /// real-valued derivative domain for the chosen exponent.
    pub fn powf(&self, exponent: E) -> Result<Self> {
        self.unary_map(
            move |x| x.powf(exponent),
            move |x, _y| exponent * x.powf(exponent - E::ONE),
        )
    }

    /// Computes elementwise square root. Gradients follow the mathematical
    /// derivative and may produce infinities or NaNs at non-positive inputs.
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_map(|x| x.sqrt(), |x, _y| E::HALF / x.sqrt())
    }

    /// Computes elementwise reciprocal square root. Gradients follow the
    /// mathematical derivative and may produce infinities or NaNs at
    /// non-positive inputs.
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary_map(|x| E::ONE / x.sqrt(), |x, _y| -E::HALF / (x * x.sqrt()))
    }

    pub fn clamp(&self, min: E, max: E) -> Result<Self> {
        self.unary_map(
            move |x| {
                if x < min {
                    min
                } else if x > max {
                    max
                } else {
                    x
                }
            },
            move |x, _y| {
                if x < min || x > max { E::ZERO } else { E::ONE }
            },
        )
    }

    pub fn gelu(&self) -> Result<Self> {
        self.unary_map(
            move |x| {
                let x3 = x * x * x;
                E::HALF * x * (E::ONE + (E::GELU_K * (x + E::GELU_C * x3)).tanh())
            },
            move |x, _y| {
                let x2 = x * x;
                let inner = E::GELU_K * (x + E::GELU_C * x * x2);
                let t = inner.tanh();
                let sech2 = E::ONE - t * t;
                E::HALF * (E::ONE + t)
                    + E::HALF * x * sech2 * E::GELU_K * (E::ONE + E::THREE * E::GELU_C * x2)
            },
        )
    }

    pub fn gt_scalar(&self, rhs: E) -> Result<Mask<S>> {
        self.compare_scalar(rhs, |a, b| a > b)
    }

    pub fn ge_scalar(&self, rhs: E) -> Result<Mask<S>> {
        self.compare_scalar(rhs, |a, b| a >= b)
    }

    pub fn lt_scalar(&self, rhs: E) -> Result<Mask<S>> {
        self.compare_scalar(rhs, |a, b| a < b)
    }

    pub fn le_scalar(&self, rhs: E) -> Result<Mask<S>> {
        self.compare_scalar(rhs, |a, b| a <= b)
    }

    pub fn eq_scalar(&self, rhs: E) -> Result<Mask<S>> {
        self.compare_scalar(rhs, |a, b| a == b)
    }

    pub fn masked_fill(&self, mask: &Mask<S>, value: E) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(ShapeError::LengthMismatch {
                expected: self.numel(),
                found: mask.values().len(),
            }
            .into());
        }
        let values = self
            .to_vec()?
            .into_iter()
            .zip(mask.values())
            .map(|(x, &m)| if m { value } else { x })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        let mask_values = mask.values().to_vec();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(&mask_values)
                .map(|(g, &m)| if m { E::ZERO } else { g })
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    pub fn where_mask(&self, mask: &Mask<S>, other: &Self) -> Result<Self> {
        self.ensure_same_device(other, "where_mask")?;
        if self.shape() != mask.shape() || other.shape() != mask.shape() {
            return Err(ShapeError::LengthMismatch {
                expected: self.numel(),
                found: mask.values().len(),
            }
            .into());
        }
        let lhs_values = self.to_vec()?;
        let rhs_values = other.to_vec()?;
        let values = lhs_values
            .into_iter()
            .zip(rhs_values)
            .zip(mask.values())
            .map(|((a, b), &m)| if m { a } else { b })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = other.raw().clone();
        let mask_values = mask.values().to_vec();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(other)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let lhs_grad = grad_values
                    .iter()
                    .zip(&mask_values)
                    .map(|(&g, &m)| if m { g } else { E::ZERO })
                    .collect();
                let rhs_grad = grad_values
                    .into_iter()
                    .zip(&mask_values)
                    .map(|(g, &m)| if m { E::ZERO } else { g })
                    .collect();
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
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

    fn unary_map(
        &self,
        forward: impl Fn(E) -> E + Copy + Send + Sync + 'static,
        backward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        let input_values = self.to_vec()?;
        let output_values = input_values
            .iter()
            .copied()
            .map(forward)
            .collect::<Vec<_>>();
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            output_values.clone(),
            self.shape().clone(),
        )?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(
                    input_values
                        .iter()
                        .copied()
                        .zip(output_values.iter().copied()),
                )
                .map(|(g, (x, y))| g * backward(x, y))
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    fn compare_scalar(&self, rhs: E, compare: impl Fn(E, E) -> bool) -> Result<Mask<S>> {
        Mask::from_vec_with_shape(
            self.to_vec()?
                .into_iter()
                .map(|value| compare(value, rhs))
                .collect(),
            self.shape().clone(),
        )
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

    pub fn stack0<R>(&self, rhs: &Tensor<D1<R>, E, B>) -> Result<Tensor<D2<C<2>, A>, E, B>>
    where
        R: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "stack0")?;
        let len = self.shape().dims()[0];
        let rhs_len = rhs.shape().dims()[0];
        bind_and_check(
            "stack0",
            [
                (DimEntry::of::<A>(0, 0), len),
                (DimEntry::of::<R>(1, 0), rhs_len),
            ],
        )?;
        if len != rhs_len {
            return Err(ShapeError::DimMismatch {
                op: "stack0",
                operand: 1,
                axis: 0,
                expected: len,
                found: rhs_len,
            }
            .into());
        }
        let mut values = self.to_vec()?;
        values.extend(rhs.to_vec()?);
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([2, len]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<C<2>, A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let values = grad.to_vec()?;
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, values[..len].to_vec())?),
                    Some(raw_from_vec_like(&rhs_raw, values[len..].to_vec())?),
                ])
            },
        )
    }

    pub fn unsqueeze0(&self) -> Result<Tensor<D2<C<1>, A>, E, B>> {
        self.reshape_with_shape::<D2<C<1>, A>>([1, self.shape().dims()[0]])
    }

    pub fn unsqueeze1(&self) -> Result<Tensor<D2<A, C<1>>, E, B>> {
        self.reshape_with_shape::<D2<A, C<1>>>([self.shape().dims()[0], 1])
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
                let mut bias_grad = vec![E::ZERO; cols];
                for row in 0..rows {
                    for col in 0..cols {
                        bias_grad[col] += grad_values[row * cols + col];
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, grad_values)?),
                    Some(raw_from_vec_like(&rhs_raw, bias_grad)?),
                ])
            },
        )
    }

    pub fn sub_row(&self, rhs: &Tensor<D1<N>, E, B>) -> Result<Self> {
        self.broadcast_row(rhs, "sub_row", |a, b| a - b, |_a, _b, g| g, |_a, _b, g| -g)
    }

    pub fn mul_row(&self, rhs: &Tensor<D1<N>, E, B>) -> Result<Self> {
        self.broadcast_row(
            rhs,
            "mul_row",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
        )
    }

    pub fn div_row(&self, rhs: &Tensor<D1<N>, E, B>) -> Result<Self> {
        self.broadcast_row(
            rhs,
            "div_row",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
        )
    }

    pub fn add_col(&self, rhs: &Tensor<D1<A>, E, B>) -> Result<Self> {
        self.broadcast_col(rhs, "add_col", |a, b| a + b, |_a, _b, g| g, |_a, _b, g| g)
    }

    pub fn sub_col(&self, rhs: &Tensor<D1<A>, E, B>) -> Result<Self> {
        self.broadcast_col(rhs, "sub_col", |a, b| a - b, |_a, _b, g| g, |_a, _b, g| -g)
    }

    pub fn mul_col(&self, rhs: &Tensor<D1<A>, E, B>) -> Result<Self> {
        self.broadcast_col(
            rhs,
            "mul_col",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
        )
    }

    pub fn div_col(&self, rhs: &Tensor<D1<A>, E, B>) -> Result<Self> {
        self.broadcast_col(
            rhs,
            "div_col",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
        )
    }

    /// Reduces rows and returns a rank-1 tensor. This uses `keepdim = false`;
    /// callers can re-expand with explicit broadcasts.
    pub fn sum_axis0(&self) -> Result<Tensor<D1<N>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let values = self.to_vec()?;
        let mut out = vec![E::ZERO; cols];
        for row in 0..rows {
            for col in 0..cols {
                out[col] += values[row * cols + col];
            }
        }
        let raw = RawTensor::from_vec_on(self.device().clone(), out, Shape::known([cols]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.to_vec()?;
                let mut values = Vec::with_capacity(rows * cols);
                for _ in 0..rows {
                    values.extend(seed.iter().copied());
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    /// Reduces columns and returns a rank-1 tensor. This uses `keepdim = false`;
    /// callers can re-expand with explicit broadcasts.
    pub fn sum_axis1(&self) -> Result<Tensor<D1<A>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let values = self.to_vec()?;
        let mut out = vec![E::ZERO; rows];
        for row in 0..rows {
            for col in 0..cols {
                out[row] += values[row * cols + col];
            }
        }
        let raw = RawTensor::from_vec_on(self.device().clone(), out, Shape::known([rows]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.to_vec()?;
                let mut values = Vec::with_capacity(rows * cols);
                for &g in &seed {
                    values.extend(std::iter::repeat_n(g, cols));
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    pub fn mean_axis0(&self) -> Result<Tensor<D1<N>, E, B>> {
        self.sum_axis0()?
            .div_scalar(E::from_usize(self.shape().dims()[0]))
    }

    pub fn mean_axis1(&self) -> Result<Tensor<D1<A>, E, B>> {
        self.sum_axis1()?
            .div_scalar(E::from_usize(self.shape().dims()[1]))
    }

    /// Reduces columns with max. Backward splits gradient evenly across tied
    /// maxima instead of selecting the first maximum.
    pub fn max_axis1(&self) -> Result<Tensor<D1<A>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let values = self.to_vec()?;
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let mut max = values[start];
            for &value in &values[start + 1..start + cols] {
                if value > max {
                    max = value;
                }
            }
            out.push(max);
        }
        let raw = RawTensor::from_vec_on(self.device().clone(), out.clone(), Shape::known([rows]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.to_vec()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                for row in 0..rows {
                    let start = row * cols;
                    let count = values[start..start + cols]
                        .iter()
                        .filter(|&&value| value == out[row])
                        .count();
                    let each = seed[row] / E::from_usize(count);
                    for col in 0..cols {
                        if values[start + col] == out[row] {
                            grad_values[start + col] = each;
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn logsumexp_axis1(&self) -> Result<Tensor<D1<A>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let values = self.to_vec()?;
        let mut softmax = vec![E::ZERO; rows * cols];
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let max = row_max(&values[start..start + cols]);
            let mut sum = E::ZERO;
            for col in 0..cols {
                let exp = (values[start + col] - max).exp();
                sum += exp;
                softmax[start + col] = exp;
            }
            for col in 0..cols {
                softmax[start + col] = softmax[start + col] / sum;
            }
            out.push(max + sum.ln());
        }
        let raw = RawTensor::from_vec_on(self.device().clone(), out, Shape::known([rows]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.to_vec()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                for row in 0..rows {
                    for col in 0..cols {
                        grad_values[row * cols + col] = seed[row] * softmax[row * cols + col];
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn softmax_axis1(&self) -> Result<Self> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let values = stable_row_softmax(&self.to_vec()?, rows, cols);
        let raw =
            RawTensor::from_vec_on(self.device().clone(), values.clone(), self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = grad.to_vec()?;
            let mut out = vec![E::ZERO; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let dot = (0..cols).fold(E::ZERO, |acc, col| {
                    acc + grad[start + col] * values[start + col]
                });
                for col in 0..cols {
                    out[start + col] = values[start + col] * (grad[start + col] - dot);
                }
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, out)?)])
        })
    }

    pub fn log_softmax_axis1(&self) -> Result<Self> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let input = self.to_vec()?;
        let softmax = stable_row_softmax(&input, rows, cols);
        let values = stable_row_log_softmax(&input, rows, cols);
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = grad.to_vec()?;
            let mut out = vec![E::ZERO; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let row_sum = (0..cols).fold(E::ZERO, |acc, col| acc + grad[start + col]);
                for col in 0..cols {
                    out[start + col] = grad[start + col] - softmax[start + col] * row_sum;
                }
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, out)?)])
        })
    }

    pub fn cross_entropy(&self, targets: &[usize]) -> Result<Scalar<E, B>> {
        self.cross_entropy_ignore_index(targets, usize::MAX)
    }

    pub fn cross_entropy_ignore_index(
        &self,
        targets: &[usize],
        ignore_index: usize,
    ) -> Result<Scalar<E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if targets.len() != rows {
            return Err(ShapeError::LengthMismatch {
                expected: rows,
                found: targets.len(),
            }
            .into());
        }
        for &target in targets {
            if target == ignore_index {
                continue;
            }
            if target >= cols {
                return Err(DataError::IndexOutOfBounds {
                    index: target,
                    len: cols,
                }
                .into());
            }
        }
        let input = self.to_vec()?;
        let softmax = stable_row_softmax(&input, rows, cols);
        let log_probs = stable_row_log_softmax(&input, rows, cols);
        let valid_count = targets
            .iter()
            .filter(|&&target| target != ignore_index)
            .count();
        let loss_sum = targets
            .iter()
            .enumerate()
            .filter(|&(_, &target)| target != ignore_index)
            .fold(E::ZERO, |acc, (row, &target)| {
                acc - log_probs[row * cols + target]
            });
        let scale = E::from_usize(valid_count);
        let loss = if valid_count == 0 {
            E::ZERO
        } else {
            loss_sum / scale
        };
        let raw = RawTensor::from_vec_on(self.device().clone(), vec![loss], Shape::known([]))?;
        let input_raw = self.raw().clone();
        let targets = targets.to_vec();
        Tensor::<D0, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            let mut values = vec![E::ZERO; rows * cols];
            if valid_count == 0 {
                return Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)]);
            }
            for row in 0..rows {
                if targets[row] == ignore_index {
                    continue;
                }
                for col in 0..cols {
                    values[row * cols + col] = softmax[row * cols + col];
                }
            }
            for row in 0..rows {
                if targets[row] == ignore_index {
                    continue;
                }
                values[row * cols + targets[row]] -= E::ONE;
            }
            for value in &mut values {
                *value = *value * seed / scale;
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
        })
    }

    pub fn select_row(&self, row: usize) -> Result<Tensor<D1<N>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if row >= rows {
            return Err(DataError::IndexOutOfBounds {
                index: row,
                len: rows,
            }
            .into());
        }
        let values = self.to_vec()?[row * cols..(row + 1) * cols].to_vec();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([cols]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let mut values = vec![E::ZERO; rows * cols];
                values[row * cols..(row + 1) * cols].copy_from_slice(&grad.to_vec()?);
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    pub fn index_select_rows<T>(&self, indices: &[usize]) -> Result<Tensor<D2<T, N>, E, B>>
    where
        T: DimSpec,
    {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        bind_and_check(
            "index_select_rows",
            [(DimEntry::of::<T>(0, 0), indices.len())],
        )?;
        for &index in indices {
            if index >= rows {
                return Err(DataError::IndexOutOfBounds { index, len: rows }.into());
            }
        }
        let input = self.to_vec()?;
        let mut values = Vec::with_capacity(indices.len() * cols);
        for &index in indices {
            values.extend_from_slice(&input[index * cols..(index + 1) * cols]);
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([indices.len(), cols]),
        )?;
        let input_raw = self.raw().clone();
        let indices = indices.to_vec();
        Tensor::<D2<T, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.to_vec()?;
                let mut values = vec![E::ZERO; rows * cols];
                for (out_row, &source_row) in indices.iter().enumerate() {
                    for col in 0..cols {
                        values[source_row * cols + col] =
                            values[source_row * cols + col] + grad[out_row * cols + col];
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    fn broadcast_row(
        &self,
        rhs: &Tensor<D1<N>, E, B>,
        op: &'static str,
        forward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
        lhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
        rhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)?;
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let rhs_cols = rhs.shape().dims()[0];
        bind_and_check(
            op,
            [
                (DimEntry::of::<A>(0, 0), rows),
                (DimEntry::of::<N>(0, 1), cols),
                (DimEntry::of::<N>(1, 0), rhs_cols),
            ],
        )?;
        if rhs_cols != cols {
            return Err(ShapeError::DimMismatch {
                op,
                operand: 1,
                axis: 0,
                expected: cols,
                found: rhs_cols,
            }
            .into());
        }
        let lhs_values = self.to_vec()?;
        let rhs_values = rhs.to_vec()?;
        let values = lhs_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| forward(value, rhs_values[idx % cols]))
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let mut lhs_grad = Vec::with_capacity(rows * cols);
                let mut rhs_grad = vec![E::ZERO; cols];
                for idx in 0..rows * cols {
                    let col = idx % cols;
                    let g = grad_values[idx];
                    lhs_grad.push(lhs_backward(lhs_values[idx], rhs_values[col], g));
                    rhs_grad[col] =
                        rhs_grad[col] + rhs_backward(lhs_values[idx], rhs_values[col], g);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }

    fn broadcast_col(
        &self,
        rhs: &Tensor<D1<A>, E, B>,
        op: &'static str,
        forward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
        lhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
        rhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)?;
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        let rhs_rows = rhs.shape().dims()[0];
        bind_and_check(
            op,
            [
                (DimEntry::of::<A>(0, 0), rows),
                (DimEntry::of::<N>(0, 1), cols),
                (DimEntry::of::<A>(1, 0), rhs_rows),
            ],
        )?;
        if rhs_rows != rows {
            return Err(ShapeError::DimMismatch {
                op,
                operand: 1,
                axis: 0,
                expected: rows,
                found: rhs_rows,
            }
            .into());
        }
        let lhs_values = self.to_vec()?;
        let rhs_values = rhs.to_vec()?;
        let values = lhs_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| forward(value, rhs_values[idx / cols]))
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let mut lhs_grad = Vec::with_capacity(rows * cols);
                let mut rhs_grad = vec![E::ZERO; rows];
                for idx in 0..rows * cols {
                    let row = idx / cols;
                    let g = grad_values[idx];
                    lhs_grad.push(lhs_backward(lhs_values[idx], rhs_values[row], g));
                    rhs_grad[row] =
                        rhs_grad[row] + rhs_backward(lhs_values[idx], rhs_values[row], g);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }
}

impl<N, E, B> Tensor<D2<C<1>, N>, E, B>
where
    N: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn squeeze0(&self) -> Result<Tensor<D1<N>, E, B>> {
        self.reshape_with_shape::<D1<N>>([self.shape().dims()[1]])
    }
}

impl<Batch, M, K, E, B> Tensor<D3<Batch, M, K>, E, B>
where
    Batch: DimSpec,
    M: DimSpec,
    K: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn transpose_last2(&self) -> Result<Tensor<D3<Batch, K, M>, E, B>> {
        let raw = self
            .raw()
            .view_with_layout(self.layout().transpose_axes(1, 2)?)?;
        Tensor::<D3<Batch, K, M>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            |grad| {
                Ok(vec![Some(
                    grad.view_with_layout(grad.layout().transpose_axes(1, 2)?)?,
                )])
            },
        )
    }

    pub fn bmm<N>(
        &self,
        rhs: &Tensor<D3<Batch, K, N>, E, B>,
    ) -> Result<Tensor<D3<Batch, M, N>, E, B>>
    where
        N: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "bmm")?;
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let batch = lhs_dims[0];
        let m = lhs_dims[1];
        let k = lhs_dims[2];
        let rhs_batch = rhs_dims[0];
        let rhs_k = rhs_dims[1];
        let n = rhs_dims[2];
        bind_and_check(
            "bmm",
            [
                (DimEntry::of::<Batch>(0, 0), batch),
                (DimEntry::of::<M>(0, 1), m),
                (DimEntry::of::<K>(0, 2), k),
                (DimEntry::of::<Batch>(1, 0), rhs_batch),
                (DimEntry::of::<K>(1, 1), rhs_k),
                (DimEntry::of::<N>(1, 2), n),
            ],
        )?;
        if batch != rhs_batch {
            return Err(ShapeError::DimMismatch {
                op: "bmm",
                operand: 1,
                axis: 0,
                expected: batch,
                found: rhs_batch,
            }
            .into());
        }
        if k != rhs_k {
            return Err(ShapeError::DimMismatch {
                op: "bmm",
                operand: 1,
                axis: 1,
                expected: k,
                found: rhs_k,
            }
            .into());
        }
        let lhs_values = self.to_vec()?;
        let rhs_values = rhs.to_vec()?;
        let values = bmm_values(&lhs_values, &rhs_values, batch, m, k, n);
        let raw =
            RawTensor::from_vec_on(self.device().clone(), values, Shape::known([batch, m, n]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D3<Batch, M, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad = grad.to_vec()?;
                let mut lhs_grad = vec![E::ZERO; batch * m * k];
                let mut rhs_grad = vec![E::ZERO; batch * k * n];
                for b in 0..batch {
                    let lhs_base = b * m * k;
                    let rhs_base = b * k * n;
                    let grad_base = b * m * n;
                    for i in 0..m {
                        for kk in 0..k {
                            let mut acc = E::ZERO;
                            for j in 0..n {
                                acc +=
                                    grad[grad_base + i * n + j] * rhs_values[rhs_base + kk * n + j];
                            }
                            lhs_grad[lhs_base + i * k + kk] = acc;
                        }
                    }
                    for kk in 0..k {
                        for j in 0..n {
                            let mut acc = E::ZERO;
                            for i in 0..m {
                                acc +=
                                    lhs_values[lhs_base + i * k + kk] * grad[grad_base + i * n + j];
                            }
                            rhs_grad[rhs_base + kk * n + j] = acc;
                        }
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }
}

impl<A, BDim, Cc, E, BackendT> Tensor<D3<A, BDim, Cc>, E, BackendT>
where
    A: DimSpec,
    BDim: DimSpec,
    Cc: DimSpec,
    E: FloatDType,
    BackendT: Backend<E>,
{
    pub fn softmax_axis2(&self) -> Result<Self> {
        let dims = self.shape().dims();
        let rows = dims[0] * dims[1];
        let cols = dims[2];
        let values = stable_row_softmax(&self.to_vec()?, rows, cols);
        let raw =
            RawTensor::from_vec_on(self.device().clone(), values.clone(), self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = grad.to_vec()?;
            let mut out = vec![E::ZERO; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let dot = (0..cols).fold(E::ZERO, |acc, col| {
                    acc + grad[start + col] * values[start + col]
                });
                for col in 0..cols {
                    out[start + col] = values[start + col] * (grad[start + col] - dot);
                }
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, out)?)])
        })
    }

    pub fn add_last_dim(&self, rhs: &Tensor<D1<Cc>, E, BackendT>) -> Result<Self> {
        self.broadcast_last_dim(
            rhs,
            "add_last_dim",
            |a, b| a + b,
            |_a, _b, g| g,
            |_a, _b, g| g,
        )
    }

    pub fn sub_last_dim(&self, rhs: &Tensor<D1<Cc>, E, BackendT>) -> Result<Self> {
        self.broadcast_last_dim(
            rhs,
            "sub_last_dim",
            |a, b| a - b,
            |_a, _b, g| g,
            |_a, _b, g| -g,
        )
    }

    pub fn mul_last_dim(&self, rhs: &Tensor<D1<Cc>, E, BackendT>) -> Result<Self> {
        self.broadcast_last_dim(
            rhs,
            "mul_last_dim",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
        )
    }

    pub fn div_last_dim(&self, rhs: &Tensor<D1<Cc>, E, BackendT>) -> Result<Self> {
        self.broadcast_last_dim(
            rhs,
            "div_last_dim",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
        )
    }

    fn broadcast_last_dim(
        &self,
        rhs: &Tensor<D1<Cc>, E, BackendT>,
        op: &'static str,
        forward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
        lhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
        rhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure_same_device::<E, BackendT>(self.device(), rhs.device(), op)?;
        let dims = self.shape().dims();
        let outer = dims[0] * dims[1];
        let cols = dims[2];
        let rhs_cols = rhs.shape().dims()[0];
        if rhs_cols != cols {
            return Err(ShapeError::DimMismatch {
                op,
                operand: 1,
                axis: 0,
                expected: cols,
                found: rhs_cols,
            }
            .into());
        }
        let lhs_values = self.to_vec()?;
        let rhs_values = rhs.to_vec()?;
        let values = lhs_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| forward(value, rhs_values[idx % cols]))
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let mut lhs_grad = Vec::with_capacity(outer * cols);
                let mut rhs_grad = vec![E::ZERO; cols];
                for idx in 0..outer * cols {
                    let col = idx % cols;
                    let g = grad_values[idx];
                    lhs_grad.push(lhs_backward(lhs_values[idx], rhs_values[col], g));
                    rhs_grad[col] =
                        rhs_grad[col] + rhs_backward(lhs_values[idx], rhs_values[col], g);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }
}

impl<A, BDim, Cc, Dd, E, BackendT> Tensor<D4<A, BDim, Cc, Dd>, E, BackendT>
where
    A: DimSpec,
    BDim: DimSpec,
    Cc: DimSpec,
    Dd: DimSpec,
    E: FloatDType,
    BackendT: Backend<E>,
{
    pub fn transpose_axes12(&self) -> Result<Tensor<D4<A, Cc, BDim, Dd>, E, BackendT>> {
        let raw = self
            .raw()
            .view_with_layout(self.layout().transpose_axes(1, 2)?)?;
        Tensor::<D4<A, Cc, BDim, Dd>, E, BackendT>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            |grad| {
                Ok(vec![Some(
                    grad.view_with_layout(grad.layout().transpose_axes(1, 2)?)?,
                )])
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

fn row_max<E: FloatDType>(values: &[E]) -> E {
    let mut max = values[0];
    for &value in &values[1..] {
        if value > max {
            max = value;
        }
    }
    max
}

fn stable_row_softmax<E: FloatDType>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut out = vec![E::ZERO; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let mut sum = E::ZERO;
        for col in 0..cols {
            let exp = (values[start + col] - max).exp();
            sum += exp;
            out[start + col] = exp;
        }
        for col in 0..cols {
            out[start + col] = out[start + col] / sum;
        }
    }
    out
}

fn stable_row_log_softmax<E: FloatDType>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut out = vec![E::ZERO; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let sum = (0..cols)
            .map(|col| (values[start + col] - max).exp())
            .fold(E::ZERO, |acc, value| acc + value);
        let logsumexp = max + sum.ln();
        for col in 0..cols {
            out[start + col] = values[start + col] - logsumexp;
        }
    }
    out
}

fn bmm_values<E: FloatDType>(
    lhs: &[E],
    rhs: &[E],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<E> {
    let mut out = vec![E::ZERO; batch * m * n];
    for b in 0..batch {
        let lhs_base = b * m * k;
        let rhs_base = b * k * n;
        let out_base = b * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = E::ZERO;
                for kk in 0..k {
                    acc += lhs[lhs_base + i * k + kk] * rhs[rhs_base + kk * n + j];
                }
                out[out_base + i * n + j] = acc;
            }
        }
    }
    out
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
