use super::{RawTensor, Scalar, Tensor, TensorInner};
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::{Result, ShapeError};
use crate::shape::ShapeSpec;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

pub fn no_grad() -> NoGradGuard {
    let previous = GRAD_ENABLED.with(|enabled| {
        let previous = enabled.get();
        enabled.set(false);
        previous
    });
    NoGradGuard { previous }
}

#[must_use]
pub struct NoGradGuard {
    previous: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|enabled| enabled.set(self.previous));
    }
}

pub(crate) struct AutogradMeta<E, B>
where
    E: DType,
    B: Backend<E>,
{
    id: u64,
    requires_grad: AtomicBool,
    is_leaf: bool,
    grad: Mutex<Option<RawTensor<E, B>>>,
    grad_fn: Mutex<Option<Arc<GradFn<E, B>>>>,
}

impl<E, B> AutogradMeta<E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub(crate) fn leaf() -> Self {
        Self::new(true)
    }

    pub(crate) fn non_leaf() -> Self {
        Self::new(false)
    }

    fn new(is_leaf: bool) -> Self {
        Self {
            id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            requires_grad: AtomicBool::new(false),
            is_leaf,
            grad: Mutex::new(None),
            grad_fn: Mutex::new(None),
        }
    }
}

#[derive(Clone)]
pub(crate) struct AnyTensor<E, B>
where
    E: DType,
    B: Backend<E>,
{
    inner: Arc<TensorInner<E, B>>,
}

impl<E, B> AnyTensor<E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub(crate) fn from_shape<S>(tensor: &Tensor<S, E, B>) -> Self
    where
        S: ShapeSpec,
    {
        Self {
            inner: Arc::clone(&tensor.inner),
        }
    }
}

type Backward<E, B> =
    dyn Fn(&RawTensor<E, B>) -> Result<Vec<Option<RawTensor<E, B>>>> + Send + Sync;

pub(crate) struct GradFn<E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub(crate) parents: Vec<AnyTensor<E, B>>,
    pub(crate) backward: Box<Backward<E, B>>,
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    pub(crate) fn autograd_output(
        raw: RawTensor<E, B>,
        parents: Vec<AnyTensor<E, B>>,
        backward: impl Fn(&RawTensor<E, B>) -> Result<Vec<Option<RawTensor<E, B>>>>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        let requires_grad = is_grad_enabled()
            && parents
                .iter()
                .any(|parent| parent.inner.autograd.requires_grad.load(Ordering::Relaxed));
        let out = Self::from_raw_non_leaf(raw)?;
        if requires_grad {
            out.inner
                .autograd
                .requires_grad
                .store(true, Ordering::Relaxed);
            *out.inner.autograd.grad_fn.lock().expect("grad_fn poisoned") =
                Some(Arc::new(GradFn {
                    parents,
                    backward: Box::new(backward),
                }));
        }
        Ok(out)
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn requires_grad(&self) -> bool {
        self.inner.autograd.requires_grad.load(Ordering::Relaxed)
    }

    pub fn with_requires_grad(self, requires_grad: bool) -> Self {
        self.set_requires_grad(requires_grad);
        self
    }

    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.inner
            .autograd
            .requires_grad
            .store(requires_grad, Ordering::Relaxed);
    }

    pub fn grad(&self) -> Option<Self> {
        if !self.inner.autograd.is_leaf {
            return None;
        }
        let grad = self.inner.autograd.grad.lock().ok()?.as_ref()?.clone();
        Self::from_raw(grad).ok()
    }

    pub fn zero_grad(&self) {
        if let Ok(mut grad) = self.inner.autograd.grad.lock() {
            *grad = None;
        }
    }

    pub(crate) fn set_grad_data(&self, grad: Vec<E>) -> Result<()> {
        let raw = RawTensor::from_vec_on(self.device().clone(), grad, self.shape().clone())?;
        *self.inner.autograd.grad.lock().expect("grad slot poisoned") = Some(raw);
        Ok(())
    }

    pub fn detach(&self) -> Self {
        Self::from_raw(self.raw().clone()).expect("detached tensor preserves validated shape")
    }

    pub fn backward_with(&self, seed: &Self) -> Result<()> {
        if self.shape() != seed.shape() {
            return Err(ShapeError::LengthMismatch {
                expected: self.numel(),
                found: seed.numel(),
            }
            .into());
        }
        self.run_backward(seed.raw().clone())
    }

    fn run_backward(&self, seed: RawTensor<E, B>) -> Result<()> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(&AnyTensor::from_shape(self), &mut visited, &mut topo);

        let mut grads = HashMap::new();
        grads.insert(self.inner.autograd.id, seed);

        for node in topo.into_iter().rev() {
            let Some(grad) = grads.remove(&node.inner.autograd.id) else {
                continue;
            };

            if node.inner.autograd.is_leaf
                && node.inner.autograd.requires_grad.load(Ordering::Relaxed)
            {
                accumulate_slot(&node.inner, &grad)?;
            }

            let Some(grad_fn) = node
                .inner
                .autograd
                .grad_fn
                .lock()
                .expect("grad_fn poisoned")
                .clone()
            else {
                continue;
            };
            let parent_grads = (grad_fn.backward)(&grad)?;
            for (parent, parent_grad) in grad_fn.parents.iter().zip(parent_grads) {
                let Some(parent_grad) = parent_grad else {
                    continue;
                };
                if !parent.inner.autograd.requires_grad.load(Ordering::Relaxed) {
                    continue;
                }
                accumulate_map(&mut grads, parent.inner.autograd.id, parent_grad)?;
            }
        }
        Ok(())
    }
}

impl<E, B> Scalar<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn backward(&self) -> Result<()> {
        let raw = RawTensor::ones_on(self.device().clone(), crate::shape::Shape::known([]))?;
        self.run_backward(raw)
    }
}

fn build_topo<E, B>(
    node: &AnyTensor<E, B>,
    visited: &mut HashSet<u64>,
    topo: &mut Vec<AnyTensor<E, B>>,
) where
    E: FloatDType,
    B: Backend<E>,
{
    if !visited.insert(node.inner.autograd.id) {
        return;
    }
    if let Some(grad_fn) = node
        .inner
        .autograd
        .grad_fn
        .lock()
        .expect("grad_fn poisoned")
        .clone()
    {
        for parent in &grad_fn.parents {
            build_topo(parent, visited, topo);
        }
    }
    topo.push(node.clone());
}

fn accumulate_slot<E, B>(inner: &TensorInner<E, B>, grad: &RawTensor<E, B>) -> Result<()>
where
    E: FloatDType,
    B: Backend<E>,
{
    let mut slot = inner.autograd.grad.lock().expect("grad slot poisoned");
    match slot.take() {
        Some(existing) => *slot = Some(raw_add(&existing, grad)?),
        None => *slot = Some(grad.clone()),
    }
    Ok(())
}

fn accumulate_map<E, B>(
    grads: &mut HashMap<u64, RawTensor<E, B>>,
    id: u64,
    grad: RawTensor<E, B>,
) -> Result<()>
where
    E: FloatDType,
    B: Backend<E>,
{
    match grads.remove(&id) {
        Some(existing) => {
            grads.insert(id, raw_add(&existing, &grad)?);
        }
        None => {
            grads.insert(id, grad);
        }
    }
    Ok(())
}

pub(crate) fn raw_add<E, B>(lhs: &RawTensor<E, B>, rhs: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: DType,
    B: Backend<E>,
{
    let lhs_data = lhs.to_vec()?;
    let rhs_data = rhs.to_vec()?;
    let storage = B::add(
        lhs.device(),
        &B::from_vec(lhs.device(), lhs_data).map_err(crate::error::Error::backend)?,
        &B::from_vec(rhs.device(), rhs_data).map_err(crate::error::Error::backend)?,
        lhs.numel(),
    )
    .map_err(crate::error::Error::backend)?;
    RawTensor::from_storage_on(lhs.device().clone(), storage, lhs.shape().clone())
}

pub(crate) fn raw_from_vec_like<E, B>(
    like: &RawTensor<E, B>,
    data: Vec<E>,
) -> Result<RawTensor<E, B>>
where
    E: DType,
    B: Backend<E>,
{
    RawTensor::from_vec_on(like.device().clone(), data, like.shape().clone())
}

pub(crate) fn raw_full_like<E, B>(like: &RawTensor<E, B>, value: E) -> Result<RawTensor<E, B>>
where
    E: DType,
    B: Backend<E>,
{
    raw_from_vec_like(like, vec![value; like.numel()])
}

pub(crate) fn raw_neg<E, B>(input: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    raw_from_vec_like(input, input.to_vec()?.into_iter().map(|x| -x).collect())
}

pub(crate) fn raw_mul<E, B>(lhs: &RawTensor<E, B>, rhs: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    raw_from_vec_like(
        lhs,
        lhs.to_vec()?
            .into_iter()
            .zip(rhs.to_vec()?)
            .map(|(a, b)| a * b)
            .collect(),
    )
}

pub(crate) fn raw_div<E, B>(lhs: &RawTensor<E, B>, rhs: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    raw_from_vec_like(
        lhs,
        lhs.to_vec()?
            .into_iter()
            .zip(rhs.to_vec()?)
            .map(|(a, b)| a / b)
            .collect(),
    )
}

pub(crate) fn raw_mul_scalar<E, B>(input: &RawTensor<E, B>, rhs: E) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    raw_from_vec_like(
        input,
        input.to_vec()?.into_iter().map(|x| x * rhs).collect(),
    )
}

pub(crate) fn raw_div_scalar<E, B>(input: &RawTensor<E, B>, rhs: E) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    raw_from_vec_like(
        input,
        input.to_vec()?.into_iter().map(|x| x / rhs).collect(),
    )
}

#[cfg(test)]
mod tests {
    use crate::shape::{AnyDim, C, D1, Shape};
    use crate::{Tensor1D, Tensor2D};

    #[test]
    fn autograd_constructors_and_grad_mode() {
        let x = Tensor1D::<2>::from_vec(vec![1.0, 2.0]).unwrap();
        assert!(!x.requires_grad());

        let x = x.with_requires_grad(true);
        assert!(x.requires_grad());

        let y = x.mul_scalar(2.0).unwrap();
        assert!(y.requires_grad());
        assert!(y.grad().is_none());

        {
            let _guard = super::no_grad();
            assert!(!super::is_grad_enabled());
            let z = x.mul_scalar(3.0).unwrap();
            assert!(!z.requires_grad());
        }
        assert!(super::is_grad_enabled());

        let detached = y.detach();
        assert!(!detached.requires_grad());
        assert_eq!(detached.to_vec().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn scalar_backward_accumulates_and_zero_grad_clears() {
        let x = Tensor1D::<2>::from_vec(vec![2.0, 3.0])
            .unwrap()
            .with_requires_grad(true);
        let loss = x.mul(&x).unwrap().sum().unwrap();

        assert!(x.grad().is_none());
        loss.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![4.0, 6.0]);

        loss.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![8.0, 12.0]);

        x.zero_grad();
        assert!(x.grad().is_none());
    }

    #[test]
    fn backward_with_supports_non_scalar_seeds() {
        let x = Tensor1D::<3>::from_vec(vec![1.0, 2.0, 3.0])
            .unwrap()
            .with_requires_grad(true);
        let y = x.mul_scalar(2.0).unwrap();
        let seed = Tensor1D::<3>::from_vec(vec![1.0, 10.0, 100.0]).unwrap();

        y.backward_with(&seed).unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![2.0, 20.0, 200.0]);
    }

    #[test]
    fn autograd_same_shape_and_scalar_formulas() {
        let x = Tensor1D::<2>::from_vec(vec![2.0, 4.0])
            .unwrap()
            .with_requires_grad(true);
        let y = Tensor1D::<2>::from_vec(vec![5.0, 8.0])
            .unwrap()
            .with_requires_grad(true);

        let loss = x
            .add(&y)
            .unwrap()
            .sub(&y)
            .unwrap()
            .mul(&y)
            .unwrap()
            .div(&x)
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sub_scalar(1.0)
            .unwrap()
            .mul_scalar(3.0)
            .unwrap()
            .div_scalar(2.0)
            .unwrap()
            .sum()
            .unwrap();
        loss.backward().unwrap();

        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(y.grad().unwrap().to_vec().unwrap(), vec![1.5, 1.5]);
    }

    #[test]
    fn autograd_matmul_formula() {
        let x = Tensor2D::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_requires_grad(true);
        let w = Tensor2D::<2, 2>::from_vec(vec![5.0, 6.0, 7.0, 8.0])
            .unwrap()
            .with_requires_grad(true);

        x.matmul(&w).unwrap().sum().unwrap().backward().unwrap();

        assert_eq!(
            x.grad().unwrap().to_vec().unwrap(),
            vec![11.0, 15.0, 11.0, 15.0]
        );
        assert_eq!(
            w.grad().unwrap().to_vec().unwrap(),
            vec![4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn autograd_view_and_cat_formulas() {
        let x = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_requires_grad(true);
        x.transpose().unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![1.0; 6]);
        x.zero_grad();

        x.transpose()
            .unwrap()
            .contiguous()
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![1.0; 6]);
        x.zero_grad();

        x.reshape1::<6>()
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![1.0; 6]);
        x.zero_grad();

        x.reshape_with_shape::<D1<AnyDim>>(Shape::known([6]))
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![1.0; 6]);
        x.zero_grad();

        x.flatten::<6>().unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().to_vec().unwrap(), vec![1.0; 6]);

        let a = Tensor1D::<2>::from_vec(vec![1.0, 2.0])
            .unwrap()
            .with_requires_grad(true);
        let b = Tensor1D::<3>::from_vec(vec![3.0, 4.0, 5.0])
            .unwrap()
            .with_requires_grad(true);
        a.cat::<C<3>, 5>(&b)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(a.grad().unwrap().to_vec().unwrap(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap().to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn finite_difference_matches_square_sum_gradient_f64() {
        let x = Tensor1D::<2, f64>::from_vec(vec![1.5, -2.0])
            .unwrap()
            .with_requires_grad(true);
        x.mul(&x).unwrap().sum().unwrap().backward().unwrap();
        let grad = x.grad().unwrap().to_vec().unwrap();

        let eps = 1e-6;
        for idx in 0..2 {
            let mut plus = [1.5, -2.0];
            plus[idx] += eps;
            let mut minus = [1.5, -2.0];
            minus[idx] -= eps;
            let f_plus: f64 = plus.iter().map(|v| v * v).sum();
            let f_minus: f64 = minus.iter().map(|v| v * v).sum();
            let numerical = (f_plus - f_minus) / (2.0 * eps);
            assert!((grad[idx] - numerical).abs() < 1e-8);
        }
    }
}
