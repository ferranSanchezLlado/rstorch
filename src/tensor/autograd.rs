//! Dynamic autograd engine for typed tensors.

use super::{Scalar, Tensor};
use crate::backend::Backend;
use crate::dtype::FloatElement;
use crate::shape::{D0, Shape};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub(super) type NodeId = u64;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub(super) fn next_node_id() -> NodeId {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Returns whether differentiable tensor operations should build graph edges.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Disables graph construction on the current thread until the returned guard is dropped.
pub fn no_grad() -> NoGradGuard {
    let previous = GRAD_ENABLED.with(|enabled| {
        let previous = enabled.get();
        enabled.set(false);
        previous
    });
    NoGradGuard { previous }
}

/// Runs a closure with graph construction disabled on the current thread.
pub fn with_no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = no_grad();
    f()
}

/// Restores the previous thread-local grad mode when dropped.
#[must_use = "no_grad is active only while the guard is kept alive"]
pub struct NoGradGuard {
    previous: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|enabled| enabled.set(self.previous));
    }
}

pub(super) struct GradFn<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    pub(super) parents: Vec<AnyTensor<E, B>>,
    pub(super) backward: Box<dyn Fn(&B::Storage) -> Vec<B::Storage> + Send + Sync>,
}

#[derive(Clone)]
pub(super) struct AnyTensor<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    id: NodeId,
    device: B::Device,
    requires_grad: bool,
    is_leaf: bool,
    grad: Arc<Mutex<Option<B::Storage>>>,
    grad_fn: Option<Arc<GradFn<E, B>>>,
}

impl<E, B> AnyTensor<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    pub(super) fn from_tensor<S>(tensor: &Tensor<S, E, B>) -> Self
    where
        S: Shape,
    {
        Self {
            id: tensor.inner.id,
            device: tensor.inner.device.clone(),
            requires_grad: tensor.inner.requires_grad,
            is_leaf: tensor.inner.is_leaf,
            grad: Arc::clone(&tensor.inner.grad),
            grad_fn: tensor.inner.grad_fn.clone(),
        }
    }
}

pub(super) fn should_track_grad(parents_require_grad: bool) -> bool {
    parents_require_grad && is_grad_enabled()
}

pub(super) fn add_storages<E, B>(
    device: &B::Device,
    lhs: &B::Storage,
    rhs: &B::Storage,
) -> B::Storage
where
    E: FloatElement,
    B: Backend<E>,
{
    let lhs = B::to_vec(lhs);
    let rhs = B::to_vec(rhs);
    assert_eq!(lhs.len(), rhs.len(), "gradient shape mismatch");

    let data = lhs
        .into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();
    B::from_vec(device, data)
}

fn run_backward<E, B>(output: AnyTensor<E, B>, seed: B::Storage)
where
    E: FloatElement,
    B: Backend<E>,
{
    let mut visited = HashSet::new();
    let mut topo = Vec::new();
    build_topo(&output, &mut visited, &mut topo);

    let mut pending = HashMap::new();
    pending.insert(output.id, seed);

    for node in topo.into_iter().rev() {
        let Some(node_grad) = pending.remove(&node.id) else {
            continue;
        };

        if node.requires_grad && node.is_leaf {
            accumulate_leaf_grad::<E, B>(&node, &node_grad);
        }

        let Some(grad_fn) = &node.grad_fn else {
            continue;
        };

        let parent_grads = (grad_fn.backward)(&node_grad);
        debug_assert_eq!(parent_grads.len(), grad_fn.parents.len());

        for (parent, parent_grad) in grad_fn.parents.iter().zip(parent_grads) {
            if !parent.requires_grad {
                continue;
            }

            pending
                .entry(parent.id)
                .and_modify(|existing| {
                    *existing = add_storages::<E, B>(&parent.device, existing, &parent_grad);
                })
                .or_insert(parent_grad);
        }
    }
}

fn build_topo<E, B>(
    node: &AnyTensor<E, B>,
    visited: &mut HashSet<NodeId>,
    topo: &mut Vec<AnyTensor<E, B>>,
) where
    E: FloatElement,
    B: Backend<E>,
{
    if !visited.insert(node.id) {
        return;
    }

    if let Some(grad_fn) = &node.grad_fn {
        for parent in &grad_fn.parents {
            build_topo(parent, visited, topo);
        }
    }

    topo.push(node.clone());
}

fn accumulate_leaf_grad<E, B>(node: &AnyTensor<E, B>, grad: &B::Storage)
where
    E: FloatElement,
    B: Backend<E>,
{
    let mut slot = node.grad.lock().expect("gradient mutex poisoned");
    *slot = Some(match slot.as_ref() {
        Some(existing) => add_storages::<E, B>(&node.device, existing, grad),
        None => grad.clone(),
    });
}

impl<S, E, B> Tensor<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    /// Returns a leaf tensor with the same value and requested gradient tracking setting.
    pub fn with_requires_grad(self, requires_grad: bool) -> Self {
        Self::from_storage_with_autograd(
            self.inner.device.clone(),
            self.inner.data.clone(),
            requires_grad,
            true,
            None,
        )
    }

    /// Returns a leaf tensor with gradient tracking enabled.
    pub fn requires_grad(self) -> Self {
        self.with_requires_grad(true)
    }

    pub fn requires_grad_enabled(&self) -> bool {
        self.inner.requires_grad
    }

    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf
    }

    pub fn grad(&self) -> Option<Self> {
        let grad = self
            .inner
            .grad
            .lock()
            .expect("gradient mutex poisoned")
            .clone()?;

        Some(Self::from_storage(self.inner.device.clone(), grad))
    }

    pub fn zero_grad(&self) {
        *self.inner.grad.lock().expect("gradient mutex poisoned") = None;
    }

    pub fn backward_with(&self, grad: Tensor<S, E, B>) {
        run_backward::<E, B>(AnyTensor::from_tensor(self), grad.inner.data.clone());
    }
}

impl<E, B> Scalar<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    pub fn backward(&self) {
        let seed = B::ones(&self.inner.device, D0::NUMEL);
        run_backward::<E, B>(AnyTensor::from_tensor(self), seed);
    }
}

#[cfg(test)]
mod tests {
    use super::{is_grad_enabled, no_grad, with_no_grad};
    use crate::tensor::{Tensor1D, Tensor2D};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    fn assert_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < tolerance,
                "{actual} != {expected}"
            );
        }
    }

    fn finite_difference<F>(values: &[f32], f: F) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let epsilon = 1e-2;
        let mut gradient = Vec::with_capacity(values.len());

        for index in 0..values.len() {
            let mut plus = values.to_vec();
            plus[index] += epsilon;
            let mut minus = values.to_vec();
            minus[index] -= epsilon;
            gradient.push((f(&plus) - f(&minus)) / (2.0 * epsilon));
        }

        gradient
    }

    #[test]
    fn scalar_backward_seeds_leaf_gradient_with_one() {
        let x: crate::tensor::Scalar = crate::tensor::Scalar::from_vec(vec![2.0_f32])
            .unwrap()
            .requires_grad();

        x.backward();

        assert_eq!(x.grad().unwrap().to_vec(), vec![1.0]);
    }

    #[test]
    fn backward_with_seeds_non_scalar_gradient() {
        let x = Tensor1D::<3>::from_array([1.0, 2.0, 3.0]).requires_grad();
        let seed = Tensor1D::<3>::from_array([3.0, 2.0, 1.0]);

        x.backward_with(seed);

        assert_eq!(x.grad().unwrap().to_vec(), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn repeated_backward_accumulates_gradients() {
        let x = Tensor1D::<2>::from_array([2.0, 3.0]).requires_grad();
        let loss = x.mul(&x).sum();

        loss.backward();
        loss.backward();

        assert_eq!(x.grad().unwrap().to_vec(), vec![8.0, 12.0]);
    }

    #[test]
    fn zero_grad_clears_accumulated_gradient() {
        let x = Tensor1D::<2>::from_array([2.0, 3.0]).requires_grad();

        x.sum().backward();
        x.zero_grad();

        assert!(x.grad().is_none());
    }

    #[test]
    fn no_grad_prevents_graph_construction() {
        let x = Tensor1D::<2>::from_array([2.0, 3.0]).requires_grad();

        {
            let _guard = no_grad();
            assert!(!is_grad_enabled());
            let y = x.mul(&x);
            assert!(!y.requires_grad_enabled());
            assert!(y.is_leaf());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn with_no_grad_prevents_graph_construction_inside_closure() {
        let x = Tensor1D::<2>::from_array([2.0, 3.0]).requires_grad();

        let y = with_no_grad(|| {
            assert!(!is_grad_enabled());
            x.mul(&x)
        });

        assert!(!y.requires_grad_enabled());
        assert!(y.is_leaf());
        assert!(is_grad_enabled());
    }

    #[test]
    fn non_requires_grad_tensors_do_not_accumulate_gradients() {
        let x = Tensor1D::<2>::from_array([2.0, 3.0]);

        x.sum().backward();

        assert!(x.grad().is_none());
    }

    #[test]
    fn elementwise_gradients_match_formulas() {
        let x = Tensor1D::<2>::from_array([2.0, 4.0]).requires_grad();
        let y = Tensor1D::<2>::from_array([5.0, 8.0]).requires_grad();

        x.add(&y).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![1.0, 1.0]);
        assert_eq!(y.grad().unwrap().to_vec(), vec![1.0, 1.0]);

        x.zero_grad();
        y.zero_grad();
        x.sub(&y).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![1.0, 1.0]);
        assert_eq!(y.grad().unwrap().to_vec(), vec![-1.0, -1.0]);

        x.zero_grad();
        y.zero_grad();
        x.mul(&y).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![5.0, 8.0]);
        assert_eq!(y.grad().unwrap().to_vec(), vec![2.0, 4.0]);

        x.zero_grad();
        y.zero_grad();
        x.div(&y).sum().backward();
        assert_close(&x.grad().unwrap().to_vec(), &[0.2, 0.125]);
        assert_close(&y.grad().unwrap().to_vec(), &[-0.08, -0.0625]);
    }

    #[test]
    fn reduction_and_unary_gradients_match_formulas() {
        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();

        x.mean().backward();
        assert_close(&x.grad().unwrap().to_vec(), &[1.0 / 3.0; 3]);

        x.zero_grad();
        x.powf(2.0).sum().backward();
        assert_close(&x.grad().unwrap().to_vec(), &[2.0, 4.0, 8.0]);

        x.zero_grad();
        x.ln().sum().backward();
        assert_close(&x.grad().unwrap().to_vec(), &[1.0, 0.5, 0.25]);

        x.zero_grad();
        x.exp().sum().backward();
        assert_close(&x.grad().unwrap().to_vec(), &x.exp().to_vec());
    }

    #[test]
    fn relu_gradient_uses_positive_mask() {
        let x = Tensor1D::<3>::from_array([-1.0, 2.0, 4.0]).requires_grad();

        x.relu().sum().backward();

        assert_eq!(x.grad().unwrap().to_vec(), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn matmul_gradients_match_formulas() {
        let x = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).requires_grad();
        let w =
            Tensor2D::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]).requires_grad();

        x.matmul(&w).sum().backward();

        assert_eq!(
            x.grad().unwrap().to_vec(),
            vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0]
        );
        assert_eq!(
            w.grad().unwrap().to_vec(),
            vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
        );
    }

    #[test]
    fn finite_difference_checks_elementwise_gradients() {
        let y = Tensor1D::<2>::from_array([5.0, 8.0]);

        let x = Tensor1D::<2>::from_array([2.0, 4.0]).requires_grad();
        x.add(&y).sum().backward();
        let expected = finite_difference(&[2.0, 4.0], |values| {
            Tensor1D::<2>::from_vec(values.to_vec())
                .unwrap()
                .add(&y)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<2>::from_array([2.0, 4.0]).requires_grad();
        x.sub(&y).sum().backward();
        let expected = finite_difference(&[2.0, 4.0], |values| {
            Tensor1D::<2>::from_vec(values.to_vec())
                .unwrap()
                .sub(&y)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<2>::from_array([2.0, 4.0]).requires_grad();
        x.mul(&y).sum().backward();
        let expected = finite_difference(&[2.0, 4.0], |values| {
            Tensor1D::<2>::from_vec(values.to_vec())
                .unwrap()
                .mul(&y)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<2>::from_array([2.0, 4.0]).requires_grad();
        x.div(&y).sum().backward();
        let expected = finite_difference(&[2.0, 4.0], |values| {
            Tensor1D::<2>::from_vec(values.to_vec())
                .unwrap()
                .div(&y)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);
    }

    #[test]
    fn finite_difference_checks_matmul_gradients() {
        let w = Tensor2D::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        let x = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).requires_grad();

        x.matmul(&w).sum().backward();

        let expected = finite_difference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |values| {
            Tensor2D::<2, 3>::from_vec(values.to_vec())
                .unwrap()
                .matmul(&w)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);
    }

    #[test]
    fn finite_difference_checks_reduction_gradients() {
        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.sum().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.mean().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .mean()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);
    }

    #[test]
    fn finite_difference_checks_unary_gradients() {
        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.relu().sum().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .relu()
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.exp().sum().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .exp()
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-1);

        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.ln().sum().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .ln()
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);

        let x = Tensor1D::<3>::from_array([1.0, 2.0, 4.0]).requires_grad();
        x.powf(2.0).sum().backward();
        let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
            Tensor1D::<3>::from_vec(values.to_vec())
                .unwrap()
                .powf(2.0)
                .sum()
                .to_vec()[0]
        });
        assert_close_with_tolerance(&x.grad().unwrap().to_vec(), &expected, 1e-2);
    }
}
