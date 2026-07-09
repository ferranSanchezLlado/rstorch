//! Native-dispatch verification via the reference-fall counter.
//!
//! Every op with a `try_*` native hook records a fall when it takes the
//! reference path. On CPU (no hooks) every op falls, which proves the counter
//! is wired at each site. On the Metal hardware lane the claimed forward set
//! must dispatch native, so each op's fall count is asserted to be zero — the
//! Epoch 16.6 acceptance check that an op actually ran on device rather than
//! silently falling back to reference.

use crate::Mask;
use crate::backend::{Backend, fall_counter};
use crate::dtype::FloatDType;
use crate::shape::{C, D1};
use crate::{Tensor1D, Tensor2D, Tensor3D, Tensor4D};

/// Op labels recorded by the forward-set exercise below.
const FORWARD_OPS: &[&str] = &[
    "sum_last",
    "softmax_last",
    "log_softmax_last",
    "cross_entropy",
    "broadcast_last",
    "broadcast_leading",
    "broadcast_channel",
    "index_select_rows",
    "layer_norm_last",
    "rms_norm_last",
    "bmm",
    "relu",
    "neg",
    "exp",
    "ln",
    "tanh",
    "sigmoid",
    "sqrt",
    "abs",
    "gelu",
    "masked_fill",
    "where_mask",
];

fn values<E: FloatDType>(xs: &[f64]) -> Vec<E> {
    xs.iter().map(|&x| E::from_f64(x)).collect()
}

/// Runs one call of every op in the claimed forward set.
fn exercise_forward_set<E, B>()
where
    E: FloatDType,
    B: Backend<E>,
{
    let mat = Tensor2D::<2, 3, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
    mat.sum_last().unwrap();

    let logits =
        Tensor2D::<2, 3, E, B>::from_vec(values(&[1.0, 2.0, 3.0, -1.0, 0.0, 1.0])).unwrap();
    logits.softmax_last().unwrap();
    logits.log_softmax_last().unwrap();
    logits.cross_entropy(&[2, 2]).unwrap();

    let row = Tensor1D::<3, E, B>::from_vec(values(&[10.0, 20.0, 30.0])).unwrap();
    mat.add_last_dim(&row).unwrap();
    let col = Tensor1D::<2, E, B>::from_vec(values(&[100.0, 200.0])).unwrap();
    mat.add_leading_dim(&col).unwrap();
    mat.index_select_rows::<C<3>>(&[1, 0, 1]).unwrap();

    let weight = Tensor1D::<3, E, B>::from_vec(values(&[1.0, 1.0, 1.0])).unwrap();
    let bias = Tensor1D::<3, E, B>::from_vec(values(&[0.0, 0.0, 0.0])).unwrap();
    mat.layer_norm_last(&weight, &bias, E::from_f64(1e-5))
        .unwrap();
    mat.rms_norm_last(&weight, E::from_f64(1e-6)).unwrap();

    let bmm_lhs = Tensor3D::<2, 2, 3, E, B>::from_vec(values(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 3.0, 1.0, 2.0,
    ]))
    .unwrap();
    let bmm_rhs = Tensor3D::<2, 3, 2, E, B>::from_vec(values(&[
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]))
    .unwrap();
    bmm_lhs.bmm(&bmm_rhs).unwrap();

    let unary = Tensor1D::<4, E, B>::from_vec(values(&[0.5, 1.0, 2.0, 4.0])).unwrap();
    unary.relu().unwrap();
    unary.neg().unwrap();
    unary.exp().unwrap();
    unary.ln().unwrap();
    unary.tanh().unwrap();
    unary.sigmoid().unwrap();
    unary.sqrt().unwrap();
    unary.abs().unwrap();
    unary.gelu().unwrap();

    let mask = Mask::<D1<C<4>>, B>::from_vec(vec![true, false, false, true]).unwrap();
    unary.masked_fill(&mask, E::from_f64(9.0)).unwrap();
    let other = Tensor1D::<4, E, B>::from_vec(values(&[10.0, 20.0, 30.0, 40.0])).unwrap();
    unary.where_mask(&mask, &other).unwrap();

    let image =
        Tensor4D::<1, 2, 2, 2, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
            .unwrap();
    let channel = Tensor1D::<2, E, B>::from_vec(values(&[100.0, 200.0])).unwrap();
    image.add_channel_dim(&channel).unwrap();
}

/// Drives a matmul backward so the strided-matmul hook (used only by the
/// matmul/bmm backward transpose products) is exercised.
fn exercise_matmul_backward<E, B>()
where
    E: FloatDType,
    B: Backend<E>,
{
    let lhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap()
        .with_requires_grad(true);
    let rhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[0.5, 0.6, 0.7, 0.8]))
        .unwrap()
        .with_requires_grad(true);
    lhs.matmul(&rhs).unwrap().sum().unwrap().backward().unwrap();
}

/// Trains a tiny `4 -> 8 -> 2` MLP for `steps` full forward/backward/optimizer
/// iterations against a fixed regression target and returns the first and last
/// MSE loss. The optimizer is built by `make_opt` so both SGD and Adam paths
/// can be exercised. This is the on-device training smoke test body: the same
/// code runs on CPU (reference) and Metal (native).
fn train_mlp<E, B, O>(make_opt: impl FnOnce() -> O, steps: usize) -> (f64, f64)
where
    E: FloatDType,
    B: Backend<E>,
    O: crate::optim::Optimizer<E, B>,
{
    use crate::nn::{HasParameters, Linear, Module, mse_loss};
    use crate::random::SmallRng;

    let mut rng = SmallRng::seed_from_u64(7);
    let mut l1 = Linear::<4, 8, E, B>::xavier_uniform(&mut rng).unwrap();
    let mut l2 = Linear::<8, 2, E, B>::xavier_uniform(&mut rng).unwrap();
    let input = Tensor2D::<3, 4, E, B>::from_vec(values(&[
        0.5, -0.2, 0.1, 0.4, -0.3, 0.8, -0.1, 0.2, 0.6, 0.0, -0.4, 0.3,
    ]))
    .unwrap();
    let target =
        Tensor2D::<3, 2, E, B>::from_vec(values(&[0.2, -0.4, -0.1, 0.5, 0.3, 0.1])).unwrap();
    let mut opt = make_opt();

    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut ctx = crate::nn::TrainContext::eval();
        let hidden = l1.forward(&input, &mut ctx).unwrap().relu().unwrap();
        let output = l2.forward(&hidden, &mut ctx).unwrap();
        let loss = mse_loss(&output, &target).unwrap();
        losses.push(loss.item().unwrap().to_f64());
        loss.backward().unwrap();

        let mut params = Vec::new();
        l1.parameters_mut(&mut params);
        l2.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }
    (losses[0], *losses.last().unwrap())
}

#[test]
fn cpu_training_step_uses_reference_optimizer_path() {
    use crate::backend::Cpu;
    use crate::optim::{Adam, Sgd};

    fall_counter::reset();
    let _ = train_mlp::<f32, Cpu, _>(|| Sgd::<f32, Cpu>::with_momentum(0.05, 0.9), 1);
    assert!(
        fall_counter::count("sgd_step") >= 1,
        "expected CPU SGD to take the reference optimizer path"
    );

    fall_counter::reset();
    let _ = train_mlp::<f32, Cpu, _>(|| Adam::<f32, Cpu>::new(0.05), 1);
    assert!(
        fall_counter::count("adam_step") >= 1,
        "expected CPU Adam to take the reference optimizer path"
    );

    let (first, last) = train_mlp::<f32, Cpu, _>(|| Sgd::<f32, Cpu>::with_momentum(0.1, 0.9), 40);
    assert!(last < first, "expected CPU training loss {last} < {first}");
}

#[test]
fn cpu_forward_set_uses_reference_path() {
    use crate::backend::Cpu;
    fall_counter::reset();
    exercise_forward_set::<f32, Cpu>();
    for op in FORWARD_OPS {
        assert!(
            fall_counter::count(op) >= 1,
            "expected CPU to take the reference path for {op}"
        );
    }

    fall_counter::reset();
    exercise_matmul_backward::<f32, Cpu>();
    assert!(
        fall_counter::count("strided_matmul") >= 1,
        "expected CPU matmul backward to take the reference path"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_native {
    use super::*;
    use crate::backend::Metal;
    use crate::dtype::f16;

    fn assert_dispatches_native<E>()
    where
        E: FloatDType,
        Metal: Backend<E>,
    {
        if <Metal as Backend<E>>::default_device().is_err() {
            return;
        }

        fall_counter::reset();
        exercise_forward_set::<E, Metal>();
        for op in FORWARD_OPS {
            assert_eq!(
                fall_counter::count(op),
                0,
                "Metal op {op} fell back to the reference path"
            );
        }

        fall_counter::reset();
        exercise_matmul_backward::<E, Metal>();
        assert_eq!(
            fall_counter::count("strided_matmul"),
            0,
            "Metal strided matmul fell back to the reference path"
        );
    }

    #[test]
    fn metal_f32_dispatches_native() {
        assert_dispatches_native::<f32>();
    }

    #[test]
    fn metal_f16_dispatches_native() {
        assert_dispatches_native::<f16>();
    }

    /// On-device training smoke test: one full step runs the optimizer update
    /// and the matmul backward native (zero reference falls), and a longer run
    /// decreases the loss. The forward set's native dispatch is covered by
    /// `assert_dispatches_native`; here the addition is the backward + fused
    /// optimizer kernels driving a real MLP training loop on hardware.
    fn assert_training_native<E>()
    where
        E: FloatDType,
        Metal: Backend<E>,
    {
        use crate::optim::{Adam, Sgd};

        if <Metal as Backend<E>>::default_device().is_err() {
            return;
        }

        fall_counter::reset();
        let _ = train_mlp::<E, Metal, _>(
            || Sgd::<E, Metal>::with_momentum(E::from_f64(0.05), E::from_f64(0.9)),
            1,
        );
        assert_eq!(
            fall_counter::count("sgd_step"),
            0,
            "Metal SGD step fell back to the reference optimizer path"
        );
        assert_eq!(
            fall_counter::count("strided_matmul"),
            0,
            "Metal matmul backward fell back to the reference path during training"
        );

        fall_counter::reset();
        let _ = train_mlp::<E, Metal, _>(|| Adam::<E, Metal>::new(E::from_f64(0.05)), 1);
        assert_eq!(
            fall_counter::count("adam_step"),
            0,
            "Metal Adam step fell back to the reference optimizer path"
        );

        let (first, last) = train_mlp::<E, Metal, _>(
            || Sgd::<E, Metal>::with_momentum(E::from_f64(0.1), E::from_f64(0.9)),
            40,
        );
        assert!(
            last < first,
            "expected Metal training loss {last} < {first}"
        );
    }

    #[test]
    fn metal_f32_training_step_runs_native() {
        assert_training_native::<f32>();
    }

    #[test]
    fn metal_f16_training_step_runs_native() {
        assert_training_native::<f16>();
    }
}
