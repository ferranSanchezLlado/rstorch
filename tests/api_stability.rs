use rstorch::prelude::*;
use rstorch::{Checkpoint, OptimizerKind, Shape, StateDict, TensorRecord};
use std::fmt::Debug;

fn assert_send_sync<T: Send + Sync>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: Debug>() {}

#[test]
fn public_auto_trait_baseline_is_locked() {
    assert_send_sync::<Tensor1D<2>>();
    assert_clone::<Tensor1D<2>>();
    assert_debug::<Tensor1D<2>>();

    assert_send_sync::<Mask<D1<C<2>>>>();
    assert_clone::<Mask<D1<C<2>>>>();
    assert_debug::<Mask<D1<C<2>>>>();

    assert_send_sync::<Shape>();
    assert_clone::<Shape>();
    assert_debug::<Shape>();

    assert_send_sync::<SmallRng>();
    assert_clone::<SmallRng>();
    assert_debug::<SmallRng>();

    assert_send_sync::<TensorRecord>();
    assert_clone::<TensorRecord>();
    assert_debug::<TensorRecord>();

    assert_send_sync::<StateDict>();
    assert_clone::<StateDict>();
    assert_debug::<StateDict>();

    assert_send_sync::<Checkpoint>();
    assert_clone::<Checkpoint>();
    assert_debug::<Checkpoint>();

    assert_send_sync::<OptimizerKind>();
    assert_clone::<OptimizerKind>();
    assert_debug::<OptimizerKind>();
}
