//! The autograd engine: graph shape, reverse topological order, cotangent
//! accumulation, the iterative drop, and the detached-capture leak gate.

use super::*;
use crate::device::Device;
use crate::dtype::DType;
use crate::nn::{Mode, Param};
use std::sync::Weak;

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().unwrap()
}

/// Element-wise comparison with an f32-sized slack, for the cases whose
/// expected values are not exactly representable.
fn assert_close(got: &Tensor, expected: &[f32]) {
    let got = v(got);
    assert_eq!(got.len(), expected.len(), "{got:?} vs {expected:?}");
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{got:?} vs {expected:?}");
    }
}

// ------------------------------------------------------------------
// record / make_leaf / traced
// ------------------------------------------------------------------

#[test]
fn record_is_inert_when_no_input_is_traced() {
    let a = t(&[1.0, 2.0], [2]);
    let b = t(&[3.0, 4.0], [2]);
    let out = a.add(&b).unwrap();
    assert!(out.node().is_none());
    assert!(out.backward().is_err());
}

#[test]
fn record_traces_as_soon_as_one_input_carries_a_node() {
    let x = t(&[1.0, 2.0], [2]).traced().unwrap();
    let c = t(&[3.0, 4.0], [2]);
    assert!(x.add(&c).unwrap().node().is_some());
    assert!(c.add(&x).unwrap().node().is_some());
    // …and a value derived from it stays traced.
    assert!(
        x.sigmoid()
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .node()
            .is_some()
    );
}

#[test]
fn record_refuses_to_trace_a_non_float_output() {
    // Casting out of float ends the graph: an i64 value has no cotangent.
    let x = t(&[1.5, -2.5], [2]).traced().unwrap();
    let ints = x.to_dtype(DType::I64).unwrap();
    assert!(ints.node().is_none());
    assert!(matches!(
        ints.backward(),
        Err(Error::NotTraced { op: "backward" })
    ));
}

#[test]
fn make_leaf_shares_storage_and_carries_the_key() {
    let value = t(&[1.0, 2.0, 3.0], [3]);
    let key = GradKey::fresh();
    let leaf = make_leaf(value.clone(), key);
    assert_eq!(leaf.node().unwrap().key, Some(key));
    assert_eq!(v(&leaf), v(&value));
    assert_eq!(leaf.dims(), value.dims());
}

#[test]
fn traced_rejects_double_tracing_and_non_float_dtypes() {
    let x = t(&[1.0], [1]);
    let xt = x.traced().unwrap();
    assert!(matches!(
        xt.traced(),
        Err(Error::InvalidArg { op: "traced", .. })
    ));
    let ints = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
    assert!(matches!(
        ints.traced(),
        Err(Error::InvalidArg { op: "traced", .. })
    ));
    // `detach` undoes tracing, so the detached value may be traced again.
    assert!(xt.detach().traced().is_ok());
}

// ------------------------------------------------------------------
// backward
// ------------------------------------------------------------------

#[test]
fn backward_on_a_graph_less_tensor_is_not_traced() {
    let x = t(&[1.0, 2.0], [2]);
    assert!(matches!(
        x.backward(),
        Err(Error::NotTraced { op: "backward" })
    ));
}

#[test]
fn backward_of_a_small_chain_matches_the_closed_form() {
    // y = sum(3·x²) → dy/dx = 6·x
    let x = t(&[1.0, -2.0, 0.5], [3]);
    let xt = x.traced().unwrap();
    let y = xt.mul(&xt).unwrap().mul_scalar(3.0).unwrap();
    let grads = y.backward().unwrap();
    assert_eq!(grads.len(), 1);
    assert_eq!(v(&grads.wrt_input(&xt).unwrap()), vec![6.0, -12.0, 3.0]);
}

#[test]
fn backward_seeds_with_ones_so_it_differentiates_the_sum() {
    // f = sum(x) over a [2, 2] tensor: every entry gets exactly 1.
    let xt = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]).traced().unwrap();
    let g = xt.mul_scalar(2.0).unwrap().backward().unwrap();
    let g = g.wrt_input(&xt).unwrap();
    assert_eq!(g.dims(), &[2, 2]);
    assert_eq!(v(&g), vec![2.0; 4]);
}

#[test]
fn repeated_use_of_one_leaf_accumulates() {
    // Three uses of the same leaf: x·x + x → 2x + 1.
    let x = t(&[2.0, -1.0], [2]);
    let xt = x.traced().unwrap();
    let y = xt.mul(&xt).unwrap().add(&xt).unwrap();
    let g = y.backward().unwrap();
    assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![5.0, -1.0]);
}

#[test]
fn a_tied_param_accumulates_both_contributions() {
    // One `Param` read twice in a forward pass is one leaf node reached
    // along two paths — the weight-tying case.
    let p = Param::new(t(&[3.0, 4.0], [2]));
    let a = p.get(Mode::TRAIN);
    let b = p.get(Mode::TRAIN);
    // The two reads are the same node…
    assert!(std::ptr::eq(
        Arc::as_ptr(a.node().unwrap()),
        Arc::as_ptr(b.node().unwrap())
    ));
    // …so d(sum(a·b))/dp = 2p.
    let g = a.mul(&b).unwrap().backward().unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(v(&g.wrt(&p).unwrap()), vec![6.0, 8.0]);
}

#[test]
fn a_diamond_sums_both_paths_before_the_closure_runs() {
    // y = (2x) + (3x) → dy/dx = 5, and the shared node must be visited
    // once, after both branches have contributed.
    let xt = t(&[1.0, 1.0], [2]).traced().unwrap();
    let shared = xt.mul_scalar(1.0).unwrap();
    let y = shared
        .mul_scalar(2.0)
        .unwrap()
        .add(&shared.mul_scalar(3.0).unwrap())
        .unwrap();
    let g = y.backward().unwrap();
    assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![5.0, 5.0]);
}

#[test]
fn reduced_graph_fan_in_accumulates_repeated_adds_in_f32() {
    for dtype in [DType::F16, DType::BF16] {
        let x = Tensor::ones((), dtype, &CPU).unwrap().traced().unwrap();
        let mut y = x.mul_scalar(1.0).unwrap();
        for _ in 1..4096 {
            y = y.add(&x.mul_scalar(1.0).unwrap()).unwrap();
        }
        let grad = y.backward().unwrap().wrt_input(&x).unwrap();
        assert_eq!(grad.dtype(), dtype);
        assert_eq!(grad.item().unwrap(), 4096.0);
    }
}

#[test]
fn several_leaves_land_under_their_own_keys() {
    let a = t(&[1.0, 2.0], [2]).traced().unwrap();
    let b = t(&[3.0, 4.0], [2]).traced().unwrap();
    let g = a.mul(&b).unwrap().backward().unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(v(&g.wrt_input(&a).unwrap()), vec![3.0, 4.0]);
    assert_eq!(v(&g.wrt_input(&b).unwrap()), vec![1.0, 2.0]);
}

#[test]
fn backward_is_pure_and_repeatable() {
    let xt = t(&[1.0, 2.0], [2]).traced().unwrap();
    let y = xt.mul(&xt).unwrap();
    let first = y.backward().unwrap();
    let second = y.backward().unwrap();
    assert_eq!(
        v(&first.wrt_input(&xt).unwrap()),
        v(&second.wrt_input(&xt).unwrap())
    );
}

#[test]
fn broadcast_gradients_reduce_back_to_the_operand_shape() {
    let row = t(&[1.0, 2.0, 3.0], [1, 3]).traced().unwrap();
    let m = t(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3]);
    let g = row.mul(&m).unwrap().backward().unwrap();
    let g = g.wrt_input(&row).unwrap();
    assert_eq!(g.dims(), &[1, 3]);
    assert_eq!(v(&g), vec![2.0, 2.0, 2.0]);
}

#[test]
fn reduced_broadcast_gradients_keep_multi_axis_sum_to_wide() {
    for dtype in [DType::F16, DType::BF16] {
        let (width, values) = if dtype == DType::F16 {
            let width = 65_520;
            let mut values = vec![1.0f32; width];
            values.extend(vec![-1.0; width]);
            (width, values)
        } else {
            let width = 257;
            let mut values = vec![1.0f32; width];
            values.extend((0..width).map(|index| if index < 256 { -1.0 } else { 0.0 }));
            (width, values)
        };
        let leaf = Tensor::ones([1, 1], dtype, &CPU).unwrap().traced().unwrap();
        let weights = match dtype {
            DType::F16 => Tensor::from_vec(
                values.into_iter().map(half::f16::from_f32).collect(),
                [2, width],
                &CPU,
            ),
            DType::BF16 => Tensor::from_vec(
                values.into_iter().map(half::bf16::from_f32).collect(),
                [2, width],
                &CPU,
            ),
            _ => unreachable!(),
        }
        .unwrap();
        let grad = leaf
            .mul(&weights)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt_input(&leaf)
            .unwrap();
        let expected = if dtype == DType::F16 { 0.0 } else { 1.0 };
        assert_eq!(grad.item().unwrap(), expected);
    }
}

// ------------------------------------------------------------------
// Mode / Param interaction
// ------------------------------------------------------------------

#[test]
fn eval_mode_and_frozen_params_record_nothing() {
    let mut p = Param::new(t(&[1.0, 2.0], [2]));
    assert!(p.get(Mode::EVAL).node().is_none());
    assert!(p.get(Mode::TRAIN.frozen()).node().is_none());
    assert!(p.get(Mode::EVAL.recorded()).node().is_some());

    p.freeze();
    assert!(p.get(Mode::TRAIN).node().is_none());
    let out = p.get(Mode::TRAIN).mul_scalar(2.0).unwrap();
    assert!(out.node().is_none());
    assert!(out.backward().is_err());

    p.unfreeze();
    assert!(p.get(Mode::TRAIN).node().is_some());
}

#[test]
fn param_set_rebuilds_the_leaf_and_leaves_a_live_graph_intact() {
    let mut p = Param::new(t(&[2.0], [1]));
    let y = p.get(Mode::TRAIN).mul(&p.get(Mode::TRAIN)).unwrap();
    p.set(t(&[10.0], [1])).unwrap();
    // The live graph kept the value it was built from — differentiating
    // it still yields 2·2, not 2·10 — and the identity is unchanged, so
    // the gradient is still found under the parameter's key.
    let g = y.backward().unwrap();
    assert_eq!(v(&g.wrt(&p).unwrap()), vec![4.0]);

    // The fresh leaf is a *different* node carrying the same key.
    let g2 = p
        .get(Mode::TRAIN)
        .mul(&p.get(Mode::TRAIN))
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(v(&g2.wrt(&p).unwrap()), vec![20.0]);
}

// ------------------------------------------------------------------
// Grads — the linear surface
// ------------------------------------------------------------------

fn grads_of(x: &Tensor, scale: f64) -> (Tensor, Grads) {
    let xt = x.traced().unwrap();
    let g = xt.mul_scalar(scale).unwrap().backward().unwrap();
    (xt, g)
}

#[test]
fn merge_unions_keys_and_sums_the_overlap() {
    let x = t(&[1.0, 1.0], [2]);
    let xt = x.traced().unwrap();
    let a = xt.mul_scalar(2.0).unwrap().backward().unwrap();
    let b = xt.mul_scalar(5.0).unwrap().backward().unwrap();
    let (yt, c) = grads_of(&t(&[1.0], [1]), 3.0);

    let merged = a.merge(b).unwrap().merge(c).unwrap();
    assert_eq!(merged.len(), 2);
    assert_eq!(v(&merged.wrt_input(&xt).unwrap()), vec![7.0, 7.0]);
    assert_eq!(v(&merged.wrt_input(&yt).unwrap()), vec![3.0]);
}

#[test]
fn repeated_reduced_grads_merges_retain_the_wide_accumulator() {
    for dtype in [DType::F16, DType::BF16] {
        let x = Tensor::ones((), dtype, &CPU).unwrap().traced().unwrap();
        let one_grad = || x.mul_scalar(1.0).unwrap().backward().unwrap();
        let mut merged = one_grad();
        for _ in 1..4096 {
            merged = merged.merge(one_grad()).unwrap();
        }
        let grad = merged.wrt_input(&x).unwrap();
        assert_eq!(grad.dtype(), dtype);
        assert_eq!(grad.item().unwrap(), 4096.0);
    }
}

#[test]
fn scale_multiplies_every_entry() {
    let (xt, g) = grads_of(&t(&[1.0, 1.0], [2]), 4.0);
    let g = g.scale(0.25).unwrap();
    assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![1.0, 1.0]);

    let (_, g) = grads_of(&t(&[1.0], [1]), 1.0);
    assert!(matches!(
        g.scale(f64::NAN),
        Err(Error::InvalidArg { op: "scale", .. })
    ));
}

#[test]
fn clip_norm_only_bites_above_the_budget() {
    // Two leaves with gradients [3, 0] and [4]: global norm 5.
    let a = t(&[1.0, 1.0], [2]).traced().unwrap();
    let b = t(&[1.0], [1]).traced().unwrap();
    let build = || {
        let lhs = a.mul(&t(&[3.0, 0.0], [2])).unwrap();
        let rhs = b.mul_scalar(4.0).unwrap();
        lhs.sum_all().unwrap().add(&rhs.sum_all().unwrap()).unwrap()
    };

    // Under budget: untouched.
    let g = build().backward().unwrap().clip_norm(10.0).unwrap();
    assert_eq!(v(&g.wrt_input(&a).unwrap()), vec![3.0, 0.0]);
    assert_eq!(v(&g.wrt_input(&b).unwrap()), vec![4.0]);

    // Over budget: scaled by 1/5 to land on it.
    let g = build().backward().unwrap().clip_norm(1.0).unwrap();
    assert_close(&g.wrt_input(&a).unwrap(), &[0.6, 0.0]);
    assert_close(&g.wrt_input(&b).unwrap(), &[0.8]);

    let g = build().backward().unwrap();
    assert!(matches!(
        g.clip_norm(0.0),
        Err(Error::InvalidArg {
            op: "clip_norm",
            ..
        })
    ));
}

#[test]
fn norm_is_the_value_clip_norm_measures_against_its_budget() {
    // Two leaves with gradients [3, 0] and [4]: global norm exactly 5 in f64.
    let a = t(&[1.0, 1.0], [2]).traced().unwrap();
    let b = t(&[1.0], [1]).traced().unwrap();
    let build = || {
        let lhs = a.mul(&t(&[3.0, 0.0], [2])).unwrap();
        let rhs = b.mul_scalar(4.0).unwrap();
        lhs.sum_all().unwrap().add(&rhs.sum_all().unwrap()).unwrap()
    };

    let g = build().backward().unwrap();
    assert_eq!(g.norm().unwrap(), 5.0);

    // A clip that does not trigger leaves the norm alone…
    let g = g.clip_norm(10.0).unwrap();
    assert_eq!(g.norm().unwrap(), 5.0);

    // …and one that does reports the clipped gradients: the budget itself.
    let g = g.clip_norm(1.0).unwrap();
    assert!((g.norm().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn norm_of_an_empty_grads_is_zero() {
    assert_eq!(Grads::from_pairs(HashMap::new()).norm().unwrap(), 0.0);
}

#[test]
fn clip_norm_of_an_empty_grads_is_a_no_op() {
    let g = Grads::from_pairs(HashMap::new());
    assert!(g.is_empty());
    let g = g.clip_norm(1.0).unwrap();
    assert_eq!(g.len(), 0);
}

#[test]
fn reduced_clip_norm_widens_before_square_and_sum() {
    for dtype in [DType::F16, DType::BF16] {
        let x = Tensor::ones([4096], dtype, &CPU).unwrap().traced().unwrap();
        let grads = x.sum_all().unwrap().backward().unwrap();
        let clipped = grads.clip_norm(32.0).unwrap().wrt_input(&x).unwrap();
        assert_eq!(clipped.dtype(), dtype);
        assert!(
            clipped
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 0.5)
        );
    }
}

#[test]
fn lookups_are_loud_about_the_wrong_binding() {
    let x = t(&[1.0, 2.0], [2]);
    let (xt, g) = grads_of(&x, 2.0);

    // The original, untraced tensor.
    assert!(matches!(
        g.wrt_input(&x),
        Err(Error::NotTraced { op: "wrt_input" })
    ));
    // An interior value rather than the leaf.
    let interior = xt.mul_scalar(1.0).unwrap();
    assert!(matches!(
        g.wrt_input(&interior),
        Err(Error::InvalidArg {
            op: "wrt_input",
            ..
        })
    ));
    // A leaf that took no part in the computation.
    let other = t(&[1.0], [1]).traced().unwrap();
    assert!(matches!(
        g.wrt_input(&other),
        Err(Error::InvalidArg {
            op: "wrt_input",
            ..
        })
    ));
    // An untrained parameter.
    let p = Param::new(t(&[1.0], [1]));
    assert!(matches!(g.wrt(&p), Err(Error::NotTraced { op: "wrt" })));
}

#[test]
fn take_and_contains_drain_by_key() {
    let p = Param::new(t(&[2.0], [1]));
    let mut g = p
        .get(Mode::TRAIN)
        .mul_scalar(3.0)
        .unwrap()
        .backward()
        .unwrap();
    assert!(g.contains(p.grad_key()));
    assert_eq!(v(&g.take(p.grad_key()).unwrap().unwrap()), vec![3.0]);
    assert!(!g.contains(p.grad_key()));
    assert!(g.take(p.grad_key()).unwrap().is_none());
    assert!(g.is_empty());
}

// ------------------------------------------------------------------
// Leak / capture discipline
// ------------------------------------------------------------------

#[test]
fn a_dropped_sigmoid_chain_frees_every_node() {
    // The detached-output capture rule: `sigmoid`'s backward needs its own
    // output, and capturing the *traced* one would make node → closure →
    // node an `Arc` cycle that never frees. Strong counts are the witness.
    let leaf = t(&[0.5, -0.25, 0.75], [3]).traced().unwrap();
    let leaf_node = Arc::clone(leaf.node().unwrap());
    assert_eq!(Arc::strong_count(&leaf_node), 2);

    let mut y = leaf.clone();
    for _ in 0..16 {
        y = y.sigmoid().unwrap().tanh().unwrap().mul(&y).unwrap();
    }
    assert!(Arc::strong_count(&leaf_node) > 2);

    // Backward must not extend the graph either.
    let g = y.backward().unwrap();
    assert_eq!(g.len(), 1);
    let count_after_backward = Arc::strong_count(&leaf_node);
    drop(g);

    drop(y);
    drop(leaf);
    assert_eq!(
        Arc::strong_count(&leaf_node),
        1,
        "graph leaked (count after backward was {count_after_backward})"
    );
}

#[test]
fn gradients_carry_no_graph_of_their_own() {
    let xt = t(&[0.5, 1.5], [2]).traced().unwrap();
    let g = xt.sigmoid().unwrap().backward().unwrap();
    assert!(g.wrt_input(&xt).unwrap().node().is_none());
}

// ------------------------------------------------------------------
// 100k-node stress: neither the walk nor the drop may use the stack
// ------------------------------------------------------------------

/// A synthetic identity chain `depth` nodes deep over a rank-0 tensor,
/// built without touching the op layer so the stress test measures the
/// engine rather than 100k kernel launches. Returns the head tensor and a
/// `Weak` to the leaf node (alive iff the graph has not been freed).
fn identity_chain(depth: usize) -> (Tensor, Weak<Node>) {
    let value = Tensor::full((), 1.0, DType::F32, &CPU).unwrap();
    let leaf = Arc::new(Node {
        op: "leaf",
        key: Some(GradKey::fresh()),
        inputs: Vec::new(),
        backward: None,
    });
    let weak = Arc::downgrade(&leaf);
    let mut head = leaf;
    for _ in 0..depth {
        head = Arc::new(Node {
            op: "identity",
            key: None,
            inputs: vec![Some(head)],
            backward: Some(Box::new(|g: &Tensor| Ok(vec![Some(g.clone())]))),
        });
    }
    let tensor = Tensor::from_parts_traced(value.storage().clone(), value.layout().clone(), head);
    (tensor, weak)
}

/// Run `body` on a thread with a deliberately small stack: a recursive
/// drop or walk over 100k nodes overflows it, an iterative one does not.
fn on_a_small_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("the engine must not recurse per node");
}

#[test]
fn dropping_a_100k_node_graph_is_iterative() {
    on_a_small_stack(|| {
        let (head, leaf) = identity_chain(100_000);
        assert!(leaf.upgrade().is_some());
        drop(head);
        assert!(leaf.upgrade().is_none(), "the chain must be freed");
    });
}

#[test]
fn backward_over_a_100k_node_graph_is_iterative() {
    on_a_small_stack(|| {
        let (head, _) = identity_chain(100_000);
        let grads = backward(&head).unwrap();
        assert_eq!(grads.len(), 1);
    });
}

#[test]
fn a_long_chain_of_real_ops_walks_and_drops_iteratively() {
    // The same stress through the op layer, at a size where 20k kernel
    // launches stay cheap: `x` scaled by 1.0 twenty thousand times still
    // has gradient 1.
    on_a_small_stack(|| {
        let xt = Tensor::full((), 2.0, DType::F32, &CPU)
            .unwrap()
            .traced()
            .unwrap();
        let mut y = xt.clone();
        for _ in 0..20_000 {
            y = y.mul_scalar(1.0).unwrap();
        }
        let g = y.backward().unwrap();
        assert_eq!(g.wrt_input(&xt).unwrap().item().unwrap(), 1.0);
        drop(g);
        drop(y);
    });
}

#[test]
fn the_graph_is_shareable_across_threads() {
    // `backward` is a pure function over an immutable `Arc` graph, so two
    // threads may differentiate the same tensor concurrently.
    let xt = t(&[1.0, 2.0, 3.0], [3]).traced().unwrap();
    let y = xt.mul(&xt).unwrap();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let y = y.clone();
                s.spawn(move || y.backward().unwrap().len())
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 1);
        }
    });
}
