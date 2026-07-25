//! **The m1 acceptance fixture** (exploration §10, implementation-plan §3):
//! the tensor op surface exercised end to end from *outside* the crate.
//!
//! This file is deliberately a downstream consumer. It imports nothing but
//! `rstorch::prelude::*`, so anything it needs must be public, documented,
//! and reachable from the eleven-item first-hour vocabulary. If a step here
//! requires a crate-internal type, an extra import, or a generic parameter
//! in a value position, that is the API failing its own acceptance test —
//! which is the point of keeping the fixtures in a separate crate.
//!
//! Coverage: construction, host round-trips, element-wise math and
//! broadcasting, operator sugar, views and reshape, matmul (2-D and
//! batched), reductions and softmax, indexing and masks, and the loud
//! two-tier error behavior. Values are asserted, not just shapes.

use rstorch::prelude::*;

/// The device under test. `best_available()` is the spelling a user writes;
/// today it resolves to CPU.
fn dev() -> Device {
    Device::best_available()
}

/// Assert two float slices agree to 1e-6, reporting the first difference.
#[track_caller]
fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(g) - f64::from(w)).abs() <= 1e-6 * f64::from(w).abs().max(1.0),
            "element {i}: got {g}, want {w} (in {got:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Construction and host round-trips
// ---------------------------------------------------------------------------

#[test]
fn construct_and_inspect() -> Result<()> {
    let dev = dev();

    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &dev)?;
    assert_eq!(x.dims(), &[2, 3]);
    assert_eq!(x.shape(), &Shape::from([2, 3]));
    assert_eq!(x.rank(), 2);
    assert_eq!(x.num_elements(), 6);
    assert_eq!(x.dtype(), DType::F32);
    assert_eq!(x.device(), dev);
    assert!(x.is_contiguous());
    // Rank assumptions are explicit and loud, never inferred.
    assert_eq!(x.dims2()?, (2, 3));

    assert_eq!(
        Tensor::zeros([2, 2], DType::F32, &dev)?.to_vec::<f32>()?,
        vec![0.0; 4]
    );
    assert_eq!(
        Tensor::ones([3], DType::I64, &dev)?.to_vec::<i64>()?,
        vec![1; 3]
    );
    assert_eq!(
        Tensor::full([2], -1.5, DType::F32, &dev)?.to_vec::<f32>()?,
        vec![-1.5, -1.5]
    );
    assert_eq!(
        Tensor::arange(0.0, 5.0, 2.0, DType::F32, &dev)?.to_vec::<f32>()?,
        vec![0.0, 2.0, 4.0]
    );

    // A single element reads back as its own type or dtype-agnostically.
    let one = Tensor::from_vec(vec![7.5f32], [1], &dev)?;
    assert_eq!(one.to_scalar::<f32>()?, 7.5);
    assert_eq!(one.item()?, 7.5);

    // Dtype is runtime data, and casts are explicit.
    let ints = x.to_dtype(DType::I64)?;
    assert_eq!(ints.dtype(), DType::I64);
    assert_eq!(ints.to_vec::<i64>()?, vec![1, 2, 3, 4, 5, 6]);

    // Debug/Display are first-hour features: a summary a user can read.
    let shown = format!("{x}");
    assert!(shown.contains("f32"), "{shown}");
    assert!(format!("{x:?}").contains("Tensor"), "{x:?}");
    Ok(())
}

#[test]
fn seeded_random_construction_is_reproducible() -> Result<()> {
    let dev = dev();
    let mut a = Rng::seed(42);
    let mut b = Rng::seed(42);
    let x = Tensor::randn([4, 4], DType::F32, &dev, &mut a)?;
    let y = Tensor::randn([4, 4], DType::F32, &dev, &mut b)?;
    assert_eq!(x.to_vec::<f32>()?, y.to_vec::<f32>()?);

    let u = Tensor::rand([64], DType::F32, &dev, &mut a)?.to_vec::<f32>()?;
    assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)), "uniform range");
    Ok(())
}

// ---------------------------------------------------------------------------
// Element-wise math, broadcasting, operator sugar
// ---------------------------------------------------------------------------

#[test]
fn elementwise_and_broadcasting() -> Result<()> {
    let dev = dev();
    let a = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], [2, 2], &dev)?;
    let b = Tensor::from_vec(vec![0.5f32, 2.0, -1.0, 4.0], [2, 2], &dev)?;

    assert_close(&a.add(&b)?.to_vec::<f32>()?, &[1.5, 0.0, 2.0, 0.0]);
    assert_close(&a.sub(&b)?.to_vec::<f32>()?, &[0.5, -4.0, 4.0, -8.0]);
    assert_close(&a.mul(&b)?.to_vec::<f32>()?, &[0.5, -4.0, -3.0, -16.0]);
    assert_close(&a.div(&b)?.to_vec::<f32>()?, &[2.0, -1.0, -3.0, -1.0]);
    assert_close(&a.maximum(&b)?.to_vec::<f32>()?, &[1.0, 2.0, 3.0, 4.0]);
    assert_close(&a.minimum(&b)?.to_vec::<f32>()?, &[0.5, -2.0, -1.0, -4.0]);

    // Operator sugar computes exactly the same thing.
    assert_eq!((&a + &b).to_vec::<f32>()?, a.add(&b)?.to_vec::<f32>()?);
    assert_close(&(&a * 2.0).to_vec::<f32>()?, &[2.0, -4.0, 6.0, -8.0]);
    assert_close(&(&a - 1.0).to_vec::<f32>()?, &[0.0, -3.0, 2.0, -5.0]);

    // NumPy/PyTorch broadcasting: a row vector against a matrix.
    let row = Tensor::from_vec(vec![10.0f32, 20.0], [1, 2], &dev)?;
    assert_close(&(&a + &row).to_vec::<f32>()?, &[11.0, 18.0, 13.0, 16.0]);

    // Unaries, including the exact-GELU and output-dependent ones.
    assert_close(&a.relu()?.to_vec::<f32>()?, &[1.0, 0.0, 3.0, 0.0]);
    assert_close(&a.neg()?.to_vec::<f32>()?, &[-1.0, 2.0, -3.0, 4.0]);
    assert_close(&a.abs()?.to_vec::<f32>()?, &[1.0, 2.0, 3.0, 4.0]);
    let sig = a.sigmoid()?.to_vec::<f32>()?;
    assert_close(&sig[..1], &[0.731_058_6]);
    let pos = Tensor::from_vec(vec![1.0f32, 4.0], [2], &dev)?;
    assert_close(&pos.sqrt()?.to_vec::<f32>()?, &[1.0, 2.0]);
    assert_close(&pos.ln()?.exp()?.to_vec::<f32>()?, &[1.0, 4.0]);
    // Exact GELU (erf), not the tanh approximation: gelu(1) = 0.841345.
    assert_close(&pos.gelu()?.to_vec::<f32>()?[..1], &[0.841_345]);

    // Comparisons produce Bool tensors that live on the device.
    let mask = a.gt(&b)?;
    assert_eq!(mask.dtype(), DType::Bool);
    assert_eq!(mask.to_vec::<bool>()?, vec![true, false, true, false]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Shapes: views, reshape, cat/stack
// ---------------------------------------------------------------------------

#[test]
fn reshape_transpose_and_views() -> Result<()> {
    let dev = dev();
    let x = Tensor::from_vec((0..12).map(|i| i as f32).collect(), [3, 4], &dev)?;

    // `reshape` is a view when the layout permits.
    let r = x.reshape([2, 6])?;
    assert_eq!(r.dims(), &[2, 6]);
    assert_eq!(r.to_vec::<f32>()?[..3], [0.0, 1.0, 2.0]);
    // Inferring is not magic: shapes are data, so state them.
    assert_eq!(x.reshape([12])?.dims(), &[12]);

    // Zero-copy views; `contiguous()` is public and explicit.
    let t = x.transpose(0, 1)?;
    assert_eq!(t.dims(), &[4, 3]);
    assert!(!t.is_contiguous());
    assert_close(&t.to_vec::<f32>()?[..3], &[0.0, 4.0, 8.0]);
    assert!(t.contiguous()?.is_contiguous());
    // A reshape that the strides cannot express copies instead of failing.
    assert_eq!(t.reshape([12])?.to_vec::<f32>()?[1], 4.0);

    assert_eq!(x.permute(&[1, 0])?.dims(), &[4, 3]);
    assert_eq!(
        x.narrow(1, 1, 2)?.to_vec::<f32>()?,
        vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
    );
    assert_eq!(x.unsqueeze(0)?.dims(), &[1, 3, 4]);
    assert_eq!(x.unsqueeze(0)?.squeeze(0)?.dims(), &[3, 4]);
    // Negative axes index from the end.
    assert_eq!(x.unsqueeze(-1)?.dims(), &[3, 4, 1]);

    let row = Tensor::from_vec(vec![1.0f32, 2.0], [1, 2], &dev)?;
    assert_eq!(
        row.broadcast_to([3, 2])?.to_vec::<f32>()?,
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );

    let a = Tensor::from_vec(vec![1.0f32, 2.0], [2], &dev)?;
    let b = Tensor::from_vec(vec![3.0f32, 4.0], [2], &dev)?;
    assert_eq!(
        Tensor::cat(&[&a, &b], 0)?.to_vec::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let s = Tensor::stack(&[&a, &b], 1)?;
    assert_eq!(s.dims(), &[2, 2]);
    assert_eq!(s.to_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

#[test]
fn matmul_2d_and_batched() -> Result<()> {
    let dev = dev();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &dev)?;
    let b = Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], &dev)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_close(&c.to_vec::<f32>()?, &[58.0, 64.0, 139.0, 154.0]);

    // `x @ w.T` — the shape every dense layer produces — goes straight
    // through a transposed view, no manual materialization.
    let w = Tensor::from_vec(vec![7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0], [2, 3], &dev)?;
    assert_close(
        &a.matmul(&w.transpose(-2, -1)?)?.to_vec::<f32>()?,
        &[58.0, 64.0, 139.0, 154.0],
    );

    // Batched, with the batch axis broadcast against a shared right operand.
    let batched = a.unsqueeze(0)?.broadcast_to([2, 2, 3])?;
    let out = batched.matmul(&b.unsqueeze(0)?)?;
    assert_eq!(out.dims(), &[2, 2, 2]);
    assert_close(
        &out.to_vec::<f32>()?,
        &[58.0, 64.0, 139.0, 154.0, 58.0, 64.0, 139.0, 154.0],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[test]
fn reductions_and_softmax() -> Result<()> {
    let dev = dev();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &dev)?;

    // One spelling per operation: `sum(axis)`, `sum_keepdim`, `sum_all`.
    assert_close(&x.sum(0)?.to_vec::<f32>()?, &[5.0, 7.0, 9.0]);
    assert_close(&x.sum(1)?.to_vec::<f32>()?, &[6.0, 15.0]);
    assert_eq!(x.sum_keepdim(1)?.dims(), &[2, 1]);
    assert_eq!(x.sum_all()?.item()?, 21.0);
    assert_eq!(x.mean_all()?.item()?, 3.5);
    assert_close(&x.mean(1)?.to_vec::<f32>()?, &[2.0, 5.0]);
    assert_close(&x.max(0)?.to_vec::<f32>()?, &[4.0, 5.0, 6.0]);
    assert_close(&x.min(1)?.to_vec::<f32>()?, &[1.0, 4.0]);
    // Variance uses correction = 1 (PyTorch's default), so [1,2,3] -> 1.0.
    assert_close(&x.var(1)?.to_vec::<f32>()?, &[1.0, 1.0]);
    assert_close(&x.std(1)?.to_vec::<f32>()?, &[1.0, 1.0]);
    // Negative axes work here too.
    assert_close(&x.sum(-1)?.to_vec::<f32>()?, &[6.0, 15.0]);

    // Softmax is stable and normalizes along the axis.
    let p = x.softmax(-1)?;
    assert_close(&p.sum(1)?.to_vec::<f32>()?, &[1.0, 1.0]);
    assert_close(
        &p.to_vec::<f32>()?[..3],
        &[0.090_030_57, 0.244_728_47, 0.665_240_96],
    );
    let lp = x.log_softmax(-1)?;
    assert_close(&lp.exp()?.to_vec::<f32>()?, &p.to_vec::<f32>()?);

    // argmax/argmin are I64 tensors — ordinary tensors, usable as indices.
    let am = x.argmax(1)?;
    assert_eq!(am.dtype(), DType::I64);
    assert_eq!(am.to_vec::<i64>()?, vec![2, 2]);
    assert_eq!(x.argmin(0)?.to_vec::<i64>()?, vec![0, 0, 0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Indexing and masks
// ---------------------------------------------------------------------------

#[test]
fn indexing_and_masks() -> Result<()> {
    let dev = dev();
    let table = Tensor::from_vec((0..12).map(|i| i as f32).collect(), [4, 3], &dev)?;

    // The embedding-lookup path: rows picked by an I64 index tensor.
    let ids = Tensor::index_vec(&[3, 0, 3], &dev)?;
    let rows = table.index_select(0, &ids)?;
    assert_eq!(rows.dims(), &[3, 3]);
    assert_close(
        &rows.to_vec::<f32>()?,
        &[9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0],
    );

    // `arange`-built ranges and per-element gather.
    assert_eq!(
        Tensor::index_range(4, &dev)?.to_vec::<i64>()?,
        vec![0, 1, 2, 3]
    );
    let picks = Tensor::from_vec(vec![2i64, 0, 1, 1], [4, 1], &dev)?;
    assert_close(
        &table.gather(1, &picks)?.to_vec::<f32>()?,
        &[2.0, 3.0, 7.0, 10.0],
    );

    // Comparison-built masks stay on the device; `masked_fill` and
    // `where_cond` consume them.
    let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], [2, 2], &dev)?;
    let neg = x.lt(&Tensor::zeros([2, 2], DType::F32, &dev)?)?;
    assert_close(
        &x.masked_fill(&neg, 0.0)?.to_vec::<f32>()?,
        &[1.0, 0.0, 3.0, 0.0],
    );
    let alt = Tensor::full([2, 2], 9.0, DType::F32, &dev)?;
    assert_close(
        &neg.where_cond(&alt, &x)?.to_vec::<f32>()?,
        &[1.0, 9.0, 3.0, 9.0],
    );

    // The attention prerequisite: a causal mask built from the same parts.
    let causal = Tensor::causal_mask(3, &dev)?;
    assert_eq!(causal.dtype(), DType::Bool);
    assert_eq!(causal.dims(), &[3, 3]);
    Ok(())
}

// ---------------------------------------------------------------------------
// The loud edges
// ---------------------------------------------------------------------------

#[test]
fn errors_are_structured_and_op_named() {
    let dev = dev();
    let a = Tensor::from_vec(vec![1.0f32; 6], [2, 3], &dev).expect("build a");
    let b = Tensor::from_vec(vec![1.0f32; 20], [4, 5], &dev).expect("build b");

    // Shape mismatch names the op and both shapes.
    let err = a.add(&b).expect_err("2x3 + 4x5 must fail");
    assert!(
        matches!(err, Error::ShapeMismatch { op: "add", .. }),
        "{err}"
    );
    assert!(err.to_string().contains("[2, 3]"), "{err}");

    // No implicit dtype promotion: the error says how to fix it.
    let ints = Tensor::ones([2, 3], DType::I64, &dev).expect("build ints");
    let err = a.add(&ints).expect_err("f32 + i64 must fail");
    assert!(
        matches!(err, Error::DTypeMismatch { op: "add", .. }),
        "{err}"
    );
    assert!(err.to_string().contains("to_dtype"), "{err}");

    // Axes are validated against the rank.
    assert!(matches!(
        a.sum(5),
        Err(Error::InvalidAxis { op: "sum", .. })
    ));
}

/// Tier two of the fallibility policy: the operator spelling panics with the
/// *identical* structured message the `Result` tier returns.
#[test]
#[should_panic(expected = "add: shape mismatch: lhs [2, 3] vs rhs [4, 5]")]
fn operator_sugar_panics_with_the_same_message() {
    let dev = dev();
    let a = Tensor::from_vec(vec![1.0f32; 6], [2, 3], &dev).expect("build a");
    let b = Tensor::from_vec(vec![1.0f32; 20], [4, 5], &dev).expect("build b");
    let _ = &a + &b;
}

// ---------------------------------------------------------------------------
// The composite: everything above in one small pipeline
// ---------------------------------------------------------------------------

/// A dense layer + activation + cross-entropy-shaped read-out, written the
/// way the flagship loop (exploration §4.7) writes it, minus the `nn` types
/// that arrive in wave 4. This is the m1 acceptance example: if this reads
/// naturally, the op surface has done its job.
#[test]
fn first_hour_pipeline() -> Result<()> {
    let dev = dev();
    let mut rng = Rng::seed(7);

    let x = Tensor::randn([8, 4], DType::F32, &dev, &mut rng)?;
    let w = Tensor::randn([3, 4], DType::F32, &dev, &mut rng)?;
    let bias = Tensor::zeros([3], DType::F32, &dev)?;
    let labels = Tensor::from_vec(vec![0i64, 1, 2, 0, 1, 2, 0, 1], [8], &dev)?;

    // logits = relu(x @ w.T + b)
    let logits = (&x.matmul(&w.transpose(-2, -1)?)? + &bias).relu()?;
    let (batch, classes) = logits.dims2()?;
    assert_eq!((batch, classes), (8, 3));

    // Negative log-likelihood via log_softmax + gather — one path, I64
    // labels, no bespoke loss plumbing.
    let logp = logits.log_softmax(-1)?;
    let picked = logp.gather(1, &labels.reshape([8, 1])?)?;
    let loss = picked.mean_all()?.neg()?;
    assert!(
        loss.item()? > 0.0,
        "nll must be positive, got {}",
        loss.item()?
    );

    // Accuracy: argmax, compare against the labels, mean of the Bool cast.
    let predicted = logits.argmax(-1)?;
    let correct = predicted.eq(&labels)?.to_dtype(DType::F32)?;
    let accuracy = correct.mean_all()?.item()?;
    assert!(
        (0.0..=1.0).contains(&accuracy),
        "accuracy {accuracy} out of range"
    );

    // Every intermediate is an ordinary immutable value: shapes line up and
    // the pipeline round-trips to host without ceremony.
    assert_eq!(logp.dims(), &[8, 3]);
    assert_eq!(picked.dims(), &[8, 1]);
    assert_eq!(predicted.dtype(), DType::I64);
    assert_eq!(loss.num_elements(), 1);
    Ok(())
}
