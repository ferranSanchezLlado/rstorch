//! Typed-wrapper overhead: the same operation through the dynamic API and
//! through the typed API, side by side.
//!
//! This is the measurement behind the plan's section 6.2 claim that "runtime
//! typed wrappers add no measurable tensor-op regression beyond the intentional
//! exact-shape/placement validation, and no extra backend dispatch". Nothing
//! else in the bench suite compares the two surfaces, so without this the claim
//! had no evidence.
//!
//! Two sizes per operation, deliberately. The typed layer's cost is a fixed
//! per-call validation (marker comparison, dtype, device, and a canonical
//! binding check), so it is worst relative to the work done on a SMALL tensor
//! and should vanish into the noise on a large one. A single large size would
//! flatter the typed path; a single small one would overstate the cost of a
//! realistic workload.
//!
//! Sample size is 30 with a 1s warm-up, per the plan's "at least 10 warmups and
//! 30 measured samples" requirement — the rest of the suite uses 10 samples,
//! which is too few for a percentage comparison.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use rstorch::typed::prelude::*;
use rstorch::{Device, Rng, Tensor};

const SMALL: usize = 8;
const LARGE: usize = 256;

fn randn(seed: u64, dims: &[usize]) -> Tensor {
    let mut rng = Rng::seed(seed);
    Tensor::randn(dims, rstorch::DType::F32, &Device::Cpu, &mut rng).expect("bench input")
}

/// Elementwise add and a reduction, at both sizes, dynamic against typed.
fn elementwise_and_reduce(c: &mut Criterion) {
    let ctx = DeviceCtx::<Cpu>::cpu().expect("cpu context");
    let mut group = c.benchmark_group("typed_overhead");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // ---- small: the case where per-call validation is most visible ----
    let a_small = randn(1, &[SMALL, SMALL]);
    let b_small = randn(2, &[SMALL, SMALL]);
    let ta_small = Tensor2::<SMALL, SMALL>::try_from_dynamic(a_small.clone(), &ctx).unwrap();
    let tb_small = Tensor2::<SMALL, SMALL>::try_from_dynamic(b_small.clone(), &ctx).unwrap();

    group.bench_function("add_8x8_dynamic", |bench| {
        bench.iter(|| black_box(a_small.add(black_box(&b_small)).unwrap()));
    });
    group.bench_function("add_8x8_typed", |bench| {
        bench.iter(|| black_box(ta_small.add(black_box(&tb_small)).unwrap()));
    });
    group.bench_function("sum_axis_8x8_dynamic", |bench| {
        bench.iter(|| black_box(a_small.sum(1).unwrap()));
    });
    group.bench_function("sum_axis_8x8_typed", |bench| {
        bench.iter(|| black_box(ta_small.sum::<1>().unwrap()));
    });

    // ---- large: the case a realistic workload actually spends time in ----
    let a_large = randn(3, &[LARGE, LARGE]);
    let b_large = randn(4, &[LARGE, LARGE]);
    let ta_large = Tensor2::<LARGE, LARGE>::try_from_dynamic(a_large.clone(), &ctx).unwrap();
    let tb_large = Tensor2::<LARGE, LARGE>::try_from_dynamic(b_large.clone(), &ctx).unwrap();

    group.bench_function("add_256x256_dynamic", |bench| {
        bench.iter(|| black_box(a_large.add(black_box(&b_large)).unwrap()));
    });
    group.bench_function("add_256x256_typed", |bench| {
        bench.iter(|| black_box(ta_large.add(black_box(&tb_large)).unwrap()));
    });
    group.bench_function("matmul_256x256_dynamic", |bench| {
        bench.iter(|| black_box(a_large.matmul(black_box(&b_large)).unwrap()));
    });
    group.bench_function("matmul_256x256_typed", |bench| {
        bench.iter(|| black_box(ta_large.matmul(black_box(&tb_large)).unwrap()));
    });
    group.bench_function("softmax_256x256_dynamic", |bench| {
        bench.iter(|| black_box(a_large.softmax(1).unwrap()));
    });
    group.bench_function("softmax_256x256_typed", |bench| {
        bench.iter(|| black_box(ta_large.softmax::<1>().unwrap()));
    });

    group.finish();
}

criterion_group!(benches, elementwise_and_reduce);
criterion_main!(benches);
