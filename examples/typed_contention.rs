//! Temporary measurement harness: multi-threaded typed vs dynamic throughput.
//!
//! Run with: cargo run --release --features typed --example typed_contention

use rstorch::typed::{DeviceCtx, Placement, Tensor2};
use rstorch::{DType, Device, Tensor};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const N: usize = 16;
const ITERS: usize = 20_000;
const REPS: usize = 7;

struct Bench;
impl Placement for Bench {}

fn typed_worker(iters: usize) {
    let ctx = DeviceCtx::<Bench>::bind(Device::Cpu).unwrap();
    let a = Tensor2::<N, N, f32, Bench>::from_vec(vec![1.0; N * N], [N, N], &ctx).unwrap();
    let b = Tensor2::<N, N, f32, Bench>::from_vec(vec![2.0; N * N], [N, N], &ctx).unwrap();
    let mut sink = 0.0f64;
    for _ in 0..iters {
        let c = a.add(&b).unwrap();
        let d: Tensor2<N, N, f32, Bench> = a.matmul(&b).unwrap();
        sink += c.dims()[0] as f64 + d.dims()[1] as f64;
    }
    assert!(sink > 0.0);
}

fn dynamic_worker(iters: usize) {
    let a = Tensor::full([N, N], 1.0, DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::full([N, N], 2.0, DType::F32, &Device::Cpu).unwrap();
    let mut sink = 0.0f64;
    for _ in 0..iters {
        let c = a.add(&b).unwrap();
        let d = a.matmul(&b).unwrap();
        sink += c.dims()[0] as f64 + d.dims()[1] as f64;
    }
    assert!(sink > 0.0);
}

fn run(threads: usize, worker: fn(usize)) -> Duration {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                worker(ITERS);
            })
        })
        .collect();
    barrier.wait();
    let start = Instant::now();
    for handle in handles {
        handle.join().unwrap();
    }
    start.elapsed()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(label: &str, worker: fn(usize)) {
    // Warm up (binds the marker, warms allocators).
    worker(500);
    let mut base = 0.0;
    for &threads in &[1usize, 2, 4, 8] {
        let throughput = median(
            (0..REPS)
                .map(|_| {
                    let elapsed = run(threads, worker);
                    (threads * ITERS) as f64 / elapsed.as_secs_f64()
                })
                .collect(),
        );
        if threads == 1 {
            base = throughput;
        }
        println!(
            "{label:>8} {threads:>2} threads: {:>12.0} iters/s  scaling {:>5.2}x  {:>7.0} ns/iter/thread",
            throughput,
            throughput / base,
            1e9 / (throughput / threads as f64),
        );
    }
}

fn main() {
    println!(
        "{}x{} f32, {ITERS} iters/thread (1 add + 1 matmul each), median of {REPS} runs",
        N, N
    );
    measure("typed", typed_worker);
    measure("dynamic", dynamic_worker);
}
