//! Synchronized CPU-vs-CUDA kernel-family benchmarks.
//!
//! Every measured operation ends in a host read so CUDA work is measured
//! through completion rather than only through asynchronous submission.

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn main() {}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod supported {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{Criterion, SamplingMode};
    use half::f16;
    use rstorch::prelude::*;

    fn values(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 257) as f32 - 128.0) * scale)
            .collect()
    }

    fn tensor(dims: &[usize], device: &Device) -> Tensor {
        Tensor::from_vec(
            values(dims.iter().product(), 1.0 / 128.0),
            dims.to_vec(),
            device,
        )
        .unwrap()
    }

    fn finish(tensor: Tensor) {
        black_box(tensor.to_vec::<f32>().unwrap());
    }

    fn finish_f16(tensor: Tensor) {
        black_box(tensor.to_vec::<f16>().unwrap());
    }

    fn bench_device(c: &mut Criterion, device: Device) {
        let mut group = c.benchmark_group(format!("cuda_ops/{device}"));
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));

        let a = tensor(&[1024, 1024], &device);
        let b = tensor(&[1024, 1024], &device);
        group.bench_function("elementwise_add/1m", |bench| {
            bench.iter(|| finish(black_box(&a).add(black_box(&b)).unwrap()));
        });
        group.bench_function("transpose_copy/1024x1024", |bench| {
            bench.iter(|| finish(black_box(&a).transpose(0, 1).unwrap().contiguous().unwrap()));
        });
        group.bench_function("reduce_last/1024x1024", |bench| {
            bench.iter(|| finish(black_box(&a).sum(-1).unwrap()));
        });
        group.bench_function("reduce_all/1024x1024", |bench| {
            bench.iter(|| finish(black_box(&a).sum_all().unwrap()));
        });
        group.bench_function("softmax_last/1024x1024", |bench| {
            bench.iter(|| finish(black_box(&a).softmax(-1).unwrap()));
        });

        let mut norm = LayerNorm::new([1024], &device).unwrap();
        group.bench_function("layer_norm/1024x1024", |bench| {
            bench.iter(|| finish(norm.forward(black_box(&a), Mode::EVAL).unwrap()));
        });

        let lhs = tensor(&[512, 512], &device);
        let rhs = tensor(&[512, 512], &device);
        group.bench_function("matmul/512x512", |bench| {
            bench.iter(|| finish(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
        });

        let table = tensor(&[2048, 128], &device).traced().unwrap();
        let ids = Tensor::from_vec(
            (0..128).map(|i| ((i * 37) % 2048) as i64).collect(),
            [128],
            &device,
        )
        .unwrap();
        group.bench_function("embedding_backward/2048x128_128ids", |bench| {
            bench.iter(|| {
                let loss = table
                    .index_select(0, black_box(&ids))
                    .unwrap()
                    .sum_all()
                    .unwrap();
                finish(loss.backward().unwrap().wrt_input(&table).unwrap());
            });
        });

        let logits = tensor(&[4096, 512], &device).traced().unwrap();
        let classes = Tensor::from_vec(
            (0..4096).map(|row| (row % 512) as i64).collect(),
            [4096, 1],
            &device,
        )
        .unwrap();
        group.bench_function("gather_backward/4096x512", |bench| {
            bench.iter(|| {
                let loss = logits
                    .gather(1, black_box(&classes))
                    .unwrap()
                    .sum_all()
                    .unwrap();
                finish(loss.backward().unwrap().wrt_input(&logits).unwrap());
            });
        });

        let image = tensor(&[4, 16, 28, 28], &device);
        let weight = tensor(&[32, 16, 3, 3], &device);
        group.bench_function("conv2d/4x16x28x28_32x3x3", |bench| {
            bench.iter(|| {
                finish(
                    black_box(&image)
                        .conv2d(black_box(&weight), (1, 1), (1, 1), (1, 1))
                        .unwrap(),
                );
            });
        });

        let a16 = a.to_dtype(DType::F16).unwrap();
        let b16 = b.to_dtype(DType::F16).unwrap();
        group.bench_function("f16/elementwise_add/1m", |bench| {
            bench.iter(|| finish_f16(black_box(&a16).add(black_box(&b16)).unwrap()));
        });
        group.bench_function("f16/reduce_last/1024x1024", |bench| {
            bench.iter(|| finish_f16(black_box(&a16).sum(-1).unwrap()));
        });
        group.bench_function("f16/softmax_last/1024x1024", |bench| {
            bench.iter(|| finish_f16(black_box(&a16).softmax(-1).unwrap()));
        });

        let lhs16 = lhs.to_dtype(DType::F16).unwrap();
        let rhs16 = rhs.to_dtype(DType::F16).unwrap();
        group.bench_function("f16/matmul/512x512", |bench| {
            bench.iter(|| finish_f16(black_box(&lhs16).matmul(black_box(&rhs16)).unwrap()));
        });

        group.finish();
    }

    pub fn run() {
        let mut criterion = Criterion::default().configure_from_args();
        bench_device(&mut criterion, Device::Cpu);
        bench_device(&mut criterion, Device::Cuda(0));
        criterion.final_summary();
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn main() {
    supported::run();
}
