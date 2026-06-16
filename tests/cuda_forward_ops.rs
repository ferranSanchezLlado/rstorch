#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

const TOLERANCE: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "index {index}: {actual} != {expected}"
        );
    }
}

fn skip_without_cuda() -> bool {
    if Cuda::is_available() {
        false
    } else {
        eprintln!("skipping CUDA test: no CUDA device is available");
        true
    }
}

#[test]
fn cuda_constructs_and_round_trips_values() {
    if skip_without_cuda() {
        return;
    }

    let zeros = Tensor1D::<3, f32, Cuda>::zeros();
    let ones = Tensor1D::<3, f32, Cuda>::ones();
    let values = Tensor1D::<3, f32, Cuda>::from_array([2.0, 4.0, 8.0]);

    assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0]);
    assert_eq!(ones.to_vec(), vec![1.0, 1.0, 1.0]);
    assert_eq!(values.to_vec(), vec![2.0, 4.0, 8.0]);
}

#[test]
fn cuda_elementwise_ops_match_cpu_values() {
    if skip_without_cuda() {
        return;
    }

    let lhs = Tensor1D::<4, f32, Cuda>::from_array([8.0, 6.0, 4.0, 2.0]);
    let rhs = Tensor1D::<4, f32, Cuda>::from_array([4.0, 3.0, 2.0, 1.0]);

    assert_eq!(lhs.add(&rhs).to_vec(), vec![12.0, 9.0, 6.0, 3.0]);
    assert_eq!(lhs.sub(&rhs).to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    assert_eq!(lhs.mul(&rhs).to_vec(), vec![32.0, 18.0, 8.0, 2.0]);
    assert_eq!(lhs.div(&rhs).to_vec(), vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn cuda_scalar_and_unary_ops_match_cpu_values() {
    if skip_without_cuda() {
        return;
    }

    let tensor = Tensor1D::<3, f32, Cuda>::from_array([2.0, 4.0, 8.0]);

    assert_eq!(tensor.add_scalar(1.0).to_vec(), vec![3.0, 5.0, 9.0]);
    assert_eq!(tensor.sub_scalar(1.0).to_vec(), vec![1.0, 3.0, 7.0]);
    assert_eq!(tensor.mul_scalar(2.0).to_vec(), vec![4.0, 8.0, 16.0]);
    assert_eq!(tensor.div_scalar(2.0).to_vec(), vec![1.0, 2.0, 4.0]);
    assert_eq!(tensor.powf(2.0).to_vec(), vec![4.0, 16.0, 64.0]);

    let activations = Tensor1D::<4, f32, Cuda>::from_array([-2.0, 3.0, -5.0, 7.0]);
    assert_eq!(activations.relu().to_vec(), vec![0.0, 3.0, 0.0, 7.0]);

    let positive = Tensor1D::<2, f32, Cuda>::from_array([1.0, 4.0]);
    assert_close(&positive.exp().ln().to_vec(), &[1.0, 4.0]);
}

#[test]
fn cuda_reductions_match_cpu_values() {
    if skip_without_cuda() {
        return;
    }

    let activations = Tensor1D::<4, f32, Cuda>::from_array([-2.0, 3.0, -5.0, 7.0]);
    assert_eq!(activations.relu().sum().to_vec(), vec![10.0]);
    assert_eq!(activations.mean().to_vec(), vec![0.75]);
}

#[test]
fn cuda_matmul_and_transpose_match_cpu_values() {
    if skip_without_cuda() {
        return;
    }

    let lhs = Tensor2D::<2, 3, f32, Cuda>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor2D::<3, 2, f32, Cuda>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    assert_eq!(lhs.matmul(&rhs).to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    assert_eq!(lhs.transpose().to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn cuda_row_and_column_addition_match_cpu_values() {
    if skip_without_cuda() {
        return;
    }

    let tensor = Tensor2D::<2, 3, f32, Cuda>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let row = Tensor1D::<3, f32, Cuda>::from_array([10.0, 20.0, 30.0]);
    let col = Tensor1D::<2, f32, Cuda>::from_array([100.0, 200.0]);

    assert_eq!(
        tensor.add_row(&row).to_vec(),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
    assert_eq!(
        tensor.add_col(&col).to_vec(),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );
}

#[test]
fn cuda_autograd_uses_backend_ops() {
    if skip_without_cuda() {
        return;
    }

    let weights = Tensor2D::<3, 2, f32, Cuda>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let input =
        Tensor2D::<2, 3, f32, Cuda>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).requires_grad();

    input.matmul(&weights).mean().backward();

    assert_close(
        &input.grad().unwrap().to_vec(),
        &[3.75, 4.75, 5.75, 3.75, 4.75, 5.75],
    );
}
