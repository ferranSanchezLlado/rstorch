#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

#[test]
fn cpu_forward_ops_compute_expected_values() {
    let lhs = Tensor1D::<4>::from_array([8.0, 6.0, 4.0, 2.0]);
    let rhs = Tensor1D::<4>::from_array([4.0, 3.0, 2.0, 1.0]);

    assert_eq!(lhs.add(&rhs).to_vec(), vec![12.0, 9.0, 6.0, 3.0]);
    assert_eq!(lhs.sub(&rhs).to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    assert_eq!(lhs.mul(&rhs).to_vec(), vec![32.0, 18.0, 8.0, 2.0]);
    assert_eq!(lhs.div(&rhs).to_vec(), vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn matmul_and_reductions_compute_expected_values() {
    let lhs = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor2D::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let product = lhs.matmul(&rhs);

    assert_eq!(product.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    assert_eq!(product.sum().to_vec(), vec![415.0]);
    assert_eq!(product.mean().to_vec(), vec![103.75]);
}

#[test]
fn explicit_row_and_column_addition_compute_expected_values() {
    let tensor = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let row = Tensor1D::<3>::from_array([10.0, 20.0, 30.0]);
    let col = Tensor1D::<2>::from_array([100.0, 200.0]);

    assert_eq!(
        tensor.add_row(&row).to_vec(),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
    assert_eq!(
        tensor.add_col(&col).to_vec(),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );
}
