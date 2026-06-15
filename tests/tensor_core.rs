#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

#[test]
fn constructors_own_data_and_report_static_shape() {
    let source = [1.0_f32, 2.0, 3.0];
    let tensor = Tensor1D::<3>::from_array(source);

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.numel(), 3);
    assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn default_dtype_is_f32() {
    let tensor: Tensor2D<2, 2> = Tensor2D::ones();
    let values: Vec<f32> = tensor.to_vec();

    assert_eq!(values, vec![1.0; 4]);
}

#[test]
fn from_vec_rejects_wrong_lengths() {
    let error = match Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0]) {
        Ok(_) => panic!("from_vec accepted the wrong length"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        TensorError::InvalidLength {
            expected: 6,
            actual: 2,
        }
    );
}
