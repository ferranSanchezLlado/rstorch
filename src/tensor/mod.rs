use std::ptr::NonNull;

pub(crate) struct RawCpuTensor<T = f64> {
    data: NonNull<T>,
    shape: Vec<usize>,
}

pub struct Tensor1D<const N: usize, T = f64> {
    raw: RawCpuTensor<T>,
}

impl<const N: usize, T> Tensor1D<N, T> {
    pub fn new(data: [T; N]) -> Self {
        let raw = RawCpuTensor {
            data: NonNull::new(data.as_ptr() as *mut T).expect("Data pointer cannot be null"),
            shape: vec![N],
        };
        Self { raw }
    }
}

pub struct Tensor2D<const M: usize, const N: usize, T = f64> {
    raw: RawCpuTensor<T>,
}

impl<const M: usize, const N: usize, T> Tensor2D<M, N, T> {
    pub fn new(data: [[T; N]; M]) -> Self {
        let raw = RawCpuTensor {
            data: NonNull::new(data.as_ptr() as *mut T).expect("Data pointer cannot be null"),
            shape: vec![M, N],
        };
        Self { raw }
    }
}
