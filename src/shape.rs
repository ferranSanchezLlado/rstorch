//! Compile-time tensor shape markers.

use std::marker::PhantomData;

/// Tensor shape known at compile time.
pub trait Shape: Clone + Copy + Send + Sync + 'static {
    const RANK: usize;
    const NUMEL: usize;

    fn dims() -> &'static [usize];
}

/// Scalar shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct D0;

/// One-dimensional shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct D1<const N: usize>(PhantomData<()>);

/// Two-dimensional shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct D2<const M: usize, const N: usize>(PhantomData<()>);

/// Three-dimensional shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct D3<const A: usize, const B: usize, const C: usize>(PhantomData<()>);

impl Shape for D0 {
    const RANK: usize = 0;
    const NUMEL: usize = 1;

    fn dims() -> &'static [usize] {
        &[]
    }
}

impl<const N: usize> Shape for D1<N> {
    const RANK: usize = 1;
    const NUMEL: usize = N;

    fn dims() -> &'static [usize] {
        &[N]
    }
}

impl<const M: usize, const N: usize> Shape for D2<M, N> {
    const RANK: usize = 2;
    const NUMEL: usize = M * N;

    fn dims() -> &'static [usize] {
        &[M, N]
    }
}

impl<const A: usize, const B: usize, const C: usize> Shape for D3<A, B, C> {
    const RANK: usize = 3;
    const NUMEL: usize = A * B * C;

    fn dims() -> &'static [usize] {
        &[A, B, C]
    }
}

#[cfg(test)]
mod tests {
    use super::{D0, D1, D2, D3, Shape};

    #[test]
    fn shape_metadata_is_compile_time_backed() {
        assert_eq!(D0::RANK, 0);
        assert_eq!(D0::NUMEL, 1);
        assert_eq!(D0::dims(), &[]);

        assert_eq!(D1::<4>::RANK, 1);
        assert_eq!(D1::<4>::NUMEL, 4);
        assert_eq!(D1::<4>::dims(), &[4]);

        assert_eq!(D2::<2, 3>::RANK, 2);
        assert_eq!(D2::<2, 3>::NUMEL, 6);
        assert_eq!(D2::<2, 3>::dims(), &[2, 3]);

        assert_eq!(D3::<2, 3, 4>::RANK, 3);
        assert_eq!(D3::<2, 3, 4>::NUMEL, 24);
        assert_eq!(D3::<2, 3, 4>::dims(), &[2, 3, 4]);
    }
}
