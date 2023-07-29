use std::ops::ControlFlow;

pub(crate) trait IteratorExt: Iterator {
    fn all_default<F>(&mut self, default: bool, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        #[inline]
        fn check<T>(mut f: impl FnMut(T) -> bool) -> impl FnMut(bool, T) -> ControlFlow<(), bool> {
            move |_, x| {
                if f(x) {
                    ControlFlow::Continue(true)
                } else {
                    ControlFlow::Break(())
                }
            }
        }
        self.try_fold(default, check(f)) == ControlFlow::Continue(true)
    }

    fn any_default<F>(&mut self, default: bool, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        #[inline]
        fn check<T>(mut f: impl FnMut(T) -> bool) -> impl FnMut(bool, T) -> ControlFlow<(), bool> {
            move |_, x| {
                if f(x) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(false)
                }
            }
        }
        self.try_fold(default, check(f)) != ControlFlow::Continue(false)
    }

    fn array_chunks_costum<const N: usize>(self) -> ArrayChunks<Self, N>
    where
        Self: Sized,
    {
        ArrayChunks::new(self)
    }
}

pub(crate) struct ArrayChunks<I, const N: usize> {
    iter: I,
}

impl<I, const N: usize> ArrayChunks<I, N> {
    pub(crate) fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I: Iterator, const N: usize> Iterator for ArrayChunks<I, N> {
    type Item = [I::Item; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mut data = Vec::with_capacity(N);
        for _ in 0..N {
            data.push(self.iter.next()?)
        }

        data.try_into().ok()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        (lower / N, upper.map(|n| n / N))
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count() / N
    }
}

impl<I: ?Sized> IteratorExt for I where I: Iterator {}
