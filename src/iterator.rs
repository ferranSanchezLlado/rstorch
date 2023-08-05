use std::iter::FusedIterator;
use std::ops::ControlFlow;

pub(crate) trait IteratorExt: Iterator {
    #[inline]
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

    #[inline]
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

    #[inline]
    fn array_chunks_costum<const N: usize>(self) -> ArrayChunks<Self, N>
    where
        Self: Sized,
    {
        ArrayChunks::new(self)
    }

    #[inline]
    fn filter_option<P>(self, predicate: P) -> OptionFilter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> Option<bool>,
    {
        OptionFilter::new(self, predicate)
    }
}

#[derive(Clone)]
pub struct ArrayChunks<I, const N: usize> {
    iter: I,
}

impl<I, const N: usize> ArrayChunks<I, N> {
    #[inline]
    #[must_use]
    pub(crate) fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I: Iterator, const N: usize> Iterator for ArrayChunks<I, N> {
    type Item = [I::Item; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (0..N)
            .map(|_| self.iter.next())
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()
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
impl<I: DoubleEndedIterator, const N: usize> DoubleEndedIterator for ArrayChunks<I, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        (0..N)
            .map(|_| self.iter.next_back())
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()
    }
}
impl<I, const N: usize> FusedIterator for ArrayChunks<I, N> where I: FusedIterator {}
impl<I: ExactSizeIterator, const N: usize> ExactSizeIterator for ArrayChunks<I, N> {}

#[derive(Clone)]
pub struct OptionFilter<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> OptionFilter<I, P> {
    #[inline]
    #[must_use]
    pub(crate) fn new(iter: I, predicate: P) -> Self {
        Self { iter, predicate }
    }
}

impl<I: Iterator, P> Iterator for OptionFilter<I, P>
where
    P: FnMut(&I::Item) -> Option<bool>,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.iter.next();
            if let Some(true) = item.as_ref().map(&mut self.predicate)? {
                return item;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();

        (0, upper) // can't know a lower bound, due to the predicate
    }
}

impl<I: DoubleEndedIterator, P> DoubleEndedIterator for OptionFilter<I, P>
where
    P: FnMut(&I::Item) -> Option<bool>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.iter.next_back();
            if let Some(true) = item.as_ref().map(&mut self.predicate)? {
                return item;
            }
        }
    }
}
impl<I: FusedIterator, P: FnMut(&I::Item) -> Option<bool>> FusedIterator for OptionFilter<I, P> {}

impl<I: ?Sized> IteratorExt for I where I: Iterator {}
