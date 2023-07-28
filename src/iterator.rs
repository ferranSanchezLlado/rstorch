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
}

impl<I: ?Sized> IteratorExt for I where I: Iterator {}
