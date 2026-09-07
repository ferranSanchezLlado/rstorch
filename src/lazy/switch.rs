//! Runtime control for deferred element-wise execution.

use std::cell::Cell;
use std::env;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::LazyLock;

/// Whether lazy element-wise execution is enabled for the current thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fusion {
    /// Materialize each operation immediately.
    #[default]
    Off,
    /// Record element-wise operations until a flush or a terminating op.
    On,
}

impl From<bool> for Fusion {
    fn from(enabled: bool) -> Fusion {
        if enabled { Fusion::On } else { Fusion::Off }
    }
}

fn process_default() -> Fusion {
    match env::var("RSTORCH_FUSION").ok().as_deref() {
        Some("on") | Some("ON") | Some("1") | Some("true") => Fusion::On,
        _ => Fusion::Off,
    }
}

static DEFAULT: LazyLock<Fusion> = LazyLock::new(process_default);

thread_local! {
    static CURRENT: Cell<Fusion> = Cell::new(*DEFAULT);
}

/// Return the process default used when a thread first enters the lazy engine.
pub fn default_fusion() -> Fusion {
    *DEFAULT
}

/// Return the current thread's setting.
pub fn fusion() -> Fusion {
    CURRENT.with(Cell::get)
}

/// Whether the current thread records pending expressions.
pub(crate) fn enabled() -> bool {
    matches!(fusion(), Fusion::On)
}

/// Temporarily set the current thread's fusion setting.
///
/// The returned guard restores the previous setting on drop. The generic
/// argument accepts both `bool` and [`Fusion`], which keeps test and embedding
/// call sites concise without introducing a process-global mutable switch.
pub fn set_fusion(value: impl Into<Fusion>) -> FusionGuard {
    let next = value.into();
    let previous = CURRENT.with(|current| {
        let previous = current.get();
        current.set(next);
        previous
    });
    FusionGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Restore guard returned by [`set_fusion`].
pub struct FusionGuard {
    previous: Fusion,
    // A thread-local setting must be restored on the thread that changed it.
    // `Rc` in the marker makes accidentally moving this guard to another
    // thread a compile-time error instead of silently restoring the wrong
    // thread's setting.
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for FusionGuard {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn guard_restores_and_threads_are_independent() {
        let before = fusion();
        {
            let _guard = set_fusion(Fusion::On);
            assert_eq!(fusion(), Fusion::On);
            let child = thread::spawn(|| {
                let child_before = fusion();
                let _guard = set_fusion(Fusion::Off);
                (child_before, fusion())
            })
            .join()
            .unwrap();
            assert_eq!(child.1, Fusion::Off);
            assert_eq!(fusion(), Fusion::On);
        }
        assert_eq!(fusion(), before);
    }
}
