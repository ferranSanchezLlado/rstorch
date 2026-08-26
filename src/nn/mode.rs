//! [`Mode`] combines layer behavior with parameter recording.
//!
//! `training` controls things such as dropout and batch-normalization
//! statistics. `record` controls whether `Param::get` creates a traced leaf.
//! Existing traced inputs still propagate a graph in either mode.

/// The behavior/recording pair threaded through every `forward`.
///
/// | want | spell it |
/// |---|---|
/// | train (dropout on, parameter recording) | [`Mode::TRAIN`] |
/// | eval (dropout off, no parameter recording) | [`Mode::EVAL`] |
/// | eval behavior with parameter recording | [`Mode::EVAL`]`.recorded()` |
/// | train behavior without parameter recording | [`Mode::TRAIN`]`.frozen()` |
///
/// Freezing a subtree is per-`Param` (`Param::freeze`), not a `Mode` side
/// effect.
///
/// # Examples
///
/// ```
/// use rstorch::nn::Mode;
///
/// let fine_tune = Mode::EVAL.recorded();
/// assert!(!fine_tune.is_training());
/// assert!(fine_tune.records());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    training: bool,
    record: bool,
}

impl Mode {
    /// Train behavior with parameter recording enabled.
    pub const TRAIN: Mode = Mode {
        training: true,
        record: true,
    };

    /// Eval behavior with parameter recording disabled.
    ///
    /// Explicitly traced inputs remain differentiable.
    pub const EVAL: Mode = Mode {
        training: false,
        record: false,
    };

    /// Force parameter recording on while keeping this mode's behavior.
    ///
    /// `Mode::EVAL.recorded()` is useful for fine-tuning with dropout disabled.
    #[must_use]
    pub fn recorded(self) -> Mode {
        Mode {
            record: true,
            ..self
        }
    }

    /// Force parameter recording off while keeping this mode's behavior.
    ///
    /// `Mode::TRAIN.frozen()` keeps training behavior without creating
    /// parameter leaves.
    #[must_use]
    pub fn frozen(self) -> Mode {
        Mode {
            record: false,
            ..self
        }
    }

    /// Whether layers should use training behavior (dropout active,
    /// `BatchNorm` using batch statistics).
    pub fn is_training(self) -> bool {
        self.training
    }

    /// Whether `Param::get` may create a traced leaf.
    pub fn records(self) -> bool {
        self.record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_constants() {
        assert!(Mode::TRAIN.is_training() && Mode::TRAIN.records());
        assert!(!Mode::EVAL.is_training() && !Mode::EVAL.records());
    }

    #[test]
    fn off_diagonals_flip_only_recording() {
        // EVAL.recorded(): eval behavior, but records.
        let ft = Mode::EVAL.recorded();
        assert!(!ft.is_training() && ft.records());
        // TRAIN.frozen(): train behavior, no parameter recording.
        let mc = Mode::TRAIN.frozen();
        assert!(mc.is_training() && !mc.records());
    }

    #[test]
    fn idempotent() {
        assert_eq!(Mode::TRAIN.recorded(), Mode::TRAIN);
        assert_eq!(Mode::EVAL.frozen(), Mode::EVAL);
    }
}
