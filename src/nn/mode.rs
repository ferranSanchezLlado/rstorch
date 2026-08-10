//! [`Mode`] — two orthogonal axes in one `Copy` value.
//!
//! `Mode` carries **layer behavior** (Train vs Eval — dropout on/off,
//! BatchNorm batch-stats vs running-stats) and **recording** (whether
//! `Param::get` hands back a traced leaf) as *independent* axes, because
//! conflating them makes standard flows inexpressible. The diagonal
//! constants cover hour one; the off-diagonals are one call away.

/// The behavior/recording pair threaded through every `forward`.
///
/// | want | spell it |
/// |---|---|
/// | train (dropout on, recording) | [`Mode::TRAIN`] |
/// | eval (dropout off, no recording) | [`Mode::EVAL`] |
/// | fine-tune (eval behavior, but record) | [`Mode::EVAL`]`.recorded()` |
/// | MC-dropout sample (train behavior, no graph) | [`Mode::TRAIN`]`.frozen()` |
///
/// Freezing a *subtree* is per-`Param` (`Param::freeze`), never a `Mode`
/// side effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    training: bool,
    record: bool,
}

impl Mode {
    /// Train behavior **and** gradient recording.
    pub const TRAIN: Mode = Mode {
        training: true,
        record: true,
    };

    /// Eval behavior **and** no recording (inference retains no
    /// activations, which is the memory win).
    pub const EVAL: Mode = Mode {
        training: false,
        record: false,
    };

    /// This mode's behavior with recording forced **on**
    /// (`Mode::EVAL.recorded()` = frozen-BatchNorm/dropout-off fine-tuning).
    pub fn recorded(self) -> Mode {
        Mode {
            record: true,
            ..self
        }
    }

    /// This mode's behavior with recording forced **off**
    /// (`Mode::TRAIN.frozen()` = MC-dropout sampling with no graph cost).
    pub fn frozen(self) -> Mode {
        Mode {
            record: false,
            ..self
        }
    }

    /// Whether layers should use **training** behavior (dropout active,
    /// BatchNorm using batch statistics).
    pub fn is_training(self) -> bool {
        self.training
    }

    /// Whether computation is **recorded** — the condition (with
    /// `!Param::is_frozen`) under which `Param::get` returns a traced leaf.
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
        // TRAIN.frozen(): train behavior, no recording.
        let mc = Mode::TRAIN.frozen();
        assert!(mc.is_training() && !mc.records());
    }

    #[test]
    fn idempotent() {
        assert_eq!(Mode::TRAIN.recorded(), Mode::TRAIN);
        assert_eq!(Mode::EVAL.frozen(), Mode::EVAL);
    }
}
