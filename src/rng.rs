//! Explicit, seedable, splittable random number generation (exploration
//! §4.4). No global seed, no ambient state: stochastic constructors take
//! `&mut Rng`, and RNG state is checkpointed alongside optimizer state for
//! resumable training.
//!
//! **Contract file** (T01). The API and doc contracts are frozen here;
//! **T12** ports the post-16.2 `SmallRng` (splitmix64-seeded, `u64` state,
//! tested and state-serializable) into these bodies without changing a
//! signature. The original untested LCG is deliberately **not** carried
//! forward.

// The `state` field and the stubbed generators are consumed by T12; the
// integrator removes this allow when T12 fills the bodies.
#![allow(dead_code)]

/// A seedable, splittable, state-serializable pseudo-random generator.
///
/// Determinism is a contract: a given seed reproduces an exact sequence, and
/// [`state`](Rng::state)/[`from_state`](Rng::from_state) round-trips resume it
/// bit-for-bit (gated by T12's determinism fixture).
#[derive(Clone, Debug)]
pub struct Rng {
    /// The full generator state (splitmix64 register). One `u64` is the
    /// entire serialized form.
    state: u64,
}

impl Rng {
    /// Create a generator seeded from `seed` (mixed through splitmix64 so
    /// even low-entropy seeds like `0` and `1` produce well-separated
    /// streams). T12 fills the body.
    pub fn seed(seed: u64) -> Rng {
        let _ = seed;
        todo!("T12: seed via splitmix64")
    }

    /// Reconstruct a generator from a previously captured
    /// [`state`](Rng::state) (checkpoint restore). T12 fills the body.
    pub fn from_state(state: u64) -> Rng {
        let _ = state;
        todo!("T12: from_state")
    }

    /// The current serializable state. Paired with
    /// [`from_state`](Rng::from_state) for checkpointing. T12 fills the body.
    pub fn state(&self) -> u64 {
        todo!("T12: state")
    }

    /// Split off an independent child generator, advancing `self`
    /// (`Dropout::new` seeds its own stream this way). T12 fills the body.
    pub fn split(&mut self) -> Rng {
        todo!("T12: split")
    }

    /// A uniform sample in `[low, high)`. T12 fills the body.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let _ = (low, high);
        todo!("T12: uniform")
    }

    /// A normal (Gaussian) sample with the given `mean` and standard
    /// deviation `std`. T12 fills the body.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let _ = (mean, std);
        todo!("T12: normal")
    }

    /// Draw a raw 64-bit word (kernel/constructor fill path). T12 fills the
    /// body.
    pub(crate) fn next_u64(&mut self) -> u64 {
        todo!("T12: next_u64")
    }
}
