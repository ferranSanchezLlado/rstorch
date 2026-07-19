//! Explicit, seedable, splittable random number generation (exploration
//! §4.4). No global seed, no ambient state: stochastic constructors take
//! `&mut Rng`, and RNG state is checkpointed alongside optimizer state for
//! resumable training.
//!
//! **Contract file** (T01), bodies filled by **T12**. The generator is the
//! post-16.2 `SmallRng` ported from v2: a `u64` register seeded through
//! splitmix64, advanced by an LCG whose output word is scrambled into the
//! returned `u64`. The original untested LCG from v0.x is deliberately **not**
//! carried forward.

/// The splitmix64 mixing function used to diffuse a raw seed (or a raw draw,
/// when splitting) into a well-separated `u64` register. Low-entropy inputs
/// such as `0` and `1` map to unrelated outputs.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

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
        Rng {
            state: splitmix64(seed),
        }
    }

    /// Reconstruct a generator from a previously captured
    /// [`state`](Rng::state) (checkpoint restore). The resumed generator
    /// produces exactly the sequence the original would have from that point.
    pub fn from_state(state: u64) -> Rng {
        Rng { state }
    }

    /// The current serializable state. Paired with
    /// [`from_state`](Rng::from_state) for checkpointing.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Split off an independent child generator, advancing `self`
    /// (`Dropout::new` seeds its own stream this way).
    ///
    /// The child is seeded by drawing one raw word from `self` (which advances
    /// the parent) and diffusing it through splitmix64, so the parent and child
    /// streams are decorrelated.
    pub fn split(&mut self) -> Rng {
        let child_seed = self.next_u64();
        Rng {
            state: splitmix64(child_seed),
        }
    }

    /// A uniform sample in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_unit()
    }

    /// A normal (Gaussian) sample with the given `mean` and standard
    /// deviation `std`.
    ///
    /// Uses the Box–Muller transform on two uniform draws.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        mean + std * radius * theta.cos()
    }

    /// Draw a raw 64-bit word (kernel/constructor fill path).
    ///
    /// Advances the LCG register and scrambles the previous state into the
    /// returned word; this is the primitive all other generators build on.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let word = ((old >> ((old >> 59) + 5)) ^ old).wrapping_mul(12605985483714917081);
        (word >> 43) ^ word
    }

    /// A raw draw mapped into `[0, 1)` with 53 bits of mantissa precision.
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    // ---- ported v2 tests (adapted to the frozen `Rng` API) --------------

    #[test]
    fn nearby_seeds_produce_different_streams() {
        let mut a = Rng::seed(1);
        let mut b = Rng::seed(2);
        let a_values = (0..8).map(|_| a.next_u64()).collect::<Vec<_>>();
        let b_values = (0..8).map(|_| b.next_u64()).collect::<Vec<_>>();
        assert_ne!(a_values, b_values);
    }

    #[test]
    fn uniform_is_in_bounds() {
        let mut rng = Rng::seed(42);
        for _ in 0..10_000 {
            let value = rng.uniform(2.0, 5.0);
            assert!((2.0..5.0).contains(&value), "value={value}");
        }
    }

    #[test]
    fn normal_has_reasonable_moments() {
        let mut rng = Rng::seed(7);
        let samples = (0..20_000)
            .map(|_| rng.normal(0.0, 1.0))
            .collect::<Vec<_>>();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / samples.len() as f64;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.08, "var={var}");
    }

    #[test]
    fn normal_shifts_and_scales() {
        // A non-standard normal is the standard draw scaled by `std` and
        // shifted by `mean`; verify against a same-seed standard stream.
        let mut a = Rng::seed(1234);
        let mut b = Rng::seed(1234);
        for _ in 0..1_000 {
            let standard = a.normal(0.0, 1.0);
            let scaled = b.normal(3.0, 2.0);
            assert!((scaled - (3.0 + 2.0 * standard)).abs() < 1e-12);
        }
    }

    #[test]
    fn streams_are_deterministic() {
        let mut a = Rng::seed(99);
        let mut b = Rng::seed(99);
        for _ in 0..128 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    // ---- determinism fixture (T12 gate) ---------------------------------

    /// Fixed seed reproduces an exact word sequence. These golden values pin
    /// the ported splitmix64-seeded generator; any change to the algorithm
    /// (a checkpoint-incompatible change) must fail here loudly.
    #[test]
    fn fixed_seed_reproduces_exact_sequence() {
        let mut rng = Rng::seed(0x00C0_FFEE);
        let words: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(
            words,
            [
                0x0a67_534d_349a_4c7a,
                0xc4be_6434_d0a3_85a0,
                0x7f78_9b62_cf0c_04f8,
                0x121c_690b_1995_196b,
            ]
        );

        let mut rng = Rng::seed(0x00C0_FFEE);
        let uniforms: Vec<f64> = (0..4).map(|_| rng.uniform(0.0, 1.0)).collect();
        assert_eq!(
            uniforms,
            [
                0.040_639_120_434_755_26,
                0.768_530_142_683_301_8,
                0.497_934_066_413_151_3,
                0.070_746_007_165_855_55,
            ]
        );

        let mut rng = Rng::seed(0x00C0_FFEE);
        let normals: Vec<f64> = (0..4).map(|_| rng.normal(0.0, 1.0)).collect();
        assert_eq!(
            normals,
            [
                0.294_016_772_997_423_85,
                1.066_161_016_484_111,
                0.133_898_689_906_669_23,
                0.112_038_442_909_365_75,
            ]
        );
    }

    /// A captured state resumes the sequence bit-for-bit via `from_state`, and
    /// `state()`/`from_state` is a faithful round trip.
    #[test]
    fn state_round_trip_resumes_identically() {
        let mut rng = Rng::seed(0xDEAD_BEEF);
        // advance an arbitrary amount, then snapshot
        for _ in 0..17 {
            let _ = rng.next_u64();
        }
        let snapshot = rng.state();

        // continue from the live generator
        let expected: Vec<u64> = (0..32).map(|_| rng.next_u64()).collect();

        // a generator rebuilt from the snapshot produces the same tail
        let mut resumed = Rng::from_state(snapshot);
        let actual: Vec<u64> = (0..32).map(|_| resumed.next_u64()).collect();
        assert_eq!(expected, actual);

        // state()/from_state is an exact round trip on its own
        let a = Rng::from_state(snapshot);
        assert_eq!(a.state(), snapshot);
    }

    /// `split` yields a child stream that is independent of the parent's
    /// continuation, and advances the parent (so re-splitting differs).
    #[test]
    fn split_streams_are_independent() {
        let mut parent = Rng::seed(2024);
        let mut child = parent.split();

        let child_stream: Vec<u64> = (0..64).map(|_| child.next_u64()).collect();
        let parent_stream: Vec<u64> = (0..64).map(|_| parent.next_u64()).collect();
        assert_ne!(child_stream, parent_stream);

        // splitting again from a parent advanced past the first split gives a
        // different child stream (the parent state moved on).
        let mut parent2 = Rng::seed(2024);
        let mut first = parent2.split();
        let mut second = parent2.split();
        let first_stream: Vec<u64> = (0..64).map(|_| first.next_u64()).collect();
        let second_stream: Vec<u64> = (0..64).map(|_| second.next_u64()).collect();
        assert_ne!(first_stream, second_stream);

        // split is deterministic: same seed => same child.
        let mut p_a = Rng::seed(555);
        let mut p_b = Rng::seed(555);
        let mut c_a = p_a.split();
        let mut c_b = p_b.split();
        let s_a: Vec<u64> = (0..16).map(|_| c_a.next_u64()).collect();
        let s_b: Vec<u64> = (0..16).map(|_| c_b.next_u64()).collect();
        assert_eq!(s_a, s_b);
    }
}
