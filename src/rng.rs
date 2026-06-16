//! Small deterministic random number generator for tests and initialization.

/// Deterministic, seedable PRNG for initialization and tests.
///
/// This generator is intentionally small and has no external dependencies. It is
/// not cryptographically secure and should not be used for security-sensitive
/// randomness.
#[derive(Clone, Debug)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    /// Creates a deterministic generator from a 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };

        Self { state }
    }

    /// Returns the next raw `u64` from the generator.
    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    /// Samples uniformly from `[low, high)` as `f32`.
    pub fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        assert!(low <= high, "uniform range must satisfy low <= high");
        if low == high {
            return low;
        }

        let unit = ((self.next_u64() >> 40) as f32) * (1.0 / (1_u32 << 24) as f32);
        low + (high - low) * unit
    }

    /// Samples uniformly from `[low, high)` as `f64`.
    pub fn uniform_f64(&mut self, low: f64, high: f64) -> f64 {
        assert!(low <= high, "uniform range must satisfy low <= high");
        if low == high {
            return low;
        }

        let unit = ((self.next_u64() >> 11) as f64) * (1.0 / (1_u64 << 53) as f64);
        low + (high - low) * unit
    }
}
