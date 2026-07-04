use crate::dtype::FloatDType;

#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }

    pub fn uniform<E: FloatDType>(&mut self, low: E, high: E) -> E {
        low + (high - low) * E::from_f64(self.next_unit())
    }

    /// Samples uniformly from `0..upper`.
    ///
    /// # Panics
    ///
    /// Panics when `upper == 0`.
    pub fn gen_range(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be greater than zero");
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
