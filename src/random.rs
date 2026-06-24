use crate::dtype::FloatDType;

#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn uniform<E: FloatDType>(&mut self, low: E, high: E) -> E {
        low + (high - low) * E::from_f64(self.next_unit())
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
