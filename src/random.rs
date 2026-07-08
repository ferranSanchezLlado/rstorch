use crate::dtype::FloatDType;

#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            state: splitmix64(seed),
        }
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

    pub fn normal<E: FloatDType>(&mut self) -> E {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        E::from_f64(radius * theta.cos())
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
        let old = self.state;
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let word = ((old >> ((old >> 59) + 5)) ^ old).wrapping_mul(12605985483714917081);
        (word >> 43) ^ word
    }

    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::SmallRng;

    #[test]
    fn nearby_seeds_produce_different_streams() {
        let mut a = SmallRng::seed_from_u64(1);
        let mut b = SmallRng::seed_from_u64(2);
        let a_values = (0..8).map(|_| a.gen_range(usize::MAX)).collect::<Vec<_>>();
        let b_values = (0..8).map(|_| b.gen_range(usize::MAX)).collect::<Vec<_>>();
        assert_ne!(a_values, b_values);
    }

    #[test]
    fn uniform_is_in_bounds() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..10_000 {
            let value = rng.uniform(2.0f64, 5.0);
            assert!((2.0..5.0).contains(&value));
        }
    }

    #[test]
    fn normal_has_reasonable_moments() {
        let mut rng = SmallRng::seed_from_u64(7);
        let samples = (0..20_000).map(|_| rng.normal::<f64>()).collect::<Vec<_>>();
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
    fn streams_are_deterministic() {
        let mut a = SmallRng::seed_from_u64(99);
        let mut b = SmallRng::seed_from_u64(99);
        for _ in 0..128 {
            assert_eq!(a.gen_range(1_000_000), b.gen_range(1_000_000));
        }
    }
}
