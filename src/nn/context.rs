use crate::random::SmallRng;

pub trait TrainingMode {
    fn is_training(&self) -> bool;
}

pub trait RngSource {
    fn rng(&mut self) -> &mut SmallRng;
}

pub struct TrainContext {
    training: bool,
    rng: SmallRng,
}

impl TrainContext {
    pub fn training(seed: u64) -> Self {
        Self {
            training: true,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn eval() -> Self {
        Self {
            training: false,
            rng: SmallRng::seed_from_u64(0),
        }
    }
}

impl TrainingMode for TrainContext {
    fn is_training(&self) -> bool {
        self.training
    }
}

impl RngSource for TrainContext {
    fn rng(&mut self) -> &mut SmallRng {
        &mut self.rng
    }
}

impl TrainingMode for () {
    fn is_training(&self) -> bool {
        false
    }
}
