use crate::dtype::FloatDType;

pub trait LrSchedule<E>
where
    E: FloatDType,
{
    fn lr(&self, step: usize) -> E;
}

pub struct ConstantLr<E> {
    lr: E,
}

impl<E> ConstantLr<E>
where
    E: FloatDType,
{
    pub fn new(lr: E) -> Self {
        Self { lr }
    }
}

impl<E> LrSchedule<E> for ConstantLr<E>
where
    E: FloatDType,
{
    fn lr(&self, _step: usize) -> E {
        self.lr
    }
}

pub struct StepLr<E> {
    initial_lr: E,
    gamma: E,
    step_size: usize,
}

impl<E> StepLr<E>
where
    E: FloatDType,
{
    pub fn new(initial_lr: E, gamma: E, step_size: usize) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
        }
    }
}

impl<E> LrSchedule<E> for StepLr<E>
where
    E: FloatDType,
{
    fn lr(&self, step: usize) -> E {
        let drops = step.checked_div(self.step_size).unwrap_or(0);
        (0..drops).fold(self.initial_lr, |lr, _| lr * self.gamma)
    }
}

pub struct CosineLr<E> {
    initial_lr: E,
    min_lr: E,
    total_steps: usize,
}

impl<E> CosineLr<E>
where
    E: FloatDType,
{
    pub fn new(initial_lr: E, min_lr: E, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
        }
    }
}

impl<E> LrSchedule<E> for CosineLr<E>
where
    E: FloatDType,
{
    fn lr(&self, step: usize) -> E {
        if self.total_steps == 0 {
            return self.min_lr;
        }
        let progress = (step.min(self.total_steps) as f64) / (self.total_steps as f64);
        let factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * E::from_f64(factor)
    }
}

pub struct WarmupLr<S, E> {
    inner: S,
    warmup_steps: usize,
    _dtype: std::marker::PhantomData<E>,
}

impl<S, E> WarmupLr<S, E>
where
    E: FloatDType,
    S: LrSchedule<E>,
{
    pub fn new(inner: S, warmup_steps: usize) -> Self {
        Self {
            inner,
            warmup_steps,
            _dtype: std::marker::PhantomData,
        }
    }
}

impl<S, E> LrSchedule<E> for WarmupLr<S, E>
where
    E: FloatDType,
    S: LrSchedule<E>,
{
    fn lr(&self, step: usize) -> E {
        let base = self.inner.lr(step);
        if self.warmup_steps == 0 || step >= self.warmup_steps {
            return base;
        }
        base * E::from_f64((step + 1) as f64 / self.warmup_steps as f64)
    }
}
