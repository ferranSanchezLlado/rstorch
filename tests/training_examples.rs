#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

struct XorMlp {
    hidden: Linear<2, 8>,
    output: Linear<8, 1>,
}

impl XorMlp {
    fn new(rng: &mut SmallRng) -> Self {
        Self {
            hidden: Linear::kaiming_uniform(rng),
            output: Linear::xavier_uniform(rng),
        }
    }

    fn forward<const BATCH: usize>(&self, input: &Tensor2D<BATCH, 2>) -> Tensor2D<BATCH, 1> {
        self.output
            .forward(&self.hidden.forward(input).tanh())
            .sigmoid()
    }

    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<f32, Cpu>> {
        let mut parameters = self.hidden.parameters_mut();
        parameters.extend(self.output.parameters_mut());
        parameters
    }

    fn zero_grad(&mut self) {
        self.hidden.zero_grad();
        self.output.zero_grad();
    }
}

#[test]
fn seeded_tiny_regression_loss_decreases() {
    let mut rng = SmallRng::seed_from_u64(7);
    let mut layer = Linear::<1, 1>::xavier_uniform(&mut rng);
    let input = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [2.0], [3.0]]);
    let target = Tensor2D::<4, 1>::from_array([[1.0], [3.0], [5.0], [7.0]]);
    let mut optimizer = Adam::new(0.05);

    let initial = mse_loss(&layer.forward(&input), &target).to_vec()[0];

    for _ in 0..1000 {
        layer.zero_grad();
        let loss = mse_loss(&layer.forward(&input), &target);
        loss.backward();
        optimizer.step(layer.parameters_mut());
    }

    let final_loss = mse_loss(&layer.forward(&input), &target).to_vec()[0];
    assert!(
        final_loss < initial,
        "initial={initial}, final={final_loss}"
    );
    assert!(final_loss < 0.01, "final loss too high: {final_loss}");
}

#[test]
fn seeded_xor_classification_loss_decreases() {
    let mut rng = SmallRng::seed_from_u64(11);
    let mut model = XorMlp::new(&mut rng);
    let input = Tensor2D::<4, 2>::from_array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let target = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [1.0], [0.0]]);
    let mut optimizer = Adam::new(0.05);

    let initial = binary_cross_entropy(&model.forward(&input), &target).to_vec()[0];

    for _ in 0..1500 {
        model.zero_grad();
        let prediction = model.forward(&input);
        let loss = binary_cross_entropy(&prediction, &target);
        loss.backward();
        optimizer.step(model.parameters_mut());
    }

    let final_loss = binary_cross_entropy(&model.forward(&input), &target).to_vec()[0];
    assert!(
        final_loss < initial,
        "initial={initial}, final={final_loss}"
    );
    assert!(final_loss < 0.1, "final loss too high: {final_loss}");
}
