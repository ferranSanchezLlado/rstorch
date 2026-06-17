#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
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
    println!("tiny regression: initial_loss={initial:.6}, final_loss={final_loss:.6}");
    assert!(final_loss < initial);
    assert!(final_loss < 0.01);
}
