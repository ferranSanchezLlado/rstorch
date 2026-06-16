#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn assert_all_zero(values: &[f32]) {
    assert!(values.iter().all(|value| *value == 0.0));
}

#[test]
fn same_seed_produces_same_prng_sequence() {
    let mut first = SmallRng::seed_from_u64(42);
    let mut second = SmallRng::seed_from_u64(42);

    let first_values: Vec<_> = (0..8).map(|_| first.next_u64()).collect();
    let second_values: Vec<_> = (0..8).map(|_| second.next_u64()).collect();

    assert_eq!(first_values, second_values);
}

#[test]
fn different_seeds_produce_different_prng_sequences() {
    let mut first = SmallRng::seed_from_u64(42);
    let mut second = SmallRng::seed_from_u64(43);

    let first_values: Vec<_> = (0..8).map(|_| first.next_u64()).collect();
    let second_values: Vec<_> = (0..8).map(|_| second.next_u64()).collect();

    assert_ne!(first_values, second_values);
}

#[test]
fn uniform_samples_stay_inside_requested_range() {
    let mut rng = SmallRng::seed_from_u64(7);

    for _ in 0..1_000 {
        let value = rng.uniform_f32(-2.5, 3.5);
        assert!(
            (-2.5..=3.5).contains(&value),
            "sample out of range: {value}"
        );
    }

    for _ in 0..1_000 {
        let value = rng.uniform_f64(-4.0, -1.0);
        assert!(
            (-4.0..=-1.0).contains(&value),
            "sample out of range: {value}"
        );
    }
}

#[test]
fn kaiming_initialized_linear_weights_are_not_all_zero() {
    let mut rng = SmallRng::seed_from_u64(11);
    let layer = Linear::<3, 4>::kaiming_uniform(&mut rng);

    assert!(
        layer
            .weight()
            .tensor()
            .to_vec()
            .iter()
            .any(|value| *value != 0.0)
    );
}

#[test]
fn initialized_linear_biases_are_zero() {
    let mut rng = SmallRng::seed_from_u64(11);
    let layer = Linear::<3, 4>::xavier_uniform(&mut rng);

    assert_all_zero(&layer.bias().tensor().to_vec());
}

#[test]
fn seeded_linear_initialization_is_deterministic() {
    let mut first_rng = SmallRng::seed_from_u64(123);
    let mut second_rng = SmallRng::seed_from_u64(123);

    let first = Linear::<2, 3>::kaiming_uniform(&mut first_rng);
    let second = Linear::<2, 3>::kaiming_uniform(&mut second_rng);

    assert_eq!(
        first.weight().tensor().to_vec(),
        second.weight().tensor().to_vec()
    );
    assert_eq!(
        first.bias().tensor().to_vec(),
        second.bias().tensor().to_vec()
    );
}

#[test]
fn linear_zeros_keeps_exact_zero_behavior() {
    let layer = Linear::<2, 3>::zeros();

    assert_all_zero(&layer.weight().tensor().to_vec());
    assert_all_zero(&layer.bias().tensor().to_vec());
}

#[test]
fn tiny_mlp_has_non_identical_initial_hidden_weights() {
    let mut rng = SmallRng::seed_from_u64(99);
    let hidden = Linear::<2, 4>::kaiming_uniform(&mut rng);
    let _output = Linear::<4, 1>::kaiming_uniform(&mut rng);
    let weights = hidden.weight().tensor().to_vec();

    let first_hidden_unit = [weights[0], weights[4]];
    let has_distinct_hidden_unit =
        (1..4).any(|unit| [weights[unit], weights[4 + unit]] != first_hidden_unit);

    assert!(has_distinct_hidden_unit);
}

#[test]
fn xavier_uniform_respects_expected_bound() {
    let mut rng = SmallRng::seed_from_u64(5);
    let layer = Linear::<3, 5, f64>::xavier_uniform(&mut rng);
    let bound = (6.0_f64 / 8.0).sqrt();

    for value in layer.weight().tensor().to_vec() {
        assert!(
            (-bound..=bound).contains(&value),
            "weight out of range: {value}"
        );
    }
}
