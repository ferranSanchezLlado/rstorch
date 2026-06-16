pub const FINITE_DIFFERENCE_EPSILON: f64 = 1e-6;
pub const FINITE_DIFFERENCE_TOLERANCE: f64 = 1e-6;

pub fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < FINITE_DIFFERENCE_TOLERANCE,
            "index {index}: {actual} != {expected}"
        );
    }
}

pub fn finite_difference(values: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut gradient = Vec::with_capacity(values.len());

    for index in 0..values.len() {
        let mut plus = values.to_vec();
        plus[index] += FINITE_DIFFERENCE_EPSILON;
        let mut minus = values.to_vec();
        minus[index] -= FINITE_DIFFERENCE_EPSILON;
        gradient.push((f(&plus) - f(&minus)) / (2.0 * FINITE_DIFFERENCE_EPSILON));
    }

    gradient
}
