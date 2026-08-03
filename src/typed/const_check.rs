//! Compile-time predicates for typed shape relationships.

use super::DYN;

pub(crate) const fn dimensions_compatible(left: usize, right: usize) -> bool {
    left == DYN || right == DYN || left == right
}

pub(crate) const fn assert_matmul_contract(left: usize, right: usize) {
    assert!(
        dimensions_compatible(left, right),
        "typed matmul contracted dimensions are incompatible"
    );
}

pub(crate) const fn assert_conv2d_channels(input: usize, weight_input: usize) {
    assert!(
        dimensions_compatible(input, weight_input),
        "typed conv2d input channels are incompatible"
    );
}

pub(crate) const fn assert_loss_rows(logits: usize, targets: usize) {
    assert!(
        dimensions_compatible(logits, targets),
        "typed loss row dimensions are incompatible"
    );
}

/// Checks a non-selected `gather` dimension against the runtime's rule.
///
/// The runtime permits an index grid that is *smaller* than the source on every
/// axis other than the gathered one and rejects only a larger one
/// (`src/tensor/ops/index.rs`: `if a != axis && i > s`). Requiring equality
/// here would make a legal partial gather a hard compile error whenever the
/// markers happen to be static, so static markers would change *acceptance*
/// rather than merely checking it — and the same call would compile under
/// `DYN`. Mirror the runtime instead, and stay silent unless both dimensions
/// are known.
pub(crate) const fn assert_gather_dimension(source: usize, indices: usize) {
    assert!(
        source == DYN || indices == DYN || indices <= source,
        "typed gather index grid exceeds the source on a non-selected dimension"
    );
}

pub(crate) const fn assert_squeezable(dimension: usize) {
    assert!(
        dimension == DYN || dimension == 1,
        "typed squeeze axis must have dimension one"
    );
}

pub(crate) const fn assert_refinement(source: &[usize], target: &[usize]) {
    let mut axis = 0;
    while axis < source.len() {
        assert!(
            source[axis] == DYN || source[axis] == target[axis],
            "typed refinement cannot change or erase a static dimension"
        );
        axis += 1;
    }
}

pub(crate) const fn assert_reshape_numel(source: &[usize], target: &[usize]) {
    let source = match reshape_numel(source) {
        Ok(value) => value,
        Err(()) => panic!("typed reshape source element count overflow"),
    };
    let target = match reshape_numel(target) {
        Ok(value) => value,
        Err(()) => panic!("typed reshape target element count overflow"),
    };
    if let (Some(source), Some(target)) = (source, target) {
        assert!(
            source == target,
            "typed reshape element counts are incompatible"
        );
    }
}

pub(crate) const fn assert_broadcast(source: &[usize], target: &[usize]) {
    assert!(
        source.len() <= target.len(),
        "typed broadcast target rank is smaller than source rank"
    );

    let offset = target.len() - source.len();
    let mut axis = 0;
    while axis < source.len() {
        let from = source[axis];
        let to = target[offset + axis];
        assert!(
            from == DYN || to == DYN || from == 1 || from == to,
            "typed broadcast dimensions are incompatible"
        );
        axis += 1;
    }
}

// Reserved for the rank/axis-generated concat surface; kept item-local so the
// rest of typed internals remain warning checked.
#[allow(dead_code)]
pub(crate) const fn assert_concat_sum(left: usize, right: usize, output: usize) {
    if left == DYN || right == DYN || output == DYN {
        return;
    }
    let sum = match left.checked_add(right) {
        Some(sum) => sum,
        None => panic!("typed concat dimension addition overflow"),
    };
    assert!(
        sum == output,
        "typed concat output dimension is incompatible"
    );
}

// CT40 consumes this relationship when typed attention is introduced.
#[allow(dead_code)]
pub(crate) const fn assert_attention_heads(embed: usize, heads: usize, head_dim: usize) {
    assert!(
        embed == DYN || embed > 0,
        "typed attention embedding dimension must be non-zero"
    );
    assert!(
        heads == DYN || heads > 0,
        "typed attention head count must be non-zero"
    );
    assert!(
        head_dim == DYN || head_dim > 0,
        "typed attention head dimension must be non-zero"
    );
    if embed == DYN || heads == DYN || head_dim == DYN {
        return;
    }
    let product = match heads.checked_mul(head_dim) {
        Some(product) => product,
        None => panic!("typed attention head dimension multiplication overflow"),
    };
    assert!(
        product == embed,
        "typed attention head dimensions are incompatible"
    );
}

const fn reshape_numel(dimensions: &[usize]) -> Result<Option<usize>, ()> {
    let mut axis = 0;
    while axis < dimensions.len() {
        if dimensions[axis] == DYN {
            return Ok(None);
        }
        axis += 1;
    }

    // Detect zero first so a mathematically zero shape never spuriously
    // overflows because a larger axis appears before its zero axis.
    axis = 0;
    while axis < dimensions.len() {
        if dimensions[axis] == 0 {
            return Ok(Some(0));
        }
        axis += 1;
    }

    let mut product = 1usize;
    axis = 0;
    while axis < dimensions.len() {
        product = match product.checked_mul(dimensions[axis]) {
            Some(product) => product,
            None => return Err(()),
        };
        axis += 1;
    }
    Ok(Some(product))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_probe<const LEFT: usize, const RIGHT: usize>() {
        const { assert_matmul_contract(LEFT, RIGHT) };
    }

    #[test]
    fn wildcard_occurrences_are_independent() {
        assert!(dimensions_compatible(DYN, 3));
        assert!(dimensions_compatible(4, DYN));
        assert!(dimensions_compatible(DYN, DYN));
        assert!(!dimensions_compatible(3, 4));
        matmul_probe::<7, 7>();
        matmul_probe::<DYN, 7>();
        matmul_probe::<7, DYN>();
        matmul_probe::<DYN, DYN>();
    }

    #[test]
    fn zero_is_static_and_has_zero_numel() {
        assert_eq!(reshape_numel(&[usize::MAX - 1, 0]), Ok(Some(0)));
        assert_reshape_numel(&[0, 9], &[3, 0]);
        assert_broadcast(&[0, 1], &[0, 7]);
    }

    #[test]
    fn wildcard_products_defer_without_binding() {
        assert_eq!(reshape_numel(&[DYN, 0]), Ok(None));
        assert_reshape_numel(&[DYN, 2], &[3, DYN]);
        assert_broadcast(&[DYN, DYN], &[3, 4]);
    }

    #[test]
    #[should_panic(expected = "typed reshape source element count overflow")]
    fn reshape_uses_checked_multiplication() {
        assert_reshape_numel(&[usize::MAX - 1, 2], &[1]);
    }

    #[test]
    #[should_panic(expected = "typed concat dimension addition overflow")]
    fn concat_uses_checked_addition() {
        assert_concat_sum(usize::MAX - 1, 2, 1);
    }

    #[test]
    #[should_panic(expected = "typed broadcast dimensions are incompatible")]
    fn known_broadcast_mismatch_is_rejected() {
        assert_broadcast(&[2, 3], &[4, 3]);
    }

    #[test]
    #[should_panic(expected = "typed attention head count must be non-zero")]
    fn zero_static_can_still_be_invalid_for_a_specific_operation() {
        assert_attention_heads(DYN, 0, DYN);
    }
}
