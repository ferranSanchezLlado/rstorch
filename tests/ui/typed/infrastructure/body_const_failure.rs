// rstorch-ui: build

const DYN: usize = usize::MAX;

fn relational_check<const LEFT: usize, const RIGHT: usize>() {
    const {
        assert!(
            LEFT == DYN || RIGHT == DYN || LEFT == RIGHT,
            "typed relational dimensions are incompatible"
        )
    };
}

fn main() {
    relational_check::<7, 8>();
}
