use rstorch::{MultiHeadAttention, SmallRng};

fn main() {
    let mut rng = SmallRng::seed_from_u64(1);

    let _ = MultiHeadAttention::<4, 8, 3, 2>::xavier_uniform(&mut rng);
}
