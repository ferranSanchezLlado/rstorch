// rstorch-ui: build

use rstorch::{Linear, SmallRng};

fn main() {
    let mut rng = SmallRng::seed_from_u64(1);

    let _ = Linear::<0, 0>::xavier_uniform(&mut rng);
}
