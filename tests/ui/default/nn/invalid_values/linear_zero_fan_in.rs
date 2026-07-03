// rstorch-ui: build

use rstorch::{Linear, SmallRng};

fn main() {
    let mut rng = SmallRng::seed_from_u64(1);

    let _ = Linear::<0, 3>::kaiming_uniform(&mut rng);
}
