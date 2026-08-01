// rstorch-ui: build

use rstorch::Rng;
use rstorch::typed::nn::{Mode, MultiHeadAttention};
use rstorch::typed::{DYN, DeviceCtx, Tensor3};

fn main() -> rstorch::Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(1))?;
    let input = Tensor3::<DYN, DYN, 5>::from_vec(vec![0.0f32; 5], [1, 1, 5], &ctx)?;
    let _ = attention.attend(&input, None, Mode::EVAL)?;
    Ok(())
}
