use rstorch::{Device, Result};
use rstorch::typed::{DeviceCtx, Placement, Tensor1};

struct Left;
impl Placement for Left {}

struct Right;
impl Placement for Right {}

fn main() -> Result<()> {
    let left_ctx = DeviceCtx::<Left>::bind(Device::Cpu)?;
    let right_ctx = DeviceCtx::<Right>::bind(Device::Cpu)?;
    let left = Tensor1::<2, f32, Left>::from_vec(vec![1.0; 2], [2], &left_ctx)?;
    let right = Tensor1::<2, f32, Right>::from_vec(vec![2.0; 2], [2], &right_ctx)?;
    let _ = left.add(&right);
    Ok(())
}
