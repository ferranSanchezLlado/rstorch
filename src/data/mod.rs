//! Data pipeline: the batch-level `Dataset` trait, `DataLoader`, and the
//! provided dataset types (exploration §4.6).
//!
//! A dataset yields **batches**, not items ([`Dataset`]), and [`DataLoader`]
//! is the only thing that decides which item positions go into which batch:
//!
//! ```
//! use rstorch::data::{DataLoader, TensorDataset};
//! # use rstorch::{Device, Result, Tensor};
//! # fn main() -> Result<()> {
//! # let dev = Device::Cpu;
//! let x = Tensor::zeros([10, 4], rstorch::DType::F32, &dev)?;
//! let y = Tensor::from_vec(vec![0i64; 10], [10], &dev)?;
//! let loader = DataLoader::new(TensorDataset::new(x, y)?, 4).shuffle(0);
//!
//! assert_eq!(loader.num_batches(), 3); // 4 + 4 + 2: the tail is kept
//! for batch in loader.batches() {
//!     let (inputs, targets) = batch?;
//!     assert_eq!(inputs.dims()[0], targets.dims()[0]);
//! }
//! # Ok(())
//! # }
//! ```

pub mod hub;

mod dataset;
mod loader;

pub use dataset::{Dataset, TensorDataset, VecDataset};
pub use loader::{Batches, DataLoader};
