//! Model-level utilities (exploration §4.4): visitor-based helpers that
//! operate on any `&dyn Module`.
//!
//! **Contract file** (T01²/T40). T01 provides the read-only walks that
//! cannot reasonably differ ([`num_params`], [`state_dict`]); **T40** fills
//! the mutating conversions ([`load_state_dict`], [`to_device`],
//! [`to_dtype`]) once the tensor movement ops exist. Signatures frozen.
//!
//! `state_dict` uses an in-memory [`BTreeMap<String, Tensor>`] — the
//! representation behind the sanctioned replication path (construct +
//! `load_state_dict(&other.state_dict())`, no disk). Persistence to
//! safetensors is a separate concern owned by [`persist`](crate::persist).
//!
//! **Params vs buffers**: `num_params` counts trainable [`Param`](crate::nn::Param)
//! elements only; `state_dict`/`load_state_dict`/`to_device`/`to_dtype`
//! operate on both parameters and non-trainable `Tensor` buffers (BatchNorm
//! running stats), so a moved or checkpointed model stays complete.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::nn::Module;
use crate::nn::visit::{Leaf, visit_all};
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// Total number of scalar elements across all **trainable parameters** of
/// `module` (buffers are excluded).
pub fn num_params(module: &dyn Module) -> usize {
    let mut total = 0usize;
    visit_all(module, &mut |_path, leaf| {
        if let Leaf::Param(p) = leaf {
            total += p.value().num_elements();
        }
    });
    total
}

/// Collect the module's parameters **and buffers** into a dotted-path → value
/// map (ordered for stable, diffable output). Values are `Arc`-cheap clones.
/// Buffers are included so a checkpoint reconstructs a model (running stats
/// survive), matching PyTorch `state_dict` semantics.
pub fn state_dict(module: &dyn Module) -> BTreeMap<String, Tensor> {
    let mut out = BTreeMap::new();
    visit_all(module, &mut |path, leaf| {
        let value = match leaf {
            Leaf::Param(p) => p.value().clone(),
            Leaf::Buffer(t) => t.clone(),
        };
        out.insert(path.to_string(), value);
    });
    out
}

/// Load values from `state` into `module` by matching dotted paths — both
/// parameters and buffers (the in-memory half of the replication/checkpoint
/// path). T40 fills the body: every non-extra leaf must be present and
/// shape/dtype compatible, else a structured [`Error`](crate::Error).
pub fn load_state_dict(module: &mut dyn Module, state: &BTreeMap<String, Tensor>) -> Result<()> {
    let _ = (module, state);
    todo!("T40: load_state_dict (path match + Param::set/buffer set + loud on mismatch)")
}

/// Move every parameter and buffer of `module` to `device` (constructors
/// initialize on their given device; this converts after). T40 fills the body
/// via the crate-private `visit_all_mut` walk + `Param::set`.
pub fn to_device(module: &mut dyn Module, device: &Device) -> Result<()> {
    let _ = (module, device);
    todo!("T40: nn::to_device")
}

/// Cast every parameter and buffer of `module` to `dtype` (exploration §4.4:
/// constructors initialize F32, convert after). T40 fills the body via the
/// crate-private `visit_all_mut` walk + `Param::set`.
pub fn to_dtype(module: &mut dyn Module, dtype: DType) -> Result<()> {
    let _ = (module, dtype);
    todo!("T40: nn::to_dtype")
}
