//! Model-level utilities: visitor-based helpers that
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
//!
//! The user-facing prose for these functions — the replication recipe and the
//! all-or-nothing guarantee — lives in the [`nn`](crate::nn) module docs,
//! which is where rustdoc renders it (this module is private; its items are
//! re-exported flat).
//!
//! Every mutating helper validates the whole walk before it swaps anything
//! (the in-memory sibling of [`persist::stage`](crate::persist::stage), 16.14
//! WS5), so a rejected load or a failed conversion leaves the model exactly as
//! it was.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::visit::{Leaf, LeafMut, visit_all, visit_all_mut};
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
/// path; see the [`nn`](crate::nn) module docs for the replication recipe).
///
/// The match is **exact and loud**. `state` must carry one entry for every
/// path the module's walk emits, and no others, each with the same dtype,
/// dims, and device as the value it replaces. A silent partial load is the
/// failure mode this rejects: every check names the offending path.
///
/// Incoming values are [`detach`](crate::Tensor::detach)ed, so a state dict
/// captured from a live graph can never smuggle autograd history into a
/// parameter.
///
/// # Errors
///
/// [`Error::InvalidArg`] (`op: "load_state_dict"`), naming the path, when
///
/// - `state` is missing a path the module expects,
/// - `state` carries a path the module does not have,
/// - a value's dims, dtype, or device differ from the current value's — for a
///   device or dtype difference, convert the source explicitly first
///   ([`to_device`], [`to_dtype`], or [`Tensor::to_device`]), because a
///   parameter silently changing device or precision is exactly the kind of
///   surprise this API refuses, or
/// - the module emits one path twice, or its two walks disagree (a path
///   emitted by `visit_mut` that `visit` never emitted) — both are `Module`
///   implementation bugs that would make a state dict lossy.
///
/// Nothing is swapped unless every check passes.
pub fn load_state_dict(module: &mut dyn Module, state: &BTreeMap<String, Tensor>) -> Result<()> {
    const OP: &str = "load_state_dict";
    // The current values double as the schema every incoming value must match.
    let current = mapped(OP, module, |t| Ok(t.clone()))?;

    // Unexpected paths first — the loudest signal that the state dict came
    // from a different model (mirrors `persist::stage`).
    for path in state.keys() {
        if !current.contains_key(path) {
            return Err(invalid(
                OP,
                format!(
                    "unexpected key `{path}` in state dict: the target module has \
                     no parameter or buffer at that path"
                ),
            ));
        }
    }

    for (path, target) in &current {
        let Some(value) = state.get(path) else {
            return Err(invalid(
                OP,
                format!("missing key `{path}` in state dict (expected by the target module)"),
            ));
        };
        if value.dims() != target.dims() {
            return Err(invalid(
                OP,
                format!(
                    "`{path}` shape mismatch: state dict has {}, target expects {}",
                    value.shape(),
                    target.shape()
                ),
            ));
        }
        if value.dtype() != target.dtype() {
            return Err(invalid(
                OP,
                format!(
                    "`{path}` dtype mismatch: state dict has {}, target expects {} \
                     (cast explicitly with to_dtype)",
                    value.dtype(),
                    target.dtype()
                ),
            ));
        }
        if value.device() != target.device() {
            return Err(invalid(
                OP,
                format!(
                    "`{path}` device mismatch: state dict has {}, target expects {} \
                     (move it explicitly with to_device)",
                    value.device(),
                    target.device()
                ),
            ));
        }
    }

    commit(OP, module, state)
}

/// Move every parameter and buffer of `module` to `device` (constructors
/// initialize on their given device; this converts after).
///
/// All-or-nothing: every value is converted first, and only a fully converted
/// module is committed, so a failure never leaves parameters split across two
/// devices.
///
/// # Errors
///
/// Whatever [`Tensor::to_device`] reports for a leaf (for a device with no
/// backend, [`Error::Unsupported`]), or [`Error::InvalidArg`]
/// (`op: "to_device"`) if the module's two walks disagree.
pub fn to_device(module: &mut dyn Module, device: &Device) -> Result<()> {
    const OP: &str = "to_device";
    let values = mapped(OP, module, |t| t.to_device(device))?;
    commit(OP, module, &values)
}

/// Cast every **floating-point** parameter and buffer of `module` to `dtype`
/// (constructors initialize F32, convert after).
///
/// Integer and boolean leaves are left untouched, as in PyTorch: an `I64`
/// index buffer or a `Bool` mask is structure, not precision, and casting it
/// to a float would corrupt the model rather than convert it.
///
/// All-or-nothing: every value is converted first, and only a fully converted
/// module is committed, so a failure never leaves a model in mixed precision.
///
/// # Errors
///
/// - [`Error::InvalidArg`] (`op: "to_dtype"`) if `dtype` is not a float dtype
///   (a whole model cast to `I64`/`Bool` is a bug, not a request), or if the
///   module's two walks disagree.
/// - Whatever [`Tensor::to_dtype`] reports for a leaf — notably
///   [`Error::Unsupported`] for a cast lane the backend does not implement.
pub fn to_dtype(module: &mut dyn Module, dtype: DType) -> Result<()> {
    const OP: &str = "to_dtype";
    if !dtype.is_float() {
        return Err(invalid(
            OP,
            format!(
                "cannot cast a module to {dtype}: nn::to_dtype converts precision, \
                 so the target must be a float dtype"
            ),
        ));
    }
    let values = mapped(OP, module, |t| {
        if t.dtype().is_float() {
            t.to_dtype(dtype)
        } else {
            Ok(t.clone())
        }
    })?;
    commit(OP, module, &values)
}

// ---- internals -----------------------------------------------------------

/// A uniform, path-naming rejection.
fn invalid(op: &'static str, msg: String) -> Error {
    Error::InvalidArg { op, msg }
}

/// Apply `f` to every leaf value, collecting the results by dotted path — the
/// fallible half of every mutating helper here, run to completion **before**
/// anything is swapped.
///
/// Two failures end the walk:
///
/// - `f` reports one. The error is returned unchanged: the underlying op's
///   error is already loud and names the op that could not convert.
/// - The module emits the same path twice, so two distinct leaves collide.
///   A tied parameter is owned once and visited once, so a
///   collision is a `Module` implementation bug; rejecting it keeps one leaf
///   from silently overwriting the other's value on the way back in.
fn mapped(
    op: &'static str,
    module: &dyn Module,
    mut f: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<BTreeMap<String, Tensor>> {
    let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut failure: Option<Error> = None;
    visit_all(module, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let current = match leaf {
            Leaf::Param(p) => p.value(),
            Leaf::Buffer(t) => t,
        };
        match f(current) {
            Ok(value) => {
                if out.insert(path.to_string(), value).is_some() {
                    failure = Some(invalid(
                        op,
                        format!(
                            "module emits the path `{path}` twice: two distinct leaves \
                             collide, so a state dict could not round-trip them"
                        ),
                    ));
                }
            }
            Err(e) => failure = Some(e),
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// The swap half: write `values[path]` into every leaf.
///
/// Callers must have validated (or produced) `values` against the module's
/// read-only walk first, so the swaps themselves cannot mismatch. What is left
/// to check is that the mutable walk visits exactly the same leaves as the
/// read-only one — a divergence would skip a leaf or write one value into two.
/// That check runs as a dry pass **before** the first swap, so even a `Module`
/// with disagreeing walks is rejected with nothing modified.
///
/// Values are detached, so no autograd history enters a parameter or buffer.
fn commit(
    op: &'static str,
    module: &mut dyn Module,
    values: &BTreeMap<String, Tensor>,
) -> Result<()> {
    let mut failure: Option<Error> = None;
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    visit_all_mut(module, &mut |path, _leaf| {
        if failure.is_some() {
            return;
        }
        if !values.contains_key(path) {
            failure = Some(invalid(
                op,
                format!(
                    "`{path}` is emitted by visit_mut but not by visit: the module's \
                     two walks disagree"
                ),
            ));
        } else if !seen.insert(path.to_string()) {
            failure = Some(invalid(
                op,
                format!(
                    "visit_mut emits the path `{path}` twice: two distinct leaves would \
                     receive the same value"
                ),
            ));
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    if seen.len() != values.len() {
        let missed = values.keys().find(|p| !seen.contains(*p));
        return Err(invalid(
            op,
            match missed {
                Some(path) => format!(
                    "`{path}` is emitted by visit but not by visit_mut: the module's \
                     two walks disagree"
                ),
                // Unreachable: the sets have equal membership yet unequal size.
                None => "the module's two walks disagree".to_string(),
            },
        ));
    }

    let mut failure: Option<Error> = None;
    visit_all_mut(module, &mut |path, leaf| {
        // The dry pass above proved every visited path is present, and the
        // caller checked every shape against these very leaves — so neither
        // arm below can fail. Report rather than ignore if one ever does.
        let Some(value) = values.get(path) else {
            return;
        };
        let value = value.detach();
        let result = match leaf {
            LeafMut::Param(p) => p.set(value),
            LeafMut::Buffer(t) => {
                *t = value;
                Ok(())
            }
        };
        if let Err(e) = result
            && failure.is_none()
        {
            failure = Some(e);
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Param;
    use crate::nn::visit::{Visitor, VisitorMut};

    fn dev() -> Device {
        Device::Cpu
    }

    fn t(data: &[f32], dims: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), dims.to_vec(), &dev()).unwrap()
    }

    /// A leaf module with one parameter and one buffer.
    #[derive(rstorch::Module)]
    struct Cell {
        weight: Param,
        running: Tensor,
    }

    impl Cell {
        fn new(fill: f32) -> Cell {
            Cell {
                weight: Param::new(t(&[fill, fill, fill, fill], &[2, 2])),
                running: t(&[fill, fill], &[2]),
            }
        }
    }

    /// A nest with every derive-supported child shape.
    #[derive(rstorch::Module)]
    struct Net {
        cell: Cell,
        blocks: Vec<Cell>,
        head: Option<Cell>,
        bias: Option<Param>,
        #[module(skip)]
        #[allow(dead_code)]
        label: &'static str,
    }

    impl Net {
        fn new(fill: f32) -> Net {
            Net {
                cell: Cell::new(fill),
                blocks: vec![Cell::new(fill), Cell::new(fill)],
                head: Some(Cell::new(fill)),
                bias: Some(Param::new(t(&[fill], &[1]))),
                label: "net",
            }
        }
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    fn err_msg(e: Error) -> String {
        e.to_string()
    }

    #[test]
    fn num_params_counts_params_not_buffers() {
        // 4 cells × 4 weight elements + 1 bias = 17; the 4 × 2 buffer
        // elements do not count.
        assert_eq!(num_params(&Net::new(1.0)), 17);
    }

    #[test]
    fn state_dict_round_trips_values() {
        let src = Net::new(2.0);
        let mut dst = Net::new(0.0);
        assert_eq!(values(dst.cell.weight.value()), vec![0.0; 4]);

        load_state_dict(&mut dst, &state_dict(&src)).unwrap();

        let a = state_dict(&src);
        let b = state_dict(&dst);
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (path, want) in &a {
            assert_eq!(values(want), values(&b[path]), "at {path}");
        }
    }

    #[test]
    fn load_rejects_missing_key() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        assert!(state.remove("cell.running").is_some());
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("missing key `cell.running`"), "{msg}");
        // Nothing was swapped.
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_unexpected_key() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        state.insert("cell.extra".to_string(), t(&[9.0], &[1]));
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("unexpected key `cell.extra`"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_shape_mismatch() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        state.insert("cell.weight".to_string(), t(&[1.0, 2.0], &[2]));
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("`cell.weight` shape mismatch"), "{msg}");
        assert!(msg.contains("[2]") && msg.contains("[2, 2]"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_dtype_mismatch() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        let i64s = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &dev()).unwrap();
        state.insert("cell.weight".to_string(), i64s);
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("`cell.weight` dtype mismatch"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_detaches_incoming_values() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        // A value that carries a live graph must not smuggle it into a Param.
        let traced = state["cell.weight"].traced().unwrap();
        state.insert("cell.weight".to_string(), traced);
        load_state_dict(&mut m, &state).unwrap();
        assert!(m.cell.weight.value().backward().is_err());
    }

    #[test]
    fn load_is_all_or_nothing_across_the_walk() {
        // `bias` sorts before `cell.weight`, so a valid `bias` would be
        // swapped first by a naive one-pass implementation.
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0));
        state.insert("cell.weight".to_string(), t(&[1.0], &[1]));
        assert!(load_state_dict(&mut m, &state).is_err());
        assert_eq!(values(m.bias.as_ref().unwrap().value()), vec![0.0]);
    }

    #[test]
    fn duplicate_paths_are_rejected() {
        struct Collide {
            a: Param,
            b: Param,
        }
        impl Module for Collide {
            fn visit(&self, v: &mut Visitor) {
                v.param("w", &self.a);
                v.param("w", &self.b);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("w", &mut self.a);
                v.param("w", &mut self.b);
            }
        }
        let mut m = Collide {
            a: Param::new(t(&[1.0], &[1])),
            b: Param::new(t(&[2.0], &[1])),
        };
        let state = state_dict(&m);
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("emits the path `w` twice"), "{msg}");
    }

    #[test]
    fn disagreeing_walks_are_rejected_before_anything_is_swapped() {
        // A hand-written `Module` whose mutable walk forgets a leaf: the load
        // would silently leave `b` untrained. (`#[derive(Module)]` cannot
        // produce this; a hand-written impl can.)
        struct Lopsided {
            a: Param,
            b: Param,
        }
        impl Module for Lopsided {
            fn visit(&self, v: &mut Visitor) {
                v.param("a", &self.a);
                v.param("b", &self.b);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("a", &mut self.a);
            }
        }
        let mut m = Lopsided {
            a: Param::new(t(&[0.0], &[1])),
            b: Param::new(t(&[0.0], &[1])),
        };
        let mut state = state_dict(&m);
        state.insert("a".to_string(), t(&[1.0], &[1]));
        state.insert("b".to_string(), t(&[2.0], &[1]));

        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(
            msg.contains("`b` is emitted by visit but not by visit_mut"),
            "{msg}"
        );
        // `a` was reachable by both walks, and still was not swapped.
        assert_eq!(values(m.a.value()), vec![0.0]);
    }

    #[test]
    fn to_device_cpu_is_a_faithful_walk() {
        let mut m = Net::new(3.0);
        to_device(&mut m, &Device::Cpu).unwrap();
        for (path, v) in state_dict(&m) {
            assert_eq!(v.device(), Device::Cpu, "at {path}");
            assert!(values(&v).iter().all(|&x| x == 3.0), "at {path}");
        }
    }

    #[test]
    fn to_dtype_leaves_non_float_leaves_alone() {
        #[derive(rstorch::Module)]
        struct Mixed {
            weight: Param,
            indices: Tensor,
        }
        let mut m = Mixed {
            weight: Param::new(t(&[1.0, 2.0], &[2])),
            indices: Tensor::from_vec(vec![0i64, 1], [2], &dev()).unwrap(),
        };
        // F32 -> F32 is the identity lane; the I64 buffer must survive it.
        to_dtype(&mut m, DType::F32).unwrap();
        assert_eq!(m.weight.value().dtype(), DType::F32);
        assert_eq!(m.indices.dtype(), DType::I64);
        assert_eq!(m.indices.to_vec::<i64>().unwrap(), vec![0, 1]);
    }

    #[test]
    fn to_dtype_rejects_non_float_targets() {
        let mut m = Net::new(1.0);
        let msg = err_msg(to_dtype(&mut m, DType::I64).unwrap_err());
        assert!(msg.contains("must be a float dtype"), "{msg}");
    }

    #[test]
    fn to_dtype_failure_leaves_the_model_untouched() {
        // F32 -> F64 is a deferred cast lane, so the conversion pass fails
        // before anything is committed.
        let mut m = Net::new(1.0);
        assert!(matches!(
            to_dtype(&mut m, DType::F64),
            Err(Error::Unsupported { .. })
        ));
        assert_eq!(m.cell.weight.value().dtype(), DType::F32);
        assert_eq!(values(m.cell.weight.value()), vec![1.0; 4]);
    }

    #[test]
    fn replication_recipe_shares_no_future_updates() {
        let src = Net::new(5.0);
        let mut replica = Net::new(0.0);
        load_state_dict(&mut replica, &state_dict(&src)).unwrap();
        // A later "optimizer step" on the replica writes a new value and does
        // not disturb the original.
        replica.cell.weight.set(t(&[7.0; 4], &[2, 2])).unwrap();
        assert_eq!(values(src.cell.weight.value()), vec![5.0; 4]);
        assert_eq!(values(replica.cell.weight.value()), vec![7.0; 4]);
    }
}
