//! Parameter/buffer visitors with dotted paths.
//!
//! [`Module::visit`](crate::nn::Module::visit) and
//! [`visit_mut`](crate::nn::Module::visit_mut) walk a module tree, calling
//! into a [`Visitor`]/[`VisitorMut`] which threads a dotted path prefix
//! (`fc1.weight`, `blocks.3.attention.q_proj.weight`) and forwards each leaf to a
//! single sink. `#[derive(Module)]` generates the walk: one method
//! call per field — [`param`](Visitor::param) for a [`Param`],
//! [`buffer`](Visitor::buffer) for a whitelisted `Tensor` buffer (e.g.
//! `BatchNorm` `running_mean`), and [`module`](Visitor::module) for a child
//! module. These path semantics are the on-disk `state_dict` key format, so
//! they are frozen here (a §9 risk item).
//!
//! **Params vs buffers**: a `Param` is trainable
//! and optimizer-visited; a `Tensor` buffer is non-trainable persistent state
//! (running statistics). Both are moved by `nn::to_device`/`to_dtype` and both
//! land in `state_dict` (so a checkpoint reconstructs a model, as `PyTorch`
//! does); only `Param`s count toward `num_params` and receive gradients.

use crate::nn::{Module, Param};
use crate::tensor::Tensor;

/// A leaf's dotted path: `name` at the top level, `prefix.name` below it.
/// Shared with the typed visitors, whose path semantics are the same.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Push child segment `name` onto a visitor's prefix, returning the length to
/// [`truncate`](String::truncate) back to once that child's walk has finished.
pub(crate) fn push_segment(path: &mut String, name: &str) -> usize {
    let saved = path.len();
    if !path.is_empty() {
        path.push('.');
    }
    path.push_str(name);
    saved
}

/// A read-only leaf handed to a visitor sink: a trainable [`Param`] or a
/// non-trainable `Tensor` buffer, distinguished so sinks can treat them
/// differently (e.g. `num_params` counts only params; `state_dict` keeps
/// both). Crate-private — the public visitor API is the `param`/`buffer`
/// methods, not this enum.
pub(crate) enum Leaf<'a> {
    Param(&'a Param),
    Buffer(&'a Tensor),
}

/// The mutable counterpart of [`Leaf`], destructured by the mutable-walk
/// consumers (`nn::load_state_dict`/`to_device`/`to_dtype`, and the optimizer
/// step).
pub(crate) enum LeafMut<'a> {
    Param(&'a mut Param),
    Buffer(&'a mut Tensor),
}

/// The read-only leaf visitor threaded through
/// [`Module::visit`](crate::nn::Module::visit).
///
/// `#[derive(Module)]` emits, per field: [`param`](Visitor::param) for a
/// [`Param`], [`buffer`](Visitor::buffer) for a whitelisted `Tensor` buffer,
/// [`module`](Visitor::module) for a child module, an indexed call per `Vec<M>`
/// element (child name `"blocks.3"`, producing `blocks.3.weight`), and a
/// field-named call when an `Option<M>` is `Some`.
pub struct Visitor<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, Leaf<'_>),
}

impl<'a> Visitor<'a> {
    pub(crate) fn new(sink: &'a mut dyn FnMut(&str, Leaf<'_>)) -> Visitor<'a> {
        Visitor {
            path: String::new(),
            sink,
        }
    }

    /// Emit a trainable parameter leaf named `name` at the current prefix.
    pub fn param(&mut self, name: &str, p: &Param) {
        let full = join(&self.path, name);
        (self.sink)(&full, Leaf::Param(p));
    }

    /// Emit a non-trainable `Tensor` buffer leaf named `name` at the current
    /// prefix (persistent state such as `BatchNorm` running statistics).
    pub fn buffer(&mut self, name: &str, t: &Tensor) {
        let full = join(&self.path, name);
        (self.sink)(&full, Leaf::Buffer(t));
    }

    /// Descend into child module `child` under segment `name`, prefixing all
    /// of its leaf paths with `name.`.
    pub fn module(&mut self, name: &str, child: &dyn Module) {
        let saved = push_segment(&mut self.path, name);
        child.visit(self);
        self.path.truncate(saved);
    }
}

/// The mutable counterpart of [`Visitor`], threaded through
/// [`Module::visit_mut`](crate::nn::Module::visit_mut). Same path semantics;
/// the sink receives a `LeafMut` (optimizer step, `load_state_dict`,
/// `to_device`/`to_dtype`).
pub struct VisitorMut<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, LeafMut<'_>),
}

impl<'a> VisitorMut<'a> {
    pub(crate) fn new(sink: &'a mut dyn FnMut(&str, LeafMut<'_>)) -> VisitorMut<'a> {
        VisitorMut {
            path: String::new(),
            sink,
        }
    }

    /// Emit a mutable parameter leaf named `name` at the current prefix.
    pub fn param(&mut self, name: &str, p: &mut Param) {
        let full = join(&self.path, name);
        (self.sink)(&full, LeafMut::Param(p));
    }

    /// Emit a mutable `Tensor` buffer leaf named `name` at the current prefix.
    pub fn buffer(&mut self, name: &str, t: &mut Tensor) {
        let full = join(&self.path, name);
        (self.sink)(&full, LeafMut::Buffer(t));
    }

    /// Descend into mutable child module `child` under segment `name`.
    pub fn module(&mut self, name: &str, child: &mut dyn Module) {
        let saved = push_segment(&mut self.path, name);
        child.visit_mut(self);
        self.path.truncate(saved);
    }
}

/// Run `sink` over every `(dotted_path, Leaf)` in `module` — parameters and
/// buffers (the shared engine behind `num_params` and `state_dict`).
pub(crate) fn visit_all(module: &dyn Module, sink: &mut dyn FnMut(&str, Leaf<'_>)) {
    let mut v = Visitor::new(sink);
    module.visit(&mut v);
}

/// Run `sink` over every `(dotted_path, LeafMut)` in `module` (optimizer
/// step, `load_state_dict`, device/dtype conversion).
pub(crate) fn visit_all_mut(module: &mut dyn Module, sink: &mut dyn FnMut(&str, LeafMut<'_>)) {
    let mut v = VisitorMut::new(sink);
    module.visit_mut(&mut v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::nn::Mode;

    fn t(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), [values.len()], &Device::Cpu).unwrap()
    }

    fn param(v: f32) -> Param {
        Param::new(t(&[v]))
    }

    /// The innermost leaf: one parameter, one optional parameter, one buffer.
    #[derive(rstorch::Module)]
    struct Proj {
        weight: Param,
        bias: Option<Param>,
        running: Tensor,
    }

    fn proj(bias: bool) -> Proj {
        Proj {
            weight: param(1.0),
            bias: bias.then(|| param(2.0)),
            running: t(&[0.0]),
        }
    }

    /// A middle layer, so paths are at least three segments deep.
    #[derive(rstorch::Module)]
    struct Attn {
        qkv: Proj,
    }

    /// Every child shape the derive supports, plus a skipped field.
    #[derive(rstorch::Module)]
    struct Net {
        fc1: Proj,
        blocks: Vec<Attn>,
        head: Option<Proj>,
        scale: Param,
        #[module(skip)]
        #[allow(dead_code)]
        name: &'static str,
        // A whitelisted primitive: configuration, never visited.
        #[allow(dead_code)]
        depth: usize,
    }

    fn net(head: bool, blocks: usize) -> Net {
        Net {
            fc1: proj(true),
            blocks: (0..blocks).map(|_| Attn { qkv: proj(true) }).collect(),
            head: head.then(|| proj(false)),
            scale: param(3.0),
            name: "net",
            depth: blocks,
        }
    }

    /// Every `(path, kind)` the read-only walk emits, in walk order.
    fn walk(module: &dyn Module) -> Vec<(String, &'static str)> {
        let mut out = Vec::new();
        visit_all(module, &mut |path, leaf| {
            let kind = match leaf {
                Leaf::Param(_) => "param",
                Leaf::Buffer(_) => "buffer",
            };
            out.push((path.to_string(), kind));
        });
        out
    }

    /// The same for the mutable walk.
    fn walk_mut(module: &mut dyn Module) -> Vec<(String, &'static str)> {
        let mut out = Vec::new();
        visit_all_mut(module, &mut |path, leaf| {
            let kind = match leaf {
                LeafMut::Param(_) => "param",
                LeafMut::Buffer(_) => "buffer",
            };
            out.push((path.to_string(), kind));
        });
        out
    }

    #[test]
    fn joins_prefixes_with_dots() {
        let paths: Vec<_> = walk(&net(true, 0)).into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            paths,
            vec![
                // A plain child module: one segment per level.
                "fc1.weight",
                "fc1.bias",
                "fc1.running",
                // `Option<M>`: present, so the field name is the segment.
                "head.weight",
                "head.running",
                // A parameter directly on the root has no prefix at all.
                "scale",
            ]
        );
    }

    #[test]
    fn a_multi_segment_child_name_indexes_a_collection() {
        // How a collection field is meant to read: the field's own segment,
        // then the element index, then the child's own paths
        // (`blocks.1.qkv.weight`). `Visitor::module` takes the joined name, so
        // one call per element is all it costs.
        struct Blocks {
            blocks: Vec<Attn>,
        }
        impl Module for Blocks {
            fn visit(&self, v: &mut Visitor) {
                for (i, m) in self.blocks.iter().enumerate() {
                    v.module(&format!("blocks.{i}"), m);
                }
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                for (i, m) in self.blocks.iter_mut().enumerate() {
                    v.module(&format!("blocks.{i}"), m);
                }
            }
        }

        let m = Blocks {
            blocks: (0..2).map(|_| Attn { qkv: proj(false) }).collect(),
        };
        let paths: Vec<_> = walk(&m).into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            paths,
            vec![
                "blocks.0.qkv.weight",
                "blocks.0.qkv.running",
                "blocks.1.qkv.weight",
                "blocks.1.qkv.running",
            ]
        );
    }

    #[test]
    fn derived_vec_fields_keep_their_field_segment() {
        // The frozen path format (this module's docs) is
        // `blocks.3.attn.qkv.weight`: the field segment, then the index. The
        // derive originally emitted only the index (`0.qkv.weight`), so two
        // `Vec` fields in one module collided on identical paths. That was found
        // it and wrote this case; the integrator fixed `rstorch-derive`
        // in the derive, so it now passes.
        let paths: Vec<_> = walk(&net(false, 2)).into_iter().map(|(p, _)| p).collect();
        assert!(
            paths.contains(&"blocks.1.qkv.weight".to_string()),
            "{paths:?}"
        );
    }

    #[test]
    fn absent_options_and_skipped_fields_emit_nothing() {
        let paths: Vec<_> = walk(&net(false, 0)).into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            paths,
            vec!["fc1.weight", "fc1.bias", "fc1.running", "scale"]
        );
        // `#[module(skip)]` and the whitelisted `usize` are invisible.
        assert!(
            !paths
                .iter()
                .any(|p| p.contains("name") || p.contains("depth"))
        );
        // `Option<Param>::None` likewise.
        let paths: Vec<_> = walk(&proj(false)).into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, vec!["weight", "running"]);
    }

    #[test]
    fn params_and_buffers_are_distinguished() {
        let kinds = walk(&proj(true));
        assert_eq!(
            kinds,
            vec![
                ("weight".to_string(), "param"),
                ("bias".to_string(), "param"),
                ("running".to_string(), "buffer"),
            ]
        );
    }

    #[test]
    fn the_two_walks_agree_exactly() {
        let mut m = net(true, 3);
        let read = walk(&m);
        let write = walk_mut(&mut m);
        assert_eq!(read, write);
    }

    #[test]
    fn descending_restores_the_prefix() {
        // Two siblings at the same level: the second must not inherit the
        // first's prefix.
        #[derive(rstorch::Module)]
        struct Two {
            a: Proj,
            b: Proj,
        }
        let paths: Vec<_> = walk(&Two {
            a: proj(false),
            b: proj(false),
        })
        .into_iter()
        .map(|(p, _)| p)
        .collect();
        assert_eq!(
            paths,
            vec!["a.weight", "a.running", "b.weight", "b.running"]
        );
    }

    #[test]
    fn a_tied_parameter_is_visited_exactly_once() {
        // Weight tying, the sanctioned way: the parent owns
        // the one `Param` and writes both uses inline. The visitor therefore
        // sees it once — so the optimizer cannot double-step it — while the
        // autograd engine still accumulates both contributions under its one
        // identity.
        #[derive(rstorch::Module)]
        struct Tied {
            embed: Param,
        }
        impl Tied {
            fn forward(&self, mode: Mode) -> Tensor {
                let e = self.embed.get(mode);
                // Two uses of the same leaf in one graph.
                e.mul(&e).unwrap().sum_all().unwrap()
            }
        }

        let m = Tied {
            embed: Param::new(t(&[1.0, 2.0, 3.0])),
        };
        let paths: Vec<_> = walk(&m).into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, vec!["embed"], "a tied param must appear once");
        assert_eq!(crate::nn::num_params(&m), 3);
        assert_eq!(crate::nn::state_dict(&m).len(), 1);

        // Both uses still contribute: d/de of sum(e ⊙ e) is 2e.
        let grads = m.forward(Mode::TRAIN).backward().unwrap();
        assert_eq!(
            grads.wrt(&m.embed).unwrap().to_vec::<f32>().unwrap(),
            vec![2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn the_mutable_walk_can_replace_leaves() {
        let mut m = proj(true);
        visit_all_mut(&mut m, &mut |_path, leaf| match leaf {
            LeafMut::Param(p) => p.set(t(&[9.0])).unwrap(),
            LeafMut::Buffer(b) => *b = t(&[8.0]),
        });
        assert_eq!(m.weight.value().to_vec::<f32>().unwrap(), vec![9.0]);
        assert_eq!(
            m.bias.as_ref().unwrap().value().to_vec::<f32>().unwrap(),
            vec![9.0]
        );
        assert_eq!(m.running.to_vec::<f32>().unwrap(), vec![8.0]);
    }
}
