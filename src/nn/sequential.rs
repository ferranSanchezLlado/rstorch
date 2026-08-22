//! [`Sequential`] — a chain of layers behind one [`Forward`] (exploration
//! §4.1).
//!
//! The container is a plain `Vec` of boxed layers. Its only subtlety is a Rust
//! one: `dyn Forward<Input> + Module` is not a legal type (E0225 — at most one
//! non-auto trait per object type), so the box is typed by a crate-private
//! combining trait `SeqLayer<Input>` with a blanket impl. Nothing about that
//! leaks: the public surface is
//! `push(impl Forward<Input, Output = Input> + Module + Send + 'static)`, and
//! the crate's public trait count is unchanged.
//!
//! `Input` defaults to [`Tensor`], so `Sequential` and `Sequential::new()`
//! still mean the tensor-to-tensor chain.

use crate::error::Result;
use crate::nn::{Forward, Mode, Module, Visitor, VisitorMut};
use crate::tensor::Tensor;

/// The stored-layer trait: everything a [`Sequential`] element must be, in one
/// object-safe bundle. Crate-private with a blanket impl, so users never name
/// it — they satisfy it by implementing [`Forward`] and [`Module`].
///
/// The `Output = Input` bound is what makes a chain a chain: each element's
/// result must be the next element's argument.
///
/// `Send` is required so a `Sequential` (and any model holding one) can move
/// between threads, matching the rest of the crate's types.
pub(crate) trait SeqLayer<Input>: Forward<Input, Output = Input> + Module + Send {}

impl<Input, T: Forward<Input, Output = Input> + Module + Send> SeqLayer<Input> for T {}

/// A chain of layers applied in order, itself a [`Forward`] **and** a
/// [`Module`] — so it nests inside another `Sequential` or a
/// `#[derive(Module)]` struct.
///
/// Children are visited with their **index** as the path segment, exactly as
/// the derive treats a `Vec<M>`: a `Sequential` field named `layers` produces
/// `layers.0.weight`, `layers.1.weight`, … Those are the on-disk `state_dict`
/// keys, so inserting a layer in the middle renames every later path — the
/// same trade `PyTorch`'s `nn.Sequential` makes.
///
/// ```
/// # use rstorch::nn::{self, Forward, Mode, Module, ModuleExt, Param, Sequential};
/// # use rstorch::{DType, Device, Result, Tensor};
/// # #[derive(rstorch::Module)]
/// # struct Scale { factor: Param }
/// # impl Forward for Scale {
/// #     type Output = Tensor;
/// #     fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
/// #         x.mul(&self.factor.get(mode))
/// #     }
/// # }
/// # fn main() -> Result<()> {
/// # let dev = Device::Cpu;
/// # let two = || -> Result<Scale> {
/// #     Ok(Scale { factor: Param::new(Tensor::full([1], 2.0, DType::F32, &dev)?) })
/// # };
/// let mut net = Sequential::new().push(two()?).push(two()?);
/// let x = Tensor::full([2], 3.0, DType::F32, &dev)?;
/// assert_eq!(net.forward(&x, Mode::EVAL)?.to_vec::<f32>()?, vec![12.0, 12.0]);
/// assert_eq!(net.state_dict().keys().collect::<Vec<_>>(), ["0.factor", "1.factor"]);
/// # Ok(())
/// # }
/// ```
pub struct Sequential<Input = Tensor> {
    layers: Vec<Box<dyn SeqLayer<Input>>>,
}

/// Hand-written rather than derived: `#[derive(Default)]` would add an
/// `Input: Default` bound for every parameter regardless of use, and `Input`
/// appears only inside `Box<dyn SeqLayer<Input>>` here — a chain over a
/// non-`Default` input type is perfectly legal and must stay constructible.
impl<Input> Default for Sequential<Input> {
    fn default() -> Sequential<Input> {
        Sequential { layers: Vec::new() }
    }
}

impl<Input> Sequential<Input> {
    /// An empty chain. Empty is legal and forwards its input unchanged.
    pub fn new() -> Sequential<Input> {
        Sequential { layers: Vec::new() }
    }

    /// Append `layer` and return the chain, so construction reads as one
    /// expression:
    /// `Sequential::new().push(Linear::new(..)?).push(Relu)`.
    ///
    /// Building in a loop is the same call:
    /// `for _ in 0..n { net = net.push(block()?); }`.
    #[must_use]
    pub fn push(
        mut self,
        layer: impl Forward<Input, Output = Input> + Module + Send + 'static,
    ) -> Sequential<Input> {
        self.layers.push(Box::new(layer));
        self
    }

    /// The number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the chain is empty (and therefore the identity).
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<Input> std::fmt::Debug for Sequential<Input> {
    /// Layers are trait objects with no `Debug` bound, so the chain reports
    /// its length rather than its contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({} layers)", self.layers.len())
    }
}

impl<Input> Module for Sequential<Input> {
    fn visit(&self, visitor: &mut Visitor) {
        for (index, layer) in self.layers.iter().enumerate() {
            // `&dyn SeqLayer` -> `&dyn Module` by trait upcasting (stable
            // since Rust 1.86, within the 1.88 MSRV).
            visitor.module(&index.to_string(), layer.as_ref());
        }
    }

    fn visit_mut(&mut self, visitor: &mut VisitorMut) {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            visitor.module(&index.to_string(), layer.as_mut());
        }
    }
}

impl<Input: Clone> Forward<Input> for Sequential<Input> {
    type Output = Input;

    /// Apply every layer in order, threading `mode` through unchanged. An
    /// empty chain returns its input unchanged (an `Arc` bump for a
    /// [`Tensor`]) — which is the only path that clones, and the only reason
    /// `Input: Clone` is required at all.
    fn forward(&mut self, x: &Input, mode: Mode) -> Result<Input> {
        let Some((first, rest)) = self.layers.split_first_mut() else {
            return Ok(x.clone());
        };
        let mut current = first.forward(x, mode)?;
        for layer in rest {
            current = layer.forward(&current, mode)?;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::nn::{ModuleExt, Param};

    /// A one-parameter layer: `x * factor`, with a buffer along for the ride.
    #[derive(rstorch::Module)]
    struct Scale {
        factor: Param,
        calls: Tensor,
    }

    fn scale(factor: f32) -> Scale {
        let dev = Device::Cpu;
        Scale {
            factor: Param::new(Tensor::full([1], f64::from(factor), DType::F32, &dev).unwrap()),
            calls: Tensor::full([1], 0.0, DType::F32, &dev).unwrap(),
        }
    }

    impl Forward for Scale {
        type Output = Tensor;

        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
            self.calls = self.calls.add_scalar(1.0)?;
            x.mul(&self.factor.get(mode))
        }
    }

    /// A second, non-commuting layer: `x + shift`.
    #[derive(rstorch::Module)]
    struct Shift {
        shift: Param,
    }

    fn shift(by: f32) -> Shift {
        Shift {
            shift: Param::new(Tensor::full([1], f64::from(by), DType::F32, &Device::Cpu).unwrap()),
        }
    }

    impl Forward for Shift {
        type Output = Tensor;

        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
            x.add(&self.shift.get(mode))
        }
    }

    fn x(v: f32) -> Tensor {
        Tensor::full([2], f64::from(v), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn applies_layers_in_order() {
        // Composition is not commutative: (3 * 2) + 1 = 7, while the reversed
        // chain would give (3 + 1) * 2 = 8.
        let mut net = Sequential::new().push(scale(2.0)).push(shift(1.0));
        let y = net.forward(&x(3.0), Mode::EVAL).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![7.0, 7.0]);
        assert_eq!(net.len(), 2);
        assert!(!net.is_empty());

        let mut reversed = Sequential::new().push(shift(1.0)).push(scale(2.0));
        let y = reversed.forward(&x(3.0), Mode::EVAL).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![8.0, 8.0]);
    }

    #[test]
    fn empty_chain_is_the_identity() {
        let mut net = Sequential::new();
        assert!(net.is_empty());
        let y = net.forward(&x(7.0), Mode::EVAL).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![7.0, 7.0]);
    }

    #[test]
    fn children_get_indexed_paths() {
        let net = Sequential::new().push(scale(2.0)).push(scale(5.0));
        let keys: Vec<_> = net.state_dict().into_keys().collect();
        assert_eq!(keys, ["0.calls", "0.factor", "1.calls", "1.factor"]);
        // Only the params count.
        assert_eq!(net.num_params(), 2);
    }

    #[test]
    fn nests_inside_a_derived_module() {
        #[derive(rstorch::Module)]
        struct Net {
            layers: Sequential,
            tail: Scale,
        }
        let net = Net {
            layers: Sequential::new().push(scale(2.0)),
            tail: scale(3.0),
        };
        let keys: Vec<_> = net.state_dict().into_keys().collect();
        assert_eq!(
            keys,
            [
                "layers.0.calls",
                "layers.0.factor",
                "tail.calls",
                "tail.factor"
            ]
        );
    }

    #[test]
    fn state_dict_round_trips_through_a_chain() {
        let src = Sequential::new().push(scale(2.0)).push(scale(5.0));
        let mut dst = Sequential::new().push(scale(0.0)).push(scale(0.0));
        dst.load_state_dict(&src.state_dict()).unwrap();
        let y = dst.forward(&x(1.0), Mode::EVAL).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![10.0, 10.0]);
    }

    #[test]
    fn a_shorter_chain_is_a_loud_mismatch() {
        let src = Sequential::new().push(scale(2.0)).push(scale(5.0));
        let mut dst = Sequential::new().push(scale(0.0));
        let err = dst.load_state_dict(&src.state_dict()).unwrap_err();
        assert!(err.to_string().contains("unexpected key `1."), "{err}");
    }

    #[test]
    fn gradients_flow_through_the_chain() {
        let mut net = Sequential::new().push(scale(2.0)).push(scale(5.0));
        let input = x(1.0).traced().unwrap();
        let loss = net.forward(&input, Mode::TRAIN).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        // d/dx sum(x * 2 * 5) = 10 per element.
        assert_eq!(
            grads.wrt_input(&input).unwrap().to_vec::<f32>().unwrap(),
            vec![10.0, 10.0]
        );
    }

    #[test]
    fn debug_reports_the_length() {
        let net = Sequential::new().push(scale(1.0));
        assert_eq!(format!("{net:?}"), "Sequential(1 layers)");
    }

    /// A two-field input: the tensor plus a mask the layer must actually use.
    /// This is the shape a user reaches for when `Mode` (crate-owned, closed)
    /// has no slot for their context.
    #[derive(Clone)]
    struct Masked {
        values: Tensor,
        keep: Tensor,
    }

    /// A user layer over `Masked`: `x * factor`, zeroed where `keep` is 0.
    #[derive(rstorch::Module)]
    struct MaskedScale {
        factor: Param,
    }

    impl Forward<Masked> for MaskedScale {
        type Output = Masked;

        fn forward(&mut self, input: &Masked, mode: Mode) -> Result<Masked> {
            Ok(Masked {
                values: input.values.mul(&self.factor.get(mode))?.mul(&input.keep)?,
                keep: input.keep.clone(),
            })
        }
    }

    /// A layer whose `Output` is not a `Tensor` at all — the associated type
    /// earning its keep.
    #[derive(rstorch::Module)]
    struct SplitHalves {
        factor: Param,
    }

    impl Forward for SplitHalves {
        type Output = (Tensor, Tensor);

        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<(Tensor, Tensor)> {
            let scaled = x.mul(&self.factor.get(mode))?;
            Ok((scaled.clone(), scaled))
        }
    }

    #[test]
    fn a_user_layer_can_take_more_than_one_tensor() {
        let dev = Device::Cpu;
        let mut layer = MaskedScale {
            factor: Param::new(Tensor::full([1], 2.0, DType::F32, &dev).unwrap()),
        };
        let input = Masked {
            values: Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3], &dev).unwrap(),
            keep: Tensor::from_vec(vec![1.0f32, 0.0, 1.0], vec![3], &dev).unwrap(),
        };
        let out = layer.forward(&input, Mode::EVAL).unwrap();
        // The mask is used, not ignored: the middle element is dropped.
        assert_eq!(out.values.to_vec::<f32>().unwrap(), vec![2.0, 0.0, 6.0]);
    }

    #[test]
    fn a_chain_can_be_built_over_a_user_input_type() {
        let dev = Device::Cpu;
        let factor = |v: f64| MaskedScale {
            factor: Param::new(Tensor::full([1], v, DType::F32, &dev).unwrap()),
        };
        let mut net: Sequential<Masked> = Sequential::new().push(factor(2.0)).push(factor(3.0));
        let input = Masked {
            values: Tensor::from_vec(vec![1.0f32, 2.0], vec![2], &dev).unwrap(),
            keep: Tensor::from_vec(vec![1.0f32, 0.0], vec![2], &dev).unwrap(),
        };
        let out = net.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(out.values.to_vec::<f32>().unwrap(), vec![6.0, 0.0]);
        // The container is still a `Module`, with the same indexed paths.
        let keys: Vec<_> = net.state_dict().into_keys().collect();
        assert_eq!(keys, ["0.factor", "1.factor"]);
    }

    #[test]
    fn an_output_need_not_be_a_tensor() {
        let dev = Device::Cpu;
        let mut layer = SplitHalves {
            factor: Param::new(Tensor::full([1], 3.0, DType::F32, &dev).unwrap()),
        };
        let (a, b) = layer.forward(&x(2.0), Mode::EVAL).unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![6.0, 6.0]);
        assert_eq!(b.to_vec::<f32>().unwrap(), vec![6.0, 6.0]);
    }

    #[test]
    fn an_input_type_need_not_be_default() {
        // `#[derive(Default)]` on the struct would have demanded
        // `Input: Default`; the hand-written impl does not.
        struct NotDefault(#[allow(dead_code)] Tensor);
        let chain: Sequential<NotDefault> = Sequential::default();
        assert!(chain.is_empty());
    }
}
