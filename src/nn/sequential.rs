//! [`Sequential`] — a chain of layers behind one [`Forward`] (exploration
//! §4.1).
//!
//! The container is a plain `Vec` of boxed layers. Its only subtlety is a Rust
//! one: `dyn Forward + Module` is not a legal type (E0225 — at most one
//! non-auto trait per object type), so the box is typed by a crate-private
//! combining trait `SeqLayer` with a blanket impl. Nothing about that leaks:
//! the public surface is `push(impl Forward + Module + Send + 'static)`, and
//! the crate's public trait count is unchanged.

use crate::error::Result;
use crate::nn::{Forward, Mode, Module, Visitor, VisitorMut};
use crate::tensor::Tensor;

/// The stored-layer trait: everything a [`Sequential`] element must be, in one
/// object-safe bundle. Crate-private with a blanket impl, so users never name
/// it — they satisfy it by implementing [`Forward`] and [`Module`].
///
/// `Send` is required so a `Sequential` (and any model holding one) can move
/// between threads, matching the rest of the crate's types.
pub(crate) trait SeqLayer: Forward + Module + Send {}

impl<T: Forward + Module + Send> SeqLayer for T {}

/// A chain of layers applied in order, itself a [`Forward`] **and** a
/// [`Module`] — so it nests inside another `Sequential` or a
/// `#[derive(Module)]` struct.
///
/// Children are visited with their **index** as the path segment, exactly as
/// the derive treats a `Vec<M>`: a `Sequential` field named `layers` produces
/// `layers.0.weight`, `layers.1.weight`, … Those are the on-disk `state_dict`
/// keys, so inserting a layer in the middle renames every later path — the
/// same trade PyTorch's `nn.Sequential` makes.
///
/// ```
/// # use rstorch::nn::{self, Forward, Mode, Module, Param, Sequential};
/// # use rstorch::{DType, Device, Result, Tensor};
/// # #[derive(rstorch::Module)]
/// # struct Scale { factor: Param }
/// # impl Forward for Scale {
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
/// assert_eq!(nn::state_dict(&net).keys().collect::<Vec<_>>(), ["0.factor", "1.factor"]);
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn SeqLayer>>,
}

impl Sequential {
    /// An empty chain. Empty is legal and forwards its input unchanged.
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new() }
    }

    /// Append `layer` and return the chain, so construction reads as one
    /// expression:
    /// `Sequential::new().push(Linear::new(..)?).push(Relu)`.
    ///
    /// Building in a loop is the same call:
    /// `for _ in 0..n { net = net.push(block()?); }`.
    #[must_use]
    pub fn push(mut self, layer: impl Forward + Module + Send + 'static) -> Sequential {
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

impl std::fmt::Debug for Sequential {
    /// Layers are trait objects with no `Debug` bound, so the chain reports
    /// its length rather than its contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({} layers)", self.layers.len())
    }
}

impl Module for Sequential {
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

impl Forward for Sequential {
    /// Apply every layer in order, threading `mode` through unchanged. An
    /// empty chain returns `x` (an `Arc` bump).
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let mut current = x.clone();
        for layer in &mut self.layers {
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
    use crate::nn::{self, Param};

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
        let keys: Vec<_> = nn::state_dict(&net).into_keys().collect();
        assert_eq!(keys, ["0.calls", "0.factor", "1.calls", "1.factor"]);
        // Only the params count.
        assert_eq!(nn::num_params(&net), 2);
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
        let keys: Vec<_> = nn::state_dict(&net).into_keys().collect();
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
        nn::load_state_dict(&mut dst, &nn::state_dict(&src)).unwrap();
        let y = dst.forward(&x(1.0), Mode::EVAL).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![10.0, 10.0]);
    }

    #[test]
    fn a_shorter_chain_is_a_loud_mismatch() {
        let src = Sequential::new().push(scale(2.0)).push(scale(5.0));
        let mut dst = Sequential::new().push(scale(0.0));
        let err = nn::load_state_dict(&mut dst, &nn::state_dict(&src)).unwrap_err();
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
}
