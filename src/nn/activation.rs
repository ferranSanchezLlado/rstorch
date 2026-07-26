//! Activation layers: [`Relu`] and [`Gelu`] (exploration §4.4).
//!
//! These are unit structs, not functions, for one reason: [`Sequential`] stores
//! `Forward + Module` values, so an activation has to *be* a layer to sit in a
//! chain. Anywhere a chain is not involved, call the tensor method directly —
//! `x.relu()?` — and skip the wrapper entirely; the layer adds nothing but the
//! trait impls.
//!
//! [`Sequential`]: crate::nn::Sequential

use crate::error::Result;
use crate::nn::{Forward, Mode};
use crate::tensor::Tensor;

/// The rectifier `max(x, 0)` as a layer ([`Tensor::relu`]).
///
/// Stateless and parameter-free, so it is `Copy` and behaves identically under
/// every [`Mode`].
///
/// ```
/// use rstorch::nn::{Forward, Mode, Relu, Sequential};
/// # use rstorch::{Device, Rng, Tensor};
/// # let dev = Device::Cpu;
/// # let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(rstorch::nn::Linear::new(4, 3, &dev, &mut rng)?)
///     .push(Relu);
/// let x = Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], [1, 4], &dev)?;
/// assert!(net.forward(&x, Mode::EVAL)?.to_vec::<f32>()?.iter().all(|&v| v >= 0.0));
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, rstorch::Module)]
pub struct Relu;

impl Forward for Relu {
    /// `max(x, 0)`, ignoring `mode` (there is no state and no parameter to
    /// read, so nothing here depends on it).
    ///
    /// # Errors
    ///
    /// Whatever [`Tensor::relu`] reports (a dtype without a kernel).
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        x.relu()
    }
}

/// The **exact** Gaussian error linear unit `x · Φ(x)` as a layer
/// ([`Tensor::gelu`]).
///
/// Exact, not the `tanh` approximation: the distinction is asserted in the T31
/// numerics suite, and this layer is a thin wrapper over the same op, so it
/// inherits it.
///
/// ```
/// use rstorch::nn::{Forward, Gelu, Mode};
/// # use rstorch::{Device, Tensor};
/// let mut act = Gelu;
/// let x = Tensor::from_vec(vec![0.0f32], [1], &Device::Cpu)?;
/// assert_eq!(act.forward(&x, Mode::EVAL)?.to_vec::<f32>()?, vec![0.0]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, rstorch::Module)]
pub struct Gelu;

impl Forward for Gelu {
    /// `x · Φ(x)`, ignoring `mode` (see [`Relu::forward`]).
    ///
    /// # Errors
    ///
    /// Whatever [`Tensor::gelu`] reports (a dtype without a kernel).
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        x.gelu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::nn::{self, Module, Sequential};
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32]) -> Tensor {
        Tensor::from_vec(data.to_vec(), [data.len()], &CPU).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    #[test]
    fn relu_clamps_at_zero() {
        let mut act = Relu;
        let y = act
            .forward(&t(&[-2.0, -0.5, 0.0, 0.5, 2.0]), Mode::EVAL)
            .unwrap();
        assert_eq!(v(&y), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn gelu_matches_the_op_it_wraps() {
        let x = t(&[-2.0, -0.5, 0.0, 0.5, 2.0]);
        let mut act = Gelu;
        assert_eq!(
            v(&act.forward(&x, Mode::EVAL).unwrap()),
            v(&x.gelu().unwrap())
        );
    }

    #[test]
    fn shapes_are_preserved() {
        let x = Tensor::zeros([2, 3, 4], crate::dtype::DType::F32, &CPU).unwrap();
        assert_eq!(Relu.forward(&x, Mode::EVAL).unwrap().dims(), &[2, 3, 4]);
        assert_eq!(Gelu.forward(&x, Mode::EVAL).unwrap().dims(), &[2, 3, 4]);
    }

    #[test]
    fn behavior_is_mode_independent() {
        let x = t(&[-1.0, 0.5, 3.0]);
        for mode in [
            Mode::TRAIN,
            Mode::EVAL,
            Mode::TRAIN.frozen(),
            Mode::EVAL.recorded(),
        ] {
            assert_eq!(v(&Relu.forward(&x, mode).unwrap()), vec![0.0, 0.5, 3.0]);
            assert_eq!(
                v(&Gelu.forward(&x, mode).unwrap()),
                v(&x.gelu().unwrap()),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn they_hold_no_parameters() {
        assert!(nn::state_dict(&Relu).is_empty());
        assert!(nn::state_dict(&Gelu).is_empty());
        assert_eq!(nn::num_params(&Relu) + nn::num_params(&Gelu), 0);
    }

    #[test]
    fn gradients_match_finite_differences() {
        // Away from the kink for ReLU (a central difference straddling zero
        // measures the average of the two one-sided slopes, not either one).
        check_grad(
            |i| Relu.forward(&i[0], Mode::TRAIN)?.mul(&i[0])?.sum_all(),
            &[t(&[1.0, -2.0, 0.5, 3.0, -1.5])],
            1e-3,
            1e-3,
        )
        .unwrap();
        check_grad(
            |i| Gelu.forward(&i[0], Mode::TRAIN)?.mul(&i[0])?.sum_all(),
            &[t(&[1.0, -2.0, 0.5, 0.0, -1.5])],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn they_compose_in_a_sequential() {
        // The composition test the layers exist for: activations between two
        // affine maps, gradients flowing to both the input and the parameters.
        use crate::nn::{Dropout, Linear};
        use crate::rng::Rng;

        let mut rng = Rng::seed(20);
        let mut net = Sequential::new()
            .push(Linear::new(4, 3, &CPU, &mut rng).unwrap())
            .push(Relu)
            .push(Dropout::new(0.5, &mut rng).unwrap())
            .push(Linear::new(3, 2, &CPU, &mut rng).unwrap())
            .push(Gelu);

        // Only the two `Linear`s contribute leaves, and they are indexed by
        // position in the chain.
        assert_eq!(
            nn::state_dict(&net).into_keys().collect::<Vec<_>>(),
            ["0.bias", "0.weight", "3.bias", "3.weight"]
        );
        assert_eq!(nn::num_params(&net), (4 * 3 + 3) + (3 * 2 + 2));

        let x = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], [1, 4], &CPU).unwrap();
        let y = net.forward(&x, Mode::EVAL).unwrap();
        assert_eq!(y.dims(), &[1, 2]);

        // Under TRAIN the whole chain is differentiable end to end.
        let traced = x.traced().unwrap();
        let grads = net
            .forward(&traced, Mode::TRAIN)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(grads.wrt_input(&traced).unwrap().dims(), &[1, 4]);
        // …and each layer's parameters received a gradient of the right shape.
        let mut params: Vec<(String, Vec<usize>)> = Vec::new();
        net.visit(&mut nn::Visitor::new(&mut |path, leaf| {
            if let crate::nn::visit::Leaf::Param(p) = leaf {
                params.push((path.to_string(), grads.wrt(p).unwrap().dims().to_vec()));
            }
        }));
        assert_eq!(
            params,
            vec![
                ("0.weight".to_string(), vec![3, 4]),
                ("0.bias".to_string(), vec![3]),
                ("3.weight".to_string(), vec![2, 3]),
                ("3.bias".to_string(), vec![2]),
            ]
        );
    }

    #[test]
    fn a_deep_chain_is_differentiable_under_finite_differences() {
        // Smooth activations only: a `Relu` kink inside the chain would put the
        // finite difference across a discontinuity in the derivative.
        use crate::nn::Linear;
        use crate::rng::Rng;

        let mut rng = Rng::seed(21);
        let net = std::cell::RefCell::new(
            Sequential::new()
                .push(Linear::new(3, 4, &CPU, &mut rng).unwrap())
                .push(Gelu)
                .push(Linear::new(4, 1, &CPU, &mut rng).unwrap()),
        );
        check_grad(
            |i| net.borrow_mut().forward(&i[0], Mode::TRAIN)?.sum_all(),
            &[Tensor::from_vec(vec![0.5f32, -1.0, 2.0], [1, 3], &CPU).unwrap()],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    /// `T::default()` behind a generic, because clippy (rightly) rejects
    /// spelling `Relu::default()` for a unit struct — the point here is that
    /// the bound is *satisfied*, which a generic call proves and a literal
    /// does not.
    fn default_of<T: Default>() -> T {
        T::default()
    }

    #[test]
    fn derives_are_unit_shaped() {
        assert_eq!(Relu, default_of::<Relu>());
        assert_eq!(Gelu, default_of::<Gelu>());
        assert_eq!(format!("{Relu:?} {Gelu:?}"), "Relu Gelu");
        // `Copy`: passing one by value does not move it away.
        let act = Relu;
        let _copy = act;
        assert_eq!(act, Relu);
    }
}
