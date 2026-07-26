//! Fixtures shared by the optimizer test suites (test builds only).
//!
//! Wave 4's layer zoo (`Linear`, `LayerNorm`, ...) is a sibling task, so these
//! models spell their arithmetic out by hand. Their *field names* matter: the
//! group-predicate tests match on `bias` and `norm`, which is the standard
//! transformer recipe the design calls out.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::nn::{Mode, Param};
use crate::tensor::Tensor;

pub(crate) const CPU: Device = Device::Cpu;

/// A 1-D `f32` tensor on the CPU.
pub(crate) fn t(values: &[f32]) -> Tensor {
    Tensor::from_vec(values.to_vec(), [values.len()], &CPU).unwrap()
}

/// The host values of a tensor.
pub(crate) fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

/// A scalar tensor's value.
pub(crate) fn scalar(tensor: &Tensor) -> f64 {
    tensor.item().unwrap()
}

/// The affine map `y = x·w + b`, the smallest thing that can actually converge.
///
/// `weight` is `[1, 1]` and `bias` is `[1]`, so a `[n, 1]` batch of inputs maps
/// to a `[n, 1]` batch of predictions and the whole model is two numbers.
#[derive(rstorch::Module)]
pub(crate) struct Affine {
    pub(crate) weight: Param,
    pub(crate) bias: Param,
}

impl Affine {
    /// Start at `w = b = 0`, far from the target the tests fit.
    pub(crate) fn zeros() -> Affine {
        Affine {
            weight: Param::new(Tensor::zeros([1, 1], DType::F32, &CPU).unwrap()),
            bias: Param::new(Tensor::zeros([1], DType::F32, &CPU).unwrap()),
        }
    }

    /// Mean-squared error of this model's predictions on `xs` against
    /// `3·x + 2`, the line the convergence tests fit.
    pub(crate) fn loss(&self, xs: &[f32], mode: Mode) -> Result<Tensor> {
        let n = xs.len();
        let x = Tensor::from_vec(xs.to_vec(), [n, 1], &CPU)?;
        let targets: Vec<f32> = xs.iter().map(|v| 3.0 * v + 2.0).collect();
        let y = Tensor::from_vec(targets, [n, 1], &CPU)?;
        let pred = x
            .matmul(&self.weight.get(mode))?
            .add(&self.bias.get(mode))?;
        pred.mse_loss(&y)
    }
}

/// A single parameter under the loss `Σ w²`, whose gradient `2w` **changes** as
/// the parameter moves.
///
/// That is the point: an update formula pinned against a *constant* gradient can
/// be satisfied by several wrong formulas (a mis-ordered bias correction, decay
/// applied to the wrong operand), because every step looks the same. With `2w`
/// the trajectory is sensitive to all of it, so a test can compare it against an
/// independent scalar implementation of the algorithm and mean it.
#[derive(rstorch::Module)]
pub(crate) struct Solo {
    pub(crate) w: Param,
}

impl Solo {
    /// One parameter holding `value`.
    pub(crate) fn new(value: f32) -> Solo {
        Solo {
            w: Param::new(t(&[value])),
        }
    }

    /// `Σ w²`, so `dL/dw = 2w`.
    pub(crate) fn square_loss(&self, mode: Mode) -> Result<Tensor> {
        let w = self.w.get(mode);
        w.mul(&w)?.sum_all()
    }

    /// The parameter's current scalar value.
    pub(crate) fn value(&self) -> f64 {
        f64::from(values(self.w.value())[0])
    }
}

/// A two-level model whose dotted paths cover what group predicates select on:
/// `trunk.weight`, `trunk.bias`, `norm.weight`, `head.weight`, `head.bias`.
#[derive(rstorch::Module)]
pub(crate) struct Net {
    pub(crate) trunk: Block,
    pub(crate) norm: Gain,
    pub(crate) head: Block,
}

/// A weight/bias pair.
#[derive(rstorch::Module)]
pub(crate) struct Block {
    pub(crate) weight: Param,
    pub(crate) bias: Param,
}

/// A lone gain, standing in for a normalization layer's scale.
#[derive(rstorch::Module)]
pub(crate) struct Gain {
    pub(crate) weight: Param,
}

impl Net {
    /// Every parameter starts at `1.0`, so a multiplicative or additive update
    /// is visible in the value itself.
    pub(crate) fn ones() -> Net {
        let p = || Param::new(t(&[1.0]));
        Net {
            trunk: Block {
                weight: p(),
                bias: p(),
            },
            norm: Gain { weight: p() },
            head: Block {
                weight: p(),
                bias: p(),
            },
        }
    }

    /// A loss whose gradient is exactly `1.0` at every parameter: the sum of
    /// all five. That makes each optimizer's update formula readable straight
    /// off the resulting values.
    pub(crate) fn unit_grad_loss(&self, mode: Mode) -> Result<Tensor> {
        let mut acc = self.trunk.weight.get(mode);
        for p in [
            &self.trunk.bias,
            &self.norm.weight,
            &self.head.weight,
            &self.head.bias,
        ] {
            acc = acc.add(&p.get(mode))?;
        }
        acc.sum_all()
    }

    /// The **untraced-weight scenario**: a forward that reads
    /// [`Param::value`](crate::nn::Param::value) for `head.bias` instead of
    /// [`Param::get`](crate::nn::Param::get), which is exactly how a real model
    /// stops training one weight without any other symptom. Its gradient is
    /// therefore absent from the resulting `Grads`.
    pub(crate) fn untraced_head_bias_loss(&self, mode: Mode) -> Result<Tensor> {
        let mut acc = self.trunk.weight.get(mode);
        for p in [&self.trunk.bias, &self.norm.weight, &self.head.weight] {
            acc = acc.add(&p.get(mode))?;
        }
        // The bug: the value, not the traced leaf.
        acc.add(self.head.bias.value())?.sum_all()
    }

    /// Every parameter's value, in `state_dict` (sorted-path) order.
    pub(crate) fn snapshot(&self) -> Vec<(String, f32)> {
        crate::nn::state_dict(self)
            .into_iter()
            .map(|(path, tensor)| (path, values(&tensor)[0]))
            .collect()
    }

    /// One parameter's scalar value, by dotted path.
    pub(crate) fn at(&self, path: &str) -> f32 {
        self.snapshot()
            .into_iter()
            .find(|(p, _)| p == path)
            .unwrap_or_else(|| panic!("no parameter at `{path}`"))
            .1
    }
}

/// `assert!` on `|a - b| < tol`, reporting both sides.
pub(crate) fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} vs {b} (tolerance {tol})");
}

/// A private temp directory for a checkpoint round-trip, removed by the caller.
pub(crate) fn tmpdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rstorch-optim-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
