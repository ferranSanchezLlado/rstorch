//! Sealed output contracts and normative typed operation signatures.
//!
//! Associated outputs avoid unstable generic const expressions. Their bounds
//! identify an operation's output family; they do not by themselves prove
//! every const-geometry relationship described below. Operation owners generate
//! complete rank-table implementations and compile-test those relationships.
//! In the signature inventory, notation such as `OutputTrait<Self, ...>` means
//! the Rust type `<Self as OutputTrait<...>>::Output`; it is not a public alias.
//!
//! # Shape and views (CT21)
//!
//! CT21 provides the following inherent methods. `Target::Dims` is passed
//! explicitly for reshape/broadcast because `DYN` values are runtime data.
//! Const-axis methods exist only where their associated output is implemented;
//! an invalid axis is therefore an ordinary trait error.
//!
//! ```text
//! reshape<Target>(&self, dims: Target::Dims) -> Result<Target>
//! transpose<const A: usize, const B: usize>(&self)
//!     -> Result<TransposeOutput<Self, A, B>>
//! squeeze<const AXIS: usize>(&self) -> Result<RemoveAxisOutput<Self, AXIS>>
//! unsqueeze<const AXIS: usize>(&self)
//!     -> Result<InsertAxisOutput<Self, AXIS, 1>>
//! narrow<const AXIS: usize>(&self, start: usize, len: usize)
//!     -> Result<ReplaceAxisOutput<Self, AXIS, DYN>>
//! broadcast_to<Target>(&self, dims: Target::Dims) -> Result<Target>
//! permute(&self, axes: &[isize]) -> Result<DynamicOutput<Self>>
//! cat<const AXIS: usize>(tensors: &[&Self]) -> Result<ConcatOutput<Self, AXIS>>
//! stack<const AXIS: usize>(tensors: &[&Self]) -> Result<StackOutput<Self, AXIS>>
//! ```
//!
//! `reshape` is caller-targeted and checks numel. `broadcast_to` is the only
//! implicit-broadcast preparation for strict elementwise operations. Runtime
//! permutation and runtime-axis transpose/narrow return the same rank with all
//! markers `DYN`; runtime-axis squeeze/unsqueeze return [`crate::Tensor`]
//! because their output rank is selected at runtime. Those escape spellings are
//! `transpose_dyn`, `narrow_dyn`, `squeeze_dyn`, and `unsqueeze_dyn`, with the
//! same runtime arguments as the dynamic operation and `Result` outputs just
//! described. Rank-increasing methods and `stack` exist only through rank 7;
//! no such typed method exists on `Tensor8`.
//!
//! # Elementwise (CT22)
//!
//! Binary methods `add`, `sub`, `mul`, `div`, `maximum`, and `minimum` take
//! `(&self, rhs: &Self) -> Result<Self>`. Comparisons `eq`, `ne`, `lt`, `le`,
//! `gt`, and `ge` take the same receiver/argument and return
//! `Result<BooleanOutput<Self>>`. Scalar methods `add_scalar`, `sub_scalar`,
//! `mul_scalar`, and `div_scalar` take `(&self, value: f64) -> Result<Self>`.
//! Shape-preserving unary methods are `neg`, `abs`, `exp`, `ln`, `sqrt`,
//! `square`, `relu`, `gelu`, `tanh`, and `sigmoid`, each `(&self) -> Result<Self>`.
//! `masked_fill(&self, mask: &BooleanOutput<Self>,
//! value: f64) -> Result<Self>` and
//! `where_cond(&self, on_true: &T, on_false: &T) -> Result<T>` require one exact
//! typed shape, dtype, and placement. Matching `DYN` markers still require
//! equal actual dimensions at runtime.
//!
//! `Add`, `Sub`, `Mul`, and `Div` sugar is strict for all owned/borrowed
//! combinations and scalar right-hand sides. Sugar panics with the named
//! method's structured error; named methods never panic for runtime failures.
//! There is no implicit typed broadcasting.
//!
//! The exact element bounds are: `NumericElement` for arithmetic, scalar
//! arithmetic, maximum/minimum, neg, abs, and square; `FloatElement` for relu,
//! gelu, exp, ln, sqrt, tanh, and sigmoid; plain `Element` for comparisons,
//! masked fill, and where. Masks are exactly `bool`.
//!
//! # Reductions (CT23)
//!
//! For each of `sum`, `mean`, `max`, `min`, `var`, and `std`:
//!
//! ```text
//! op<const AXIS: usize>(&self) -> Result<RemoveAxisOutput<Self, AXIS>>
//! op_keepdim<const AXIS: usize>(&self) -> Result<KeepDimOutput<Self, AXIS>>
//! op_all(&self) -> Result<ScalarOutput<Self>>
//! ```
//!
//! `softmax<const AXIS>` and `log_softmax<const AXIS>` preserve `Self`.
//! `argmax<const AXIS>` and `argmin<const AXIS>` return `ArgOutput<Self, AXIS>`;
//! keepdim variants return `ArgKeepDimOutput<Self, AXIS>`. Arg outputs retain
//! placement and use `i64`. Runtime-axis escape spellings append `_dyn`, take
//! `axis: isize`, and return an all-`DYN` wrapper of the known output rank (or
//! `Self` for softmax). Every method returns `Result`.
//! Sum, mean, min, max, argmin, and argmax require `NumericElement`; var, std,
//! softmax, and log-softmax require `FloatElement`. Bool has no typed reduction.
//!
//! # Matmul (CT24)
//!
//! `matmul<Rhs>(&self, rhs: &Rhs) -> Result<MatmulOutput<Self, Rhs>>` exists only
//! for rank-2 by rank-2, rank-3 through rank-8 by unbatched rank-2 weight, and
//! same-rank rank-3 through rank-8 with identical typed batch prefixes. The
//! latter compares actual batch prefixes too. There is no vector promotion and
//! no batch broadcasting. Known contraction mismatch is a body const failure;
//! any `DYN` contraction mismatch is a runtime shape error.
//! Both operands require the same `NumericElement` type and placement. Their
//! contracted const markers are independent generic constants, so a static
//! marker and `DYN` compile and defer their relationship to the method body.
//!
//! # Indexing (CT25)
//!
//! ```text
//! Tensor1::<LEN, i64, P>::arange(start: i64, end: i64, ctx: &DeviceCtx<P>)
//!     -> Result<Self>
//! Tensor1::<LEN, i64, P>::from_indices(values: Vec<i64>, ctx: &DeviceCtx<P>)
//!     -> Result<Self>
//! index_select<const AXIS, const LEN>(&self, indices: &Tensor1<LEN, i64, P>)
//!     -> Result<IndexSelectOutput<Self, AXIS, Tensor1<LEN, i64, P>>>
//! gather<const AXIS, Indices>(&self, indices: &Indices)
//!     -> Result<GatherOutput<Self, Indices>>
//! Tensor2::<S, S, bool, P>::causal_mask(seq_len: usize, ctx: &DeviceCtx<P>)
//!     -> Result<Self>
//! ```
//!
//! Gather returns index geometry with source element type. Index rank/axis,
//! dtype, and placement are static; index values and bounds remain runtime.
//! A causal mask is square. `seq_len` is checked against static `S`, or retained
//! as the actual size when `S == DYN`.
//!
//! # Convolution and pooling (CT26)
//!
//! ```text
//! // self is Tensor4<BATCH, INPUT_CHANNELS, H, W, E, P>
//! conv2d<const OUT, const WEIGHT_INPUT, const KH, const KW>(
//!        &self, weight: &Tensor4<OUT, WEIGHT_INPUT, KH, KW, E, P>,
//!        stride: (usize, usize), padding: (usize, usize),
//!        dilation: (usize, usize)) -> Result<Conv2dOutput<Self, Weight>>
//! max_pool2d(&self, kernel: (usize, usize), stride: (usize, usize),
//!            padding: (usize, usize)) -> Result<Pool2dOutput<Self>>
//! avg_pool2d(&self, kernel: (usize, usize), stride: (usize, usize),
//!            padding: (usize, usize)) -> Result<Pool2dOutput<Self>>
//! ```
//!
//! Inputs are NCHW rank 4. Conv retains batch and output-channel markers;
//! pooling retains batch/channel markers; computed spatial axes are `DYN`.
//! Known channel mismatch is a body const failure and `DYN` mismatch is runtime.
//! `INPUT_CHANNELS` and `WEIGHT_INPUT` are independent consts so static-versus-
//! `DYN` is accepted. Conv and pool require `NumericElement`; Bool is absent.
//!
//! # Losses (CT27)
//!
//! `mse_loss(&self, target: &Self) -> Result<Tensor0<E, P>>` is strict and
//! never broadcasts. Rank-2 float logits provide
//! `cross_entropy<const TARGET_ROWS>(&self,
//! targets: &Tensor1<TARGET_ROWS, i64, P>)` and
//! `cross_entropy_ignore_index<const TARGET_ROWS>(&self,
//! targets: &Tensor1<TARGET_ROWS, i64, P>,
//! ignore_index: i64)`, both returning `Result<Tensor0<E, P>>`. Known row
//! mismatch is a body const failure; `DYN` rows defer to runtime.
//! The logits row marker and `TARGET_ROWS` are independent. All losses require
//! `FloatElement`.

use super::{
    DYN, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7,
    Tensor8, TypedTensor,
};
use crate::{Element, Error};

mod conv;
mod elementwise;
mod index;
mod loss;
mod matmul;
mod reduce;
mod shape;

mod sealed {
    pub trait OutputContract {}
}

fn relabel_error_op(op: &'static str, error: Error) -> Error {
    match error {
        Error::ShapeMismatch { lhs, rhs, .. } => Error::ShapeMismatch { op, lhs, rhs },
        Error::RankMismatch { expected, got, .. } => Error::RankMismatch { op, expected, got },
        Error::InvalidAxis { axis, rank, .. } => Error::InvalidAxis { op, axis, rank },
        Error::DTypeMismatch { expected, got, .. } => Error::DTypeMismatch { op, expected, got },
        Error::DeviceMismatch { expected, got, .. } => Error::DeviceMismatch { op, expected, got },
        Error::ReshapeMismatch { from, to, .. } => Error::ReshapeMismatch { op, from, to },
        Error::IndexOutOfBounds {
            index, axis, size, ..
        } => Error::IndexOutOfBounds {
            op,
            index,
            axis,
            size,
        },
        Error::Unsupported { device, dtype, .. } => Error::Unsupported { op, device, dtype },
        Error::NotTraced { .. } => Error::NotTraced { op },
        Error::InvalidArg { msg, .. } => Error::InvalidArg { op, msg },
        Error::Backend { msg, .. } => Error::Backend { op, msg },
        other => other,
    }
}

/// Changes only element type.
#[doc(hidden)]
pub trait WithElement<F: Element>: sealed::OutputContract {
    /// Output with identical geometry and placement.
    type Output: TypedTensor<Elem = F>;
}

/// Changes only logical placement.
#[doc(hidden)]
pub trait WithPlacement<Q: Placement>: sealed::OutputContract {
    /// Output with identical geometry and element type.
    type Output: TypedTensor<Placement = Q>;
}

/// Comparison output preserving geometry and placement.
#[doc(hidden)]
pub trait BooleanOutput: sealed::OutputContract {
    /// Boolean output tensor.
    type Output: TypedTensor<Elem = bool>;
}

/// Same-rank output with every marker erased to [`DYN`].
#[doc(hidden)]
pub trait DynamicOutput: sealed::OutputContract {
    /// Erased output tensor.
    type Output: TypedTensor;
}

/// Strict binary compatibility, implemented only for an identical type.
#[doc(hidden)]
pub trait StrictlyCompatible<Rhs = Self>: sealed::OutputContract {
    /// Unchanged common type.
    type Output: TypedTensor;
}

impl<T> StrictlyCompatible<T> for T
where
    T: TypedTensor + sealed::OutputContract,
{
    type Output = T;
}

/// Whole-reduction output.
#[doc(hidden)]
pub trait ScalarOutput: sealed::OutputContract {
    /// Rank-zero output retaining element and placement.
    type Output: TypedTensor;
}

/// Const-axis transpose output.
#[doc(hidden)]
pub trait TransposeOutput<const A: usize, const B: usize>: sealed::OutputContract {
    /// Output with the selected markers exchanged.
    type Output: TypedTensor;
}

/// Const-axis removal output.
#[doc(hidden)]
pub trait RemoveAxisOutput<const AXIS: usize>: sealed::OutputContract {
    /// Rank-minus-one output.
    type Output: TypedTensor;
}

/// Const-axis insertion output.
#[doc(hidden)]
pub trait InsertAxisOutput<const AXIS: usize, const DIM: usize>: sealed::OutputContract {
    /// Rank-plus-one output.
    type Output: TypedTensor;
}

/// Const-axis replacement output.
#[doc(hidden)]
pub trait ReplaceAxisOutput<const AXIS: usize, const DIM: usize>: sealed::OutputContract {
    /// Same-rank output with one replaced marker.
    type Output: TypedTensor;
}

/// Keepdim reduction output.
#[doc(hidden)]
pub trait KeepDimOutput<const AXIS: usize>: sealed::OutputContract {
    /// Same-rank output with the selected marker replaced by one.
    type Output: TypedTensor;
}

/// Caller-targeted reshape output.
#[doc(hidden)]
pub trait ReshapeOutput<Target: TypedTensor>: sealed::OutputContract {
    /// Checked caller-named target.
    type Output: TypedTensor;
}

/// Explicit broadcast output.
#[doc(hidden)]
pub trait BroadcastOutput<Target: TypedTensor>: sealed::OutputContract {
    /// Checked caller-named target.
    type Output: TypedTensor;
}

/// Checked refinement from another typed wrapper.
#[doc(hidden)]
pub trait RefinementOf<Source: TypedTensor>: TypedTensor + sealed::OutputContract {}

/// Matmul output under the exact-prefix policy.
#[doc(hidden)]
pub trait MatmulOutput<Rhs: TypedTensor>: sealed::OutputContract {
    /// Matmul result.
    type Output: TypedTensor;
}

/// Arg reduction output with its selected axis removed.
#[doc(hidden)]
pub trait ArgOutput<const AXIS: usize>: sealed::OutputContract {
    /// `i64` output retaining geometry and placement policy.
    type Output: TypedTensor<Elem = i64>;
}

/// Keepdim arg reduction output.
#[doc(hidden)]
pub trait ArgKeepDimOutput<const AXIS: usize>: sealed::OutputContract {
    /// `i64` output with selected marker one.
    type Output: TypedTensor<Elem = i64>;
}

/// Gather output.
#[doc(hidden)]
pub trait GatherOutput<Indices: TypedTensor>: sealed::OutputContract {
    /// Index geometry with source element and placement.
    type Output: TypedTensor;
}

/// Index-select output.
#[doc(hidden)]
pub trait IndexSelectOutput<const AXIS: usize, Indices: TypedTensor>:
    sealed::OutputContract
{
    /// Source geometry with selected axis replaced by index length.
    type Output: TypedTensor;
}

/// Concatenation output.
#[doc(hidden)]
pub trait ConcatOutput<const AXIS: usize>: sealed::OutputContract {
    /// Same-rank output with a `DYN` concatenation axis.
    type Output: TypedTensor;
}

/// Stacking output.
#[doc(hidden)]
pub trait StackOutput<const AXIS: usize>: sealed::OutputContract {
    /// Rank-plus-one output with a `DYN` input-count axis.
    type Output: TypedTensor;
}

/// Conv2d output.
#[doc(hidden)]
pub trait Conv2dOutput<Weight: TypedTensor>: sealed::OutputContract {
    /// NCHW output retaining batch/output channels and erasing spatial axes.
    type Output: TypedTensor;
}

/// Pool2d output.
#[doc(hidden)]
pub trait Pool2dOutput: sealed::OutputContract {
    /// NCHW output retaining batch/channels and erasing spatial axes.
    type Output: TypedTensor;
}

/// Loss output relationship.
#[doc(hidden)]
pub trait LossOutput<Target: TypedTensor>: sealed::OutputContract {
    /// Rank-zero loss output.
    type Output: TypedTensor;
}

macro_rules! impl_simple_outputs {
    ($(($name:ident, [$($dim:ident),*], [$($dyn:expr),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement> sealed::OutputContract
                for $name<$($dim,)* E, P>
            {}

            impl<$(const $dim: usize,)* E: Element, F: Element, P: Placement> WithElement<F>
                for $name<$($dim,)* E, P>
            {
                type Output = $name<$($dim,)* F, P>;
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement, Q: Placement> WithPlacement<Q>
                for $name<$($dim,)* E, P>
            {
                type Output = $name<$($dim,)* E, Q>;
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> BooleanOutput
                for $name<$($dim,)* E, P>
            {
                type Output = $name<$($dim,)* bool, P>;
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> DynamicOutput
                for $name<$($dim,)* E, P>
            {
                type Output = $name<$($dyn,)* E, P>;
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> ScalarOutput
                for $name<$($dim,)* E, P>
            {
                type Output = Tensor0<E, P>;
            }
        )+
    };
}

impl_simple_outputs! {
    (Tensor0, [], []),
    (Tensor1, [D0], [DYN]),
    (Tensor2, [D0, D1], [DYN, DYN]),
    (Tensor3, [D0, D1, D2], [DYN, DYN, DYN]),
    (Tensor4, [D0, D1, D2, D3], [DYN, DYN, DYN, DYN]),
    (Tensor5, [D0, D1, D2, D3, D4], [DYN, DYN, DYN, DYN, DYN]),
    (Tensor6, [D0, D1, D2, D3, D4, D5], [DYN, DYN, DYN, DYN, DYN, DYN]),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6], [DYN, DYN, DYN, DYN, DYN, DYN, DYN]),
    (Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7], [DYN, DYN, DYN, DYN, DYN, DYN, DYN, DYN]),
}

impl<Source, Target> ReshapeOutput<Target> for Source
where
    Source: TypedTensor + sealed::OutputContract,
    Target: TypedTensor<Elem = Source::Elem, Placement = Source::Placement>,
{
    type Output = Target;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::Cpu;

    struct Pair<const D0: usize, const D1: usize>;
    struct One<const D0: usize>;

    impl<const D0: usize, const D1: usize> sealed::OutputContract for Pair<D0, D1> {}
    impl<const D0: usize> sealed::OutputContract for One<D0> {}

    impl<const D0: usize, const D1: usize> TransposeOutput<0, 1> for Pair<D0, D1> {
        type Output = Tensor2<D1, D0>;
    }

    impl<const D0: usize, const D1: usize> RemoveAxisOutput<0> for Pair<D0, D1> {
        type Output = Tensor1<D1>;
    }

    impl<const D0: usize> InsertAxisOutput<1, 1> for One<D0> {
        type Output = Tensor2<D0, 1>;
    }

    impl<const D0: usize, const D1: usize> KeepDimOutput<1> for Pair<D0, D1> {
        type Output = Tensor2<D0, 1>;
    }

    impl<const D0: usize, const D1: usize> ArgOutput<1> for Pair<D0, D1> {
        type Output = Tensor1<D0, i64, Cpu>;
    }

    fn assert_transpose<T, Expected>()
    where
        T: TransposeOutput<0, 1, Output = Expected>,
        Expected: TypedTensor,
    {
    }

    fn assert_remove<T, Expected>()
    where
        T: RemoveAxisOutput<0, Output = Expected>,
        Expected: TypedTensor,
    {
    }

    fn assert_insert<T, Expected>()
    where
        T: InsertAxisOutput<1, 1, Output = Expected>,
        Expected: TypedTensor,
    {
    }

    #[test]
    fn representative_rank_changing_outputs_compile_on_msrv() {
        assert_transpose::<Pair<2, 3>, Tensor2<3, 2, f32, Cpu>>();
        assert_remove::<Pair<2, 3>, Tensor1<3, f32, Cpu>>();
        assert_insert::<One<3>, Tensor2<3, 1, f32, Cpu>>();
    }

    // The relational dimension check is not probed here. It used to be, via a
    // local re-declaration of the predicate, which pinned only the copy: the
    // library's real check is `const_check::assert_matmul_contract`, whose
    // accept side is instantiated by `const_check`'s own `matmul_probe` and
    // whose reject side is pinned as an `E0080` by the build-mode UI harness
    // (a const-assert failure cannot live in an ordinary unit-test target).
}
