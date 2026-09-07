//! Typed operations and the sealed output contracts that name their results.
//!
//! Associated outputs exist because generic const expressions are unstable: a
//! trait bound is the only stable way to say "this op's output has one fewer
//! axis than its input". Their bounds identify an operation's *output family*;
//! they do not by themselves prove every const-geometry relationship. Each op
//! family generates complete rank-table implementations and compile-tests those
//! relationships in `tests/typed_ui.rs`.
//!
//! The per-op signatures are on the methods themselves; what follows is the
//! policy they share, which no individual signature states.
//!
//! # Static versus deferred checking
//!
//! Const-axis methods exist only where their associated output is implemented,
//! so an invalid axis is an ordinary trait error rather than a runtime one.
//! Where two const markers are *related* rather than equal — a matmul
//! contraction, conv channels, loss rows — the markers are independent generic
//! constants so that a static marker paired with [`DYN`] still compiles; the
//! relationship is then a body const assertion when both sides are known, and a
//! runtime shape error when either is `DYN`.
//!
//! # Escaping to runtime axes
//!
//! Every op whose axis can only be known at runtime has a `_dyn` spelling
//! (`transpose_dyn`, `narrow_dyn`, `squeeze_dyn`, `unsqueeze_dyn`, and the
//! reductions' `*_dyn`). Those that keep their rank return the same rank with
//! every marker erased to `DYN`; those whose *rank* is selected at runtime
//! (`squeeze_dyn`, `unsqueeze_dyn`) return a plain [`crate::Tensor`], because no
//! typed wrapper can name a rank that is not yet known.
//!
//! Rank-increasing methods and `stack` exist only through rank 7 — there is no
//! typed rank 9 to grow into.
//!
//! # Strictness
//!
//! There is **no implicit typed broadcasting** and no vector promotion or batch
//! broadcasting in `matmul`. `broadcast_to` is the only preparation step, and it
//! is explicit. `reshape` and `broadcast_to` are caller-targeted: the caller
//! names the output type and passes its dims, because `DYN` values are runtime
//! data the type cannot carry. Matching `DYN` markers still require equal actual
//! dimensions at runtime.
//!
//! Operator sugar (`Add`/`Sub`/`Mul`/`Div`) panics with the named method's
//! structured error; the named methods themselves never panic for a runtime
//! failure. Sugar is a convenience for code that has already established its
//! shapes, not an alternative error-handling path.
//!
//! # Element bounds
//!
//! [`NumericElement`](super::NumericElement) for arithmetic and
//! `sum`/`mean`/`min`/`max`/`argmin`/`argmax`/conv/pool;
//! [`FloatElement`](super::FloatElement) for the transcendentals, activations,
//! `var`/`std`/`softmax`/`log_softmax`, and every loss; plain [`Element`] for
//! comparisons, `masked_fill`, and `where_cond`. Masks are exactly `bool`.
//! `Bool` has no typed reduction.
//!
//! # Geometry preserved by each family
//!
//! Reductions come in three spellings per op — const-axis (drops the axis),
//! `_keepdim` (replaces it with 1), and `_all` (rank 0). `argmax`/`argmin`
//! retain placement and produce `i64`. `gather` returns the *index* geometry
//! with the *source* element type. Conv and pool are NCHW rank 4: conv retains
//! the batch and output-channel markers, pooling retains batch and channel, and
//! the computed spatial axes become `DYN`. A causal mask is square, its
//! `seq_len` checked against a static marker or retained when that marker is
//! `DYN`.

use super::device::validate_binding;
use super::sealed::TypedTensor as SealedTypedTensor;
use super::tensor::checked_wrap;
use super::{
    DYN, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7,
    Tensor8, TypedTensor,
};
use crate::{Element, Result, Tensor};
use std::sync::Arc;

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

/// Checks the operand's placement binding and hands back its runtime tensor.
pub(super) fn dynamic<'a, T: TypedTensor>(input: &'a T, op: &'static str) -> Result<&'a Tensor> {
    validate_binding::<T::Placement>(SealedTypedTensor::binding(input), op)?;
    Ok(SealedTypedTensor::dynamic(input))
}

/// Rewraps a runtime result under `input`'s placement binding.
pub(super) fn wrap<T: TypedTensor, O: TypedTensor>(
    input: &T,
    output: Tensor,
    op: &'static str,
) -> Result<O> {
    checked_wrap(output, Arc::clone(SealedTypedTensor::binding(input)), op)
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
