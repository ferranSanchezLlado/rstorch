use super::{Mode, Module, ToDType, ToDevice, TypedParam, TypedVisitor, TypedVisitorMut};
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{DYN, DeviceCtx, FloatElement, Placement, TypedTensor};
use crate::{Error, Result, Rng, Shape, Tensor};
use std::collections::BTreeMap;
use std::sync::Arc;

const fn assert_configuration(embed: usize, heads: usize) {
    assert!(
        embed != DYN && embed > 0,
        "typed attention EMBED must be static and non-zero"
    );
    assert!(
        heads != DYN && heads > 0,
        "typed attention HEADS must be static and non-zero"
    );
    assert!(
        embed.is_multiple_of(heads),
        "typed attention HEADS must divide EMBED"
    );
}

const fn assert_input_width(markers: &[usize], embed: usize) {
    let width = markers[markers.len() - 1];
    assert!(
        width == DYN || width == embed,
        "typed attention input width is incompatible with EMBED"
    );
}

/// Input/output mapping for typed attention ranks two through seven.
pub trait AttentionInput<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement>:
    TypedTensor<Elem = E, Placement = P>
{
    /// Attention output with dynamic batch/query axes and static embedding width.
    type Output: TypedTensor<Elem = E, Placement = P>;
    /// Projected head representation.
    type Context: AttentionContext<EMBED, HEADS, E, P, Output = Self::Output>;
}

/// A projected `[..., HEADS, sequence, head_width]` attention representation.
pub trait AttentionContext<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement>:
    TypedTensor<Elem = E, Placement = P>
{
    /// Merged representation produced by output projection.
    type Output: TypedTensor<Elem = E, Placement = P>;
    /// Exact score geometry required after explicit mask broadcasting.
    type Mask: TypedTensor<Elem = bool, Placement = P>;
}

macro_rules! attention_rank {
    ($input:ident [$($dim:ident),+], $output:ident [$($out:expr),+],
     $context:ident [$($ctx:expr),+], $mask:ident [$($msk:expr),+]) => {
        impl<$(const $dim: usize,)+ const EMBED: usize, const HEADS: usize,
            E: FloatElement, P: Placement> AttentionInput<EMBED, HEADS, E, P>
            for crate::typed::$input<$($dim,)+ E, P>
        {
            type Output = crate::typed::$output<$($out,)+ E, P>;
            type Context = crate::typed::$context<$($ctx,)+ E, P>;
        }

        impl<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement>
            AttentionContext<EMBED, HEADS, E, P>
            for crate::typed::$context<$($ctx,)+ E, P>
        {
            type Output = crate::typed::$output<$($out,)+ E, P>;
            type Mask = crate::typed::$mask<$($msk,)+ bool, P>;
        }
    };
}

attention_rank!(Tensor2 [D0, D1], Tensor2 [DYN, EMBED], Tensor3 [HEADS, DYN, DYN], Tensor3 [HEADS, DYN, DYN]);
attention_rank!(Tensor3 [D0, D1, D2], Tensor3 [DYN, DYN, EMBED], Tensor4 [DYN, HEADS, DYN, DYN], Tensor4 [DYN, HEADS, DYN, DYN]);
attention_rank!(Tensor4 [D0, D1, D2, D3], Tensor4 [DYN, DYN, DYN, EMBED], Tensor5 [DYN, DYN, HEADS, DYN, DYN], Tensor5 [DYN, DYN, HEADS, DYN, DYN]);
attention_rank!(Tensor5 [D0, D1, D2, D3, D4], Tensor5 [DYN, DYN, DYN, DYN, EMBED], Tensor6 [DYN, DYN, DYN, HEADS, DYN, DYN], Tensor6 [DYN, DYN, DYN, HEADS, DYN, DYN]);
attention_rank!(Tensor6 [D0, D1, D2, D3, D4, D5], Tensor6 [DYN, DYN, DYN, DYN, DYN, EMBED], Tensor7 [DYN, DYN, DYN, DYN, HEADS, DYN, DYN], Tensor7 [DYN, DYN, DYN, DYN, HEADS, DYN, DYN]);
attention_rank!(Tensor7 [D0, D1, D2, D3, D4, D5, D6], Tensor7 [DYN, DYN, DYN, DYN, DYN, DYN, EMBED], Tensor8 [DYN, DYN, DYN, DYN, DYN, HEADS, DYN, DYN], Tensor8 [DYN, DYN, DYN, DYN, DYN, HEADS, DYN, DYN]);

fn same_binding<T: TypedTensor, U: TypedTensor>(
    left: &T,
    right: &U,
    op: &'static str,
) -> Result<()> {
    validate_binding::<T::Placement>(left.binding(), op)?;
    if !Arc::ptr_eq(left.binding(), right.binding()) {
        return Err(Error::InvalidArg {
            op,
            msg: "operands do not share the canonical placement binding".into(),
        });
    }
    Ok(())
}

fn check_explicit_mask(mask: &Tensor, q: &Tensor, k: &Tensor) -> Result<()> {
    let rank = q.rank();
    let mut expected = Vec::with_capacity(rank);
    for axis in 0..rank - 2 {
        let (left, right) = (q.dims()[axis], k.dims()[axis]);
        expected.push(if left == right {
            left
        } else if left == 1 {
            right
        } else if right == 1 {
            left
        } else {
            return Err(Error::ShapeMismatch {
                op: "typed::nn::scaled_dot_product_attention",
                lhs: q.shape().clone(),
                rhs: k.shape().clone(),
            });
        });
    }
    expected.push(q.dims()[rank - 2]);
    expected.push(k.dims()[rank - 2]);
    if mask.dims() != expected {
        return Err(Error::ShapeMismatch {
            op: "typed::nn::scaled_dot_product_attention",
            lhs: Shape::from(expected),
            rhs: mask.shape().clone(),
        });
    }
    Ok(())
}

/// Rejects a runtime width that a `DYN` trailing marker left unchecked.
///
/// The mirror of the runtime layer's own pre-check
/// (`nn::MultiHeadAttention::project`): without it the first failure is
/// `matmul`'s, which names an operation the caller never invoked and reports
/// the raw weight shape instead of the width this layer requires.
fn check_input_width(input: &Tensor, embed: usize) -> Result<()> {
    let dims = input.dims();
    let rank = dims.len();
    if dims[rank - 1] != embed {
        return Err(Error::ShapeMismatch {
            op: "typed::nn::MultiHeadAttention::project",
            lhs: Shape::from([dims[rank - 2], embed]),
            rhs: input.shape().clone(),
        });
    }
    Ok(())
}

fn context_shape_error(actual: &Tensor, embed: usize, heads: usize) -> Error {
    let mut expected = actual.dims().to_vec();
    let rank = expected.len();
    // Substituting the head axis is defensive only and currently always a
    // no-op — see the head-axis note in `check_context_geometry`. Only the
    // trailing axis can actually differ from what was expected.
    expected[rank - 3] = heads;
    expected[rank - 1] = embed / heads;
    Error::ShapeMismatch {
        op: "typed::nn::scaled_dot_product_attention",
        lhs: Shape::from(expected),
        rhs: actual.shape().clone(),
    }
}

fn check_context_geometry(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    embed: usize,
    heads: usize,
) -> Result<()> {
    let head_width = embed / heads;
    for context in [q, k, v] {
        let rank = context.rank();
        // The head-axis half of this comparison is defensive only: it cannot
        // fire today. `AttentionContext` is implemented exclusively for the
        // rank shapes whose head-axis *marker* is `HEADS` itself (see the
        // `attention_rank!` table above), and every constructor validates a
        // static marker against the runtime dimension — building, say, a
        // `Tensor4<DYN, 2, DYN, DYN>` from `[1, 1, 2, 2]` is rejected at
        // `from_vec` with `ShapeMismatch { lhs: [1, 1, 2, 2], rhs: [1, 2, 2, 2] }`.
        // It is kept because this is a soundness boundary (`head_width` below
        // is genuinely `DYN` and genuinely needs checking, and the two are one
        // geometric rule) and because the check costs nothing at runtime; a
        // reader must not mistake the branch for a live one.
        if context.dims()[rank - 3] != heads || context.dims()[rank - 1] != head_width {
            return Err(context_shape_error(context, embed, heads));
        }
    }
    if k.dims()[k.rank() - 2] != v.dims()[v.rank() - 2] {
        return Err(Error::ShapeMismatch {
            op: "typed::nn::scaled_dot_product_attention",
            lhs: k.shape().clone(),
            rhs: v.shape().clone(),
        });
    }

    for axis in 0..q.rank() - 2 {
        let mut dimension = q.dims()[axis];
        for context in [k, v] {
            let next = context.dims()[axis];
            if dimension == 1 {
                dimension = next;
            } else if next != 1 && next != dimension {
                return Err(Error::ShapeMismatch {
                    op: "typed::nn::scaled_dot_product_attention",
                    lhs: q.shape().clone(),
                    rhs: context.shape().clone(),
                });
            }
        }
    }
    Ok(())
}

/// Delegates scaled dot-product attention after requiring an explicitly broadcast mask.
pub fn scaled_dot_product_attention<
    const EMBED: usize,
    const HEADS: usize,
    E: FloatElement,
    P: Placement,
    C: AttentionContext<EMBED, HEADS, E, P>,
>(
    q: &C,
    k: &C,
    v: &C,
    mask: Option<&C::Mask>,
) -> Result<C> {
    const { assert_configuration(EMBED, HEADS) };
    same_binding(q, k, "typed::nn::scaled_dot_product_attention")?;
    same_binding(q, v, "typed::nn::scaled_dot_product_attention")?;
    check_context_geometry(q.dynamic(), k.dynamic(), v.dynamic(), EMBED, HEADS)?;
    if let Some(mask) = mask {
        same_binding(q, mask, "typed::nn::scaled_dot_product_attention")?;
        check_explicit_mask(mask.dynamic(), q.dynamic(), k.dynamic())?;
    }
    checked_wrap(
        crate::nn::scaled_dot_product_attention(
            q.dynamic(),
            k.dynamic(),
            v.dynamic(),
            mask.map(SealedTypedTensor::dynamic),
        )?,
        Arc::clone(q.binding()),
        "typed::nn::scaled_dot_product_attention",
    )
}

struct Projection<const EMBED: usize, E: FloatElement, P: Placement> {
    weight: TypedParam<crate::typed::Tensor2<EMBED, EMBED, E, P>>,
    bias: Option<TypedParam<crate::typed::Tensor1<EMBED, E, P>>>,
}

impl<const EMBED: usize, E: FloatElement, P: Placement> Projection<EMBED, E, P> {
    fn from_runtime_state(
        prefix: &str,
        state: &BTreeMap<String, Tensor>,
        ctx: &DeviceCtx<P>,
    ) -> Result<Self> {
        let missing = |path: &str| Error::InvalidArg {
            op: "typed::nn::MultiHeadAttention::new",
            msg: format!("runtime attention state omitted {path}"),
        };
        let weight_path = format!("{prefix}.weight");
        let weight = state
            .get(&weight_path)
            .ok_or_else(|| missing(&weight_path))?
            .to_dtype(E::DTYPE)?;
        let weight = TypedParam::new(crate::typed::Tensor2::try_from_dynamic(weight, ctx)?)?;
        let bias_path = format!("{prefix}.bias");
        let bias = state
            .get(&bias_path)
            .map(|value| {
                value
                    .to_dtype(E::DTYPE)
                    .and_then(|value| crate::typed::Tensor1::try_from_dynamic(value, ctx))
                    .and_then(TypedParam::new)
            })
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn apply<I>(&self, input: &I, mode: Mode) -> Result<Tensor>
    where
        I: TypedTensor<Elem = E, Placement = P>,
    {
        let weight = self.weight.get(mode)?;
        same_binding(input, &weight, "typed::nn::MultiHeadAttention::project")?;
        let output = input
            .dynamic()
            .matmul(&weight.dynamic().transpose(-2, -1)?)?;
        match &self.bias {
            Some(bias) => output.add(bias.get(mode)?.dynamic()),
            None => Ok(output),
        }
    }

    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        visitor.param("weight", &self.weight);
        if let Some(bias) = &self.bias {
            visitor.param("bias", bias);
        }
    }

    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        visitor.param("weight", &mut self.weight);
        if let Some(bias) = &mut self.bias {
            visitor.param("bias", bias);
        }
    }

    fn move_to<Q: Placement>(self, target: &DeviceCtx<Q>) -> Result<Projection<EMBED, E, Q>> {
        Ok(Projection {
            weight: self.weight.to_device(target)?,
            bias: self.bias.map(|value| value.to_device(target)).transpose()?,
        })
    }

    fn cast_to<F: FloatElement>(self) -> Result<Projection<EMBED, F, P>> {
        Ok(Projection {
            weight: self.weight.to_dtype()?,
            bias: self.bias.map(TypedParam::to_dtype).transpose()?,
        })
    }
}

/// Typed multi-head attention with static embedding and head configuration.
pub struct MultiHeadAttention<
    const EMBED: usize,
    const HEADS: usize,
    E: FloatElement = f32,
    P: Placement = crate::typed::Cpu,
> {
    q_proj: Projection<EMBED, E, P>,
    k_proj: Projection<EMBED, E, P>,
    v_proj: Projection<EMBED, E, P>,
    out_proj: Projection<EMBED, E, P>,
}

impl<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement>
    MultiHeadAttention<EMBED, HEADS, E, P>
{
    /// Constructs the four biased runtime-equivalent projections.
    ///
    /// A known invalid static configuration fails during monomorphization.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::DeviceCtx;
    /// use rstorch::typed::nn::MultiHeadAttention;
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let _ = MultiHeadAttention::<6, 4>::new(&ctx, &mut Rng::seed(1));
    /// ```
    pub fn new(ctx: &DeviceCtx<P>, rng: &mut Rng) -> Result<Self> {
        Self::build(true, ctx, rng)
    }

    /// Constructs the four projections without biases.
    pub fn new_without_bias(ctx: &DeviceCtx<P>, rng: &mut Rng) -> Result<Self> {
        Self::build(false, ctx, rng)
    }

    fn build(bias: bool, ctx: &DeviceCtx<P>, rng: &mut Rng) -> Result<Self> {
        const { assert_configuration(EMBED, HEADS) };
        validate_binding::<P>(ctx.binding(), "typed::nn::MultiHeadAttention::new")?;
        let runtime = if bias {
            crate::nn::MultiHeadAttention::new(EMBED, HEADS, &ctx.device(), rng)?
        } else {
            crate::nn::MultiHeadAttention::new_without_bias(EMBED, HEADS, &ctx.device(), rng)?
        };
        let state = crate::nn::state_dict(&runtime);
        Ok(Self {
            q_proj: Projection::from_runtime_state("q_proj", &state, ctx)?,
            k_proj: Projection::from_runtime_state("k_proj", &state, ctx)?,
            v_proj: Projection::from_runtime_state("v_proj", &state, ctx)?,
            out_proj: Projection::from_runtime_state("out_proj", &state, ctx)?,
        })
    }

    /// Returns the static embedding width.
    pub const fn embed_dim(&self) -> usize {
        EMBED
    }

    /// Returns the static head count.
    pub const fn num_heads(&self) -> usize {
        HEADS
    }

    /// Returns the runtime head width represented by a `DYN` output marker.
    pub const fn head_dim(&self) -> usize {
        EMBED / HEADS
    }

    fn project<I>(
        &self,
        projection: &Projection<EMBED, E, P>,
        input: &I,
        mode: Mode,
    ) -> Result<I::Context>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        const {
            assert_configuration(EMBED, HEADS);
            assert_input_width(I::MARKERS, EMBED);
        }
        check_input_width(input.dynamic(), EMBED)?;
        let projected = projection.apply(input, mode)?;
        let rank = projected.rank();
        let mut split = projected.dims()[..rank - 1].to_vec();
        split.push(HEADS);
        split.push(self.head_dim());
        let split = projected.reshape(split)?;
        let mut axes = (0..rank as isize + 1).collect::<Vec<_>>();
        axes.swap(rank - 2, rank - 1);
        checked_wrap(
            split.permute(&axes)?,
            Arc::clone(input.binding()),
            "typed::nn::MultiHeadAttention::project",
        )
    }

    /// Projects and splits queries into heads.
    pub fn project_query<I>(&self, input: &I, mode: Mode) -> Result<I::Context>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        self.project(&self.q_proj, input, mode)
    }

    /// Projects and splits keys into heads.
    pub fn project_keys<I>(&self, input: &I, mode: Mode) -> Result<I::Context>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        self.project(&self.k_proj, input, mode)
    }

    /// Projects and splits values into heads.
    pub fn project_values<I>(&self, input: &I, mode: Mode) -> Result<I::Context>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        self.project(&self.v_proj, input, mode)
    }

    /// Projects both halves stored by a KV cache.
    pub fn project_keys_values<I>(&self, input: &I, mode: Mode) -> Result<(I::Context, I::Context)>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        Ok((
            self.project_keys(input, mode)?,
            self.project_values(input, mode)?,
        ))
    }

    /// Merges projected heads and applies the output projection.
    pub fn project_output<C>(&self, context: &C, mode: Mode) -> Result<C::Output>
    where
        C: AttentionContext<EMBED, HEADS, E, P>,
    {
        const { assert_configuration(EMBED, HEADS) };
        const OP: &str = "typed::nn::MultiHeadAttention::project_output";
        // Validated before any axis motion, as every sibling entry point in
        // this area does (`project` through `Projection::apply`,
        // `scaled_dot_product_attention` through `same_binding`, and the
        // normalization layers directly). The reshape below would reject a
        // forged binding too, but only after the metadata work.
        validate_binding::<P>(context.binding(), OP)?;
        let rank = context.dynamic().rank();
        let dims = context.dynamic().dims();
        // The head-axis half is defensive only and cannot fire — `C`'s head
        // axis marker *is* `HEADS`, so it is validated at construction (see
        // `check_context_geometry`). `head_dim` is the `DYN` half that matters.
        if dims[rank - 3] != HEADS || dims[rank - 1] != self.head_dim() {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: Shape::from([HEADS, dims[rank - 2], self.head_dim()]),
                rhs: context.dynamic().shape().clone(),
            });
        }
        let mut axes = (0..rank as isize).collect::<Vec<_>>();
        axes.swap(rank - 3, rank - 2);
        let merged = context.dynamic().permute(&axes)?;
        let mut shape = dims[..rank - 3].to_vec();
        shape.push(dims[rank - 2]);
        shape.push(EMBED);
        let merged =
            checked_wrap::<C::Output>(merged.reshape(shape)?, Arc::clone(context.binding()), OP)?;
        let output = self.out_proj.apply(&merged, mode)?;
        checked_wrap(output, Arc::clone(context.binding()), OP)
    }

    /// Runs self-attention with an optional already-broadcast mask.
    ///
    /// Mask dtype and placement are compile-time contracts.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DYN, DeviceCtx, Tensor3, Tensor4};
    /// use rstorch::typed::nn::{Mode, MultiHeadAttention};
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let input = Tensor3::<DYN, DYN, 4>::from_vec(vec![0.0; 4], [1, 1, 4], &ctx).unwrap();
    /// let float_mask = Tensor4::<DYN, 2, DYN, DYN>::from_vec(
    ///     vec![0.0; 2], [1, 2, 1, 1], &ctx,
    /// ).unwrap();
    /// let _ = attention.attend(&input, Some(&float_mask), Mode::EVAL);
    /// ```
    ///
    /// ```compile_fail
    /// use rstorch::{Device, Rng};
    /// use rstorch::typed::{DYN, DeviceCtx, Placement, Tensor3, Tensor4};
    /// use rstorch::typed::nn::{Mode, MultiHeadAttention};
    /// struct MaskDevice;
    /// impl Placement for MaskDevice {}
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let mask_ctx = DeviceCtx::<MaskDevice>::bind(Device::Cpu).unwrap();
    /// let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let input = Tensor3::<DYN, DYN, 4>::from_vec(vec![0.0; 4], [1, 1, 4], &ctx).unwrap();
    /// let mask = Tensor4::<DYN, 2, DYN, DYN, bool, MaskDevice>::from_vec(
    ///     vec![false; 2], [1, 2, 1, 1], &mask_ctx,
    /// ).unwrap();
    /// let _ = attention.attend(&input, Some(&mask), Mode::EVAL);
    /// ```
    ///
    /// Rank eight cannot be head-split within the typed rank ceiling.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DeviceCtx, Tensor8};
    /// use rstorch::typed::nn::{Mode, MultiHeadAttention};
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let input = Tensor8::<1, 1, 1, 1, 1, 1, 1, 4>::from_vec(
    ///     vec![0.0; 4], [1, 1, 1, 1, 1, 1, 1, 4], &ctx,
    /// ).unwrap();
    /// let _ = attention.attend(&input, None, Mode::EVAL);
    /// ```
    pub fn attend<I>(
        &self,
        input: &I,
        mask: Option<&<I::Context as AttentionContext<EMBED, HEADS, E, P>>::Mask>,
        mode: Mode,
    ) -> Result<I::Output>
    where
        I: AttentionInput<EMBED, HEADS, E, P>,
    {
        self.attend_to(input, input, mask, mode)
    }

    /// Runs cross-attention with independent runtime query and KV lengths.
    ///
    /// The two sequence lengths and the batch axes are runtime relations, so
    /// `query` may be `[b, 1, EMBED]` against a `[b, kv, EMBED]` prefix and the
    /// batch axes need only broadcast.
    ///
    /// # A deliberate strict difference
    ///
    /// The **ranks** must be equal: `K::Context == Q::Context` is a type
    /// requirement, so a rank-3 query cannot attend over rank-2 keys/values.
    /// The dynamic layer *accepts* that call — `nn::MultiHeadAttention::<4, 2>`
    /// with `query [2, 3, 4]` and `keys_values [5, 4]` returns `Ok([2, 3, 4])`,
    /// because the head-split contexts broadcast inside `matmul`. Typed
    /// attention refuses it at compile time rather than silently choosing which
    /// rank the mask and the output should follow; erase one side with
    /// [`as_dynamic`](crate::typed::Tensor3::as_dynamic) if the broadcast is
    /// really wanted.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DYN, DeviceCtx, Tensor2, Tensor3};
    /// use rstorch::typed::nn::{Mode, MultiHeadAttention};
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let query = Tensor3::<DYN, DYN, 4>::from_vec(vec![0.0; 24], [2, 3, 4], &ctx).unwrap();
    /// let keys_values = Tensor2::<DYN, 4>::from_vec(vec![0.0; 20], [5, 4], &ctx).unwrap();
    /// let _ = attention.attend_to(&query, &keys_values, None, Mode::EVAL);
    /// ```
    pub fn attend_to<Q, K>(
        &self,
        query: &Q,
        keys_values: &K,
        mask: Option<&<Q::Context as AttentionContext<EMBED, HEADS, E, P>>::Mask>,
        mode: Mode,
    ) -> Result<Q::Output>
    where
        Q: AttentionInput<EMBED, HEADS, E, P>,
        K: AttentionInput<EMBED, HEADS, E, P, Context = Q::Context>,
    {
        let q = self.project_query(query, mode)?;
        let k = self.project_keys(keys_values, mode)?;
        let v = self.project_values(keys_values, mode)?;
        let context = scaled_dot_product_attention::<EMBED, HEADS, E, P, _>(&q, &k, &v, mask)?;
        self.project_output(&context, mode)
    }
}

impl<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement> Module
    for MultiHeadAttention<EMBED, HEADS, E, P>
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        for (name, projection) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("out_proj", &self.out_proj),
        ] {
            let saved = ProjectionModule(projection);
            visitor.module(name, &saved);
        }
    }

    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        visitor.module("q_proj", &mut ProjectionModuleMut(&mut self.q_proj));
        visitor.module("k_proj", &mut ProjectionModuleMut(&mut self.k_proj));
        visitor.module("v_proj", &mut ProjectionModuleMut(&mut self.v_proj));
        visitor.module("out_proj", &mut ProjectionModuleMut(&mut self.out_proj));
    }
}

struct ProjectionModule<'a, const EMBED: usize, E: FloatElement, P: Placement>(
    &'a Projection<EMBED, E, P>,
);
impl<const EMBED: usize, E: FloatElement, P: Placement> Module
    for ProjectionModule<'_, EMBED, E, P>
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        self.0.visit(visitor);
    }
    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}
struct ProjectionModuleMut<'a, const EMBED: usize, E: FloatElement, P: Placement>(
    &'a mut Projection<EMBED, E, P>,
);
impl<const EMBED: usize, E: FloatElement, P: Placement> Module
    for ProjectionModuleMut<'_, EMBED, E, P>
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        self.0.visit(visitor);
    }
    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        self.0.visit_mut(visitor);
    }
}

impl<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement, Q: Placement>
    ToDevice<Q> for MultiHeadAttention<EMBED, HEADS, E, P>
{
    type Output = MultiHeadAttention<EMBED, HEADS, E, Q>;
    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(MultiHeadAttention {
            q_proj: self.q_proj.move_to(target)?,
            k_proj: self.k_proj.move_to(target)?,
            v_proj: self.v_proj.move_to(target)?,
            out_proj: self.out_proj.move_to(target)?,
        })
    }
}

impl<const EMBED: usize, const HEADS: usize, E: FloatElement, P: Placement, F: FloatElement>
    ToDType<F> for MultiHeadAttention<EMBED, HEADS, E, P>
{
    type Output = MultiHeadAttention<EMBED, HEADS, F, P>;
    fn to_dtype(self) -> Result<Self::Output> {
        Ok(MultiHeadAttention {
            q_proj: self.q_proj.cast_to()?,
            k_proj: self.k_proj.cast_to()?,
            v_proj: self.v_proj.cast_to()?,
            out_proj: self.out_proj.cast_to()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, Tensor3, Tensor4, TypedGradsExt};

    fn input(ctx: &DeviceCtx<Cpu>, batch: usize, sequence: usize) -> Tensor3<DYN, DYN, 4> {
        Tensor3::from_vec(vec![0.1; batch * sequence * 4], [batch, sequence, 4], ctx).unwrap()
    }

    /// The sibling of `norm::tests::forged_noncanonical_input_binding_is_rejected_before_arithmetic`
    /// for the one entry point that takes an already-projected context.
    ///
    /// The context is forged *and* geometrically wrong (head width 5, not
    /// `EMBED / HEADS == 2`), which is what makes the ordering observable: the
    /// binding must be rejected first. Were the `validate_binding` call
    /// removed, the geometry check would fire instead and this would be a
    /// `ShapeMismatch` — and a plain forged-but-well-shaped context would not
    /// catch that, because the trailing `checked_wrap` reports the same
    /// `InvalidArg` under the same op.
    #[test]
    fn forged_context_binding_is_rejected_before_the_geometry_check() {
        use crate::typed::sealed::DeviceBinding;
        use crate::{Device, Tensor};

        let attention =
            MultiHeadAttention::<4, 2>::new(&DeviceCtx::<Cpu>::cpu().unwrap(), &mut Rng::seed(11))
                .unwrap();
        let dynamic = Tensor::from_vec(vec![0.1f32; 2 * 3 * 5], [2, 3, 5], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let context =
            <Tensor3<2, DYN, DYN> as SealedTypedTensor>::trusted_from_validated(dynamic, forged);
        assert!(matches!(
            attention.project_output(&context, Mode::EVAL),
            Err(Error::InvalidArg {
                op: "typed::nn::MultiHeadAttention::project_output",
                ..
            })
        ));
    }

    #[test]
    fn self_cross_masks_cache_gradients_mode_and_paths_work() {
        let ctx = DeviceCtx::cpu().unwrap();
        let attention = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(7)).unwrap();
        let query = input(&ctx, 2, 2);
        let kv = input(&ctx, 2, 5);
        let mask = Tensor4::<DYN, 2, DYN, DYN, bool>::from_vec(
            vec![false; 2 * 2 * 2 * 5],
            [2, 2, 2, 5],
            &ctx,
        )
        .unwrap();
        assert_eq!(
            attention
                .attend_to(&query, &kv, Some(&mask), Mode::EVAL)
                .unwrap()
                .dims(),
            [2, 2, 4]
        );
        let traced_query = Tensor3::<DYN, DYN, 4>::from_vec(
            (0..16).map(|index| index as f32 * 0.03 - 0.2).collect(),
            [2, 2, 4],
            &ctx,
        )
        .unwrap()
        .traced()
        .unwrap();
        let traced_kv = Tensor3::<DYN, DYN, 4>::from_vec(
            (0..40).map(|index| index as f32 * -0.02 + 0.3).collect(),
            [2, 5, 4],
            &ctx,
        )
        .unwrap()
        .traced()
        .unwrap();
        let cross_loss = attention
            .attend_to(&traced_query, &traced_kv, Some(&mask), Mode::TRAIN)
            .unwrap()
            .as_dynamic()
            .sum_all()
            .unwrap();
        let cross_grads = cross_loss.backward().unwrap();
        assert!(
            cross_grads
                .wrt_typed_input(&traced_query)
                .unwrap()
                .to_vec()
                .unwrap()
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            cross_grads
                .wrt_typed_input(&traced_kv)
                .unwrap()
                .to_vec()
                .unwrap()
                .iter()
                .any(|value| *value != 0.0)
        );
        assert_eq!(
            attention.attend(&query, None, Mode::EVAL).unwrap().dims(),
            [2, 2, 4]
        );

        let first = input(&ctx, 2, 3);
        let step = input(&ctx, 2, 1);
        let (old_k, old_v) = attention.project_keys_values(&first, Mode::EVAL).unwrap();
        let (new_k, new_v) = attention.project_keys_values(&step, Mode::EVAL).unwrap();
        let cached_k = Tensor4::cat::<2>(&[&old_k, &new_k]).unwrap();
        let cached_v = Tensor4::cat::<2>(&[&old_v, &new_v]).unwrap();
        let q = attention.project_query(&step, Mode::EVAL).unwrap();
        let context =
            scaled_dot_product_attention::<4, 2, f32, Cpu, _>(&q, &cached_k, &cached_v, None)
                .unwrap();
        assert_eq!(
            attention
                .project_output(&context, Mode::EVAL)
                .unwrap()
                .dims(),
            [2, 1, 4]
        );

        let loss = attention
            .attend(&query, None, Mode::TRAIN)
            .unwrap()
            .as_dynamic()
            .sum_all()
            .unwrap();
        assert_eq!(loss.backward().unwrap().len(), 8);
        assert!(
            attention
                .attend(&query, None, Mode::EVAL)
                .unwrap()
                .as_dynamic()
                .backward()
                .is_err()
        );
        assert_eq!(
            super::super::state_dict(&attention)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            [
                "k_proj.bias",
                "k_proj.weight",
                "out_proj.bias",
                "out_proj.weight",
                "q_proj.bias",
                "q_proj.weight",
                "v_proj.bias",
                "v_proj.weight"
            ]
        );
    }

    #[test]
    fn seeded_values_and_numerics_match_runtime_attention() {
        let ctx = DeviceCtx::cpu().unwrap();
        let typed = MultiHeadAttention::<4, 2>::new(&ctx, &mut Rng::seed(23)).unwrap();
        let runtime =
            crate::nn::MultiHeadAttention::new(4, 2, &ctx.device(), &mut Rng::seed(23)).unwrap();
        let sample = input(&ctx, 2, 3);
        let actual = typed.attend(&sample, None, Mode::EVAL).unwrap();
        let expected = runtime
            .attend(sample.as_dynamic(), None, Mode::EVAL)
            .unwrap();
        assert_eq!(
            actual.as_dynamic().to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
        let query = input(&ctx, 2, 2);
        let kv = input(&ctx, 2, 5);
        assert_eq!(
            typed
                .attend_to(&query, &kv, None, Mode::EVAL)
                .unwrap()
                .to_vec()
                .unwrap(),
            runtime
                .attend_to(query.as_dynamic(), kv.as_dynamic(), None, Mode::EVAL)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            super::super::state_dict(&typed)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            crate::nn::state_dict(&runtime)
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn explicit_mask_and_dynamic_width_errors_are_early() {
        let ctx = DeviceCtx::cpu().unwrap();
        let attention =
            MultiHeadAttention::<4, 2>::new_without_bias(&ctx, &mut Rng::seed(1)).unwrap();
        let input = input(&ctx, 1, 2);
        let unbroadcast =
            Tensor4::<DYN, 2, DYN, DYN, bool>::from_vec(vec![false; 4], [1, 2, 1, 2], &ctx)
                .unwrap();
        // `Display` carries the op and both shapes, so comparing the rendered
        // message pins the whole payload — a bare `ShapeMismatch { .. }` match
        // is what let a `matmul`-named width rejection survive review.
        assert_eq!(
            attention
                .attend(&input, Some(&unbroadcast), Mode::EVAL)
                .unwrap_err()
                .to_string(),
            "typed::nn::scaled_dot_product_attention: shape mismatch: \
             lhs [1, 2, 2, 2] vs rhs [1, 2, 1, 2]"
        );

        // A `DYN` trailing marker defers the width relation to runtime, so the
        // rejection must name this layer rather than the `matmul` it delegates
        // to, and must report the width the layer requires.
        let wrong = Tensor3::<DYN, DYN, DYN>::from_vec(vec![0.0; 6], [1, 2, 3], &ctx).unwrap();
        assert_eq!(
            attention
                .attend(&wrong, None, Mode::EVAL)
                .unwrap_err()
                .to_string(),
            "typed::nn::MultiHeadAttention::project: shape mismatch: lhs [2, 4] vs rhs [1, 2, 3]"
        );
        assert_eq!(super::super::state_dict(&attention).unwrap().len(), 4);
    }

    #[test]
    fn reported_configuration_matches_the_runtime_layer_it_was_built_from() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let typed = MultiHeadAttention::<6, 3>::new(&ctx, &mut Rng::seed(13)).unwrap();
        let runtime =
            crate::nn::MultiHeadAttention::new(6, 3, &ctx.device(), &mut Rng::seed(13)).unwrap();
        assert_eq!(
            (typed.embed_dim(), typed.num_heads(), typed.head_dim()),
            (runtime.embed_dim(), runtime.num_heads(), runtime.head_dim())
        );
        assert_eq!(
            (typed.embed_dim(), typed.num_heads(), typed.head_dim()),
            (6, 3, 2)
        );

        // One head is the degenerate configuration `head_dim == EMBED`.
        let single =
            MultiHeadAttention::<4, 1>::new_without_bias(&ctx, &mut Rng::seed(13)).unwrap();
        assert_eq!(
            (single.embed_dim(), single.num_heads(), single.head_dim()),
            (4, 1, 4)
        );
    }

    #[test]
    fn sdpa_preflights_every_dynamic_context_axis_before_delegation() {
        type Context = Tensor4<DYN, 2, DYN, DYN>;
        let ctx = DeviceCtx::cpu().unwrap();
        let context = |shape: [usize; 4]| {
            Context::from_vec(vec![0.1; shape.iter().product()], shape, &ctx).unwrap()
        };
        let q = context([1, 2, 2, 2]);
        let k = context([1, 2, 3, 2]);
        let v = context([1, 2, 3, 2]);
        let expected = crate::nn::scaled_dot_product_attention(
            q.as_dynamic(),
            k.as_dynamic(),
            v.as_dynamic(),
            None,
        )
        .unwrap();
        assert_eq!(
            scaled_dot_product_attention::<4, 2, f32, Cpu, _>(&q, &k, &v, None)
                .unwrap()
                .to_vec()
                .unwrap(),
            expected.to_vec::<f32>().unwrap()
        );

        for (bad_q, bad_k, bad_v) in [
            (context([1, 2, 2, 1]), k.clone(), v.clone()),
            (q.clone(), context([1, 2, 3, 1]), v.clone()),
            (q.clone(), k.clone(), context([1, 2, 3, 1])),
            (q.clone(), k.clone(), context([1, 2, 4, 2])),
            (context([2, 2, 2, 2]), context([3, 2, 3, 2]), v.clone()),
        ] {
            assert!(matches!(
                scaled_dot_product_attention::<4, 2, f32, Cpu, _>(&bad_q, &bad_k, &bad_v, None),
                Err(Error::ShapeMismatch {
                    op: "typed::nn::scaled_dot_product_attention",
                    ..
                })
            ));
        }
    }

    #[test]
    fn consuming_movement_preserves_attention_state_and_contract() {
        struct Main;
        impl Placement for Main {}

        let source = DeviceCtx::<Main>::bind(crate::Device::Cpu).unwrap();
        let target = DeviceCtx::<Cpu>::cpu().unwrap();
        let attention =
            MultiHeadAttention::<4, 2, f32, Main>::new(&source, &mut Rng::seed(31)).unwrap();
        let paths = super::super::state_dict(&attention)
            .unwrap()
            .paths()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let attention = ToDevice::to_device(attention, &target).unwrap();
        assert_eq!(
            super::super::state_dict(&attention)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            paths.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(
            attention
                .attend(&input(&target, 1, 2), None, Mode::EVAL)
                .unwrap()
                .dims(),
            [1, 2, 4]
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn reduced_precision_retyping_runs_where_the_backend_supports_it() {
        use crate::typed::Metal;
        use half::f16;

        let ctx = DeviceCtx::<Metal<0>>::bind(crate::Device::Metal(0)).unwrap();
        let attention =
            MultiHeadAttention::<4, 2, f32, Metal<0>>::new(&ctx, &mut Rng::seed(37)).unwrap();
        let attention: MultiHeadAttention<4, 2, f16, Metal<0>> =
            ToDType::to_dtype(attention).unwrap();
        let input = Tensor3::<DYN, DYN, 4, f16, Metal<0>>::from_vec(
            vec![f16::from_f32(0.1); 8],
            [1, 2, 4],
            &ctx,
        )
        .unwrap();
        assert_eq!(
            attention.attend(&input, None, Mode::EVAL).unwrap().dims(),
            [1, 2, 4]
        );
    }
}
