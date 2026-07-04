use std::marker::PhantomData;

use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{HasParameters, Layer, Module, ParameterRef, ParameterRefMut};

/// A type-checked stack of modules that consumes a fixed `Input` type.
///
/// `Sequential` pins the network input type up front (via [`sequential`]) and
/// carries `M`, the composed module built so far. Because the input type is
/// known, [`add_module`](Self::add_module) can require the next stage to
/// implement `Layer` for the *current* output type: appending a layer whose
/// input shape does not match — or a value that is not a module at all, such as
/// an integer — is a compile error at that call site, not later at `forward`.
///
/// The composite is itself a [`Module`] and forwards [`HasParameters`] to its
/// stages, so it plugs straight into the optimizer loop.
///
/// Parameter names are part of the persistence contract. A two-stage tuple uses
/// `0` and `1` as path segments, so `Sequential::new(Linear, Relu)` exposes
/// names such as `0.weight` and `0.bias`; additional `.add_module` calls nest
/// another tuple and preserve deterministic left-to-right order.
///
/// ```ignore
/// let model = Sequential::new(Linear::<784, 128>::zeros()?, Relu)
///     .add_module(Linear::<128, 10>::zeros()?);
/// let mut ctx = TrainContext::training(0);
/// let logits = model.forward(&images, &mut ctx)?; // pins the input type
/// ```
/// Builds a [`Sequential`] stack from two or more modules.
///
/// `seq![a, b, c]` expands to `Sequential::new(a, b).add_module(c)`, so the same
/// compile-time checks apply: each stage must accept the previous stage's output
/// type, and the input type is inferred from where the model is later applied.
/// It is purely sugar for the builder — there is no runtime cost and no separate
/// type. At least two modules are required, matching [`Sequential::new`].
///
/// ```ignore
/// let model = seq![
///     Linear::<784, 128>::zeros()?,
///     Relu,
///     Linear::<128, 10>::zeros()?,
/// ];
/// ```
#[macro_export]
macro_rules! seq {
    ($first:expr, $second:expr $(, $rest:expr)* $(,)?) => {
        $crate::Sequential::new($first, $second)
            $(.add_module($rest))*
    };
}

pub struct Sequential<M, In> {
    module: M,
    _in: PhantomData<fn() -> In>,
}

impl<M, In> Sequential<M, In> {
    /// Appends `next`, requiring it to accept the current output type.
    ///
    /// The bound `Next: Layer<<M as Layer<In>>::Output>` is what makes the
    /// stack type-safe: only a module compatible with the running output can be
    /// added, and the tracked output type advances to `Next`'s output.
    pub fn add_module<Next>(self, next: Next) -> Sequential<(M, Next), In>
    where
        M: Layer<In>,
        Next: Layer<<M as Layer<In>>::Output>,
    {
        Sequential {
            module: (self.module, next),
            _in: PhantomData,
        }
    }

    /// Returns the composed inner module, discarding the input-type marker.
    pub fn into_inner(self) -> M {
        self.module
    }
}

impl<First, Second, In> Sequential<(First, Second), In>
where
    First: Layer<In>,
    Second: Layer<<First as Layer<In>>::Output>,
{
    /// Starts a stack from its first two stages.
    ///
    /// The input type `In` is normally inferred from where the model is later
    /// applied (its `forward` call), so a call reads `Sequential::new(a, b)`. It
    /// can be pinned explicitly with `Sequential::<_, In>::new(a, b)` when the
    /// model is built without being used in the same scope. `second` is checked
    /// against `first`'s output; further stages are added with
    /// [`add_module`](Sequential::add_module).
    pub fn new(first: First, second: Second) -> Self {
        Self {
            module: (first, second),
            _in: PhantomData,
        }
    }
}

impl<M, In> Layer<In> for Sequential<M, In>
where
    M: Layer<In>,
{
    type Output = <M as Layer<In>>::Output;
}

impl<M, In, Context> Module<In, Context> for Sequential<M, In>
where
    M: Module<In, Context>,
{
    fn forward(&self, input: &In, ctx: &mut Context) -> Result<Self::Output> {
        self.module.forward(input, ctx)
    }
}

impl<M, In, E, B> HasParameters<E, B> for Sequential<M, In>
where
    E: FloatDType,
    B: Backend<E>,
    M: HasParameters<E, B>,
{
    fn visit_parameters<'a>(
        &'a self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRef<'a, E, B>),
    ) {
        self.module.visit_parameters(prefix, visit);
    }

    fn visit_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRefMut<'a, E, B>),
    ) {
        self.module.visit_parameters_mut(prefix, visit);
    }
}

/// Two stages composed front-to-back: the output of `.0` feeds `.1`.
///
/// This is the node type the stack is built from; a chain of `n` modules is an
/// `n`-deep nest of tuples. It is shape-checked through each stage's [`Layer`]
/// impl and forwards [`HasParameters`] to both stages.
impl<Input, A, B> Layer<Input> for (A, B)
where
    A: Layer<Input>,
    B: Layer<<A as Layer<Input>>::Output>,
{
    type Output = <B as Layer<<A as Layer<Input>>::Output>>::Output;
}

impl<Input, Context, A, B> Module<Input, Context> for (A, B)
where
    A: Module<Input, Context>,
    B: Module<<A as Layer<Input>>::Output, Context>,
{
    fn forward(&self, input: &Input, ctx: &mut Context) -> Result<Self::Output> {
        let hidden = self.0.forward(input, ctx)?;
        self.1.forward(&hidden, ctx)
    }
}

impl<A, B, E, Bk> HasParameters<E, Bk> for (A, B)
where
    E: FloatDType,
    Bk: Backend<E>,
    A: HasParameters<E, Bk>,
    B: HasParameters<E, Bk>,
{
    fn visit_parameters<'a>(
        &'a self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRef<'a, E, Bk>),
    ) {
        self.0
            .visit_parameters(&crate::nn::parameter_path(prefix, "0"), visit);
        self.1
            .visit_parameters(&crate::nn::parameter_path(prefix, "1"), visit);
    }

    fn visit_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRefMut<'a, E, Bk>),
    ) {
        self.0
            .visit_parameters_mut(&crate::nn::parameter_path(prefix, "0"), visit);
        self.1
            .visit_parameters_mut(&crate::nn::parameter_path(prefix, "1"), visit);
    }
}

#[cfg(test)]
mod tests {
    use super::Sequential;
    use crate::backend::Cpu;
    use crate::nn::{Dropout, HasParameters, Linear, Module, ParameterRef, Relu, TrainContext};
    use crate::shape::{C, D2, Sym};
    use crate::tensor::{Tensor, Tensor1D, Tensor2D};

    #[test]
    fn forward_accepts_dynamic_runtime_batch_sizes() {
        // A symbolic leading axis means the batch size is a runtime value, not
        // baked into the type. The model is built once and pins its input type
        // to `D2<Sym<Batch>, C<2>>` on the first `forward` call below.
        #[derive(Debug)]
        struct Batch;

        let model = Sequential::new(Linear::<2, 3>::zeros().unwrap(), Relu)
            .add_module(Linear::<3, 1>::zeros().unwrap());

        // The *same* model value handles different runtime batch sizes: only the
        // feature dim (`C<2>`) is fixed, the `Sym<Batch>` axis varies at runtime.
        let two =
            Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(vec![1.0, -2.0, 3.0, 4.0], [2, 2])
                .unwrap();
        let five =
            Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(vec![0.0; 10], [5, 2]).unwrap();

        let mut ctx = TrainContext::eval();
        let out_two = model.forward(&two, &mut ctx).unwrap();
        let out_five = model.forward(&five, &mut ctx).unwrap();

        assert_eq!(out_two.shape().dims(), &[2, 1]);
        assert_eq!(out_five.shape().dims(), &[5, 1]);
    }

    #[test]
    fn seq_macro_builds_an_equivalent_stack() {
        // `seq![..]` is sugar for `Sequential::new(..).add_module(..)`, so it is
        // the same type and behaves identically to the builder.
        let model = seq![
            Linear::<2, 3>::zeros().unwrap(),
            Relu,
            Linear::<3, 1>::zeros().unwrap(),
        ];

        let input = Tensor2D::<2, 2>::from_vec(vec![1.0, -2.0, 3.0, 4.0]).unwrap();
        let mut ctx = TrainContext::eval();
        let output = model.forward(&input, &mut ctx).unwrap();
        assert_eq!(output.shape().dims(), &[2, 1]);

        let mut params: Vec<ParameterRef<'_, f32, Cpu>> = Vec::new();
        model.parameters(&mut params);
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn add_module_chains_layers_and_collects_their_parameters() {
        // The input type is inferred from the `forward` call below.
        let model = Sequential::new(Linear::<2, 3>::zeros().unwrap(), Relu)
            .add_module(Linear::<3, 1>::zeros().unwrap());

        let input = Tensor2D::<2, 2>::from_vec(vec![1.0, -2.0, 3.0, 4.0]).unwrap();
        let mut ctx = TrainContext::eval();
        let output = model.forward(&input, &mut ctx).unwrap();
        assert_eq!(output.shape().dims(), &[2, 1]);

        // Two Linear stages contribute weight + bias each; Relu contributes none.
        let mut params: Vec<ParameterRef<'_, f32, Cpu>> = Vec::new();
        model.parameters(&mut params);
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn dropout_composes_in_sequential_and_uses_context_mode() {
        let model = seq![Relu, Dropout::new(0.5), Relu];
        let input = Tensor1D::<4>::ones().unwrap();

        let mut train_ctx = TrainContext::training(7);
        let train = model
            .forward(&input, &mut train_ctx)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(train.iter().all(|&value| value == 0.0 || value == 2.0));

        let mut eval_ctx = TrainContext::eval();
        let eval = model
            .forward(&input, &mut eval_ctx)
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(eval, vec![1.0; 4]);
    }
}
