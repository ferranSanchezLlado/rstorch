use crate::backend::{Backend, Cpu};
use crate::dtype::{DTypeId, FloatDType};
use crate::error::Result;
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterId(u64);

pub struct Parameter<S, E = f32, B = Cpu>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    id: ParameterId,
    tensor: Tensor<S, E, B>,
}

impl<S, E, B> Parameter<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(tensor: Tensor<S, E, B>) -> Self {
        tensor.set_requires_grad(true);
        Self {
            id: ParameterId(NEXT_PARAMETER_ID.fetch_add(1, Ordering::Relaxed)),
            tensor,
        }
    }

    pub fn id(&self) -> ParameterId {
        self.id
    }

    pub fn tensor(&self) -> &Tensor<S, E, B> {
        &self.tensor
    }

    pub fn grad(&self) -> Option<Tensor<S, E, B>> {
        self.tensor.grad()
    }

    pub fn zero_grad(&self) {
        self.tensor.zero_grad();
    }

    pub(crate) fn as_ref(&self) -> ParameterRef<'_, E, B> {
        ParameterRef { inner: self }
    }

    pub(crate) fn as_mut(&mut self) -> ParameterRefMut<'_, E, B> {
        ParameterRefMut { inner: self }
    }
}

trait ParameterAccess<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn id(&self) -> ParameterId;
    fn dtype(&self) -> DTypeId;
    fn dims(&self) -> Vec<usize>;
    fn data(&self) -> Result<Vec<E>>;
    fn zero_grad(&self);
}

trait ParameterAccessMut<E, B>: ParameterAccess<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn grad(&self) -> Result<Option<Vec<E>>>;
    fn set_grad(&mut self, data: Vec<E>) -> Result<()>;
    fn set_data(&mut self, data: Vec<E>) -> Result<()>;
}

impl<S, E, B> ParameterAccess<E, B> for Parameter<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn id(&self) -> ParameterId {
        self.id()
    }

    fn dtype(&self) -> DTypeId {
        self.tensor.dtype()
    }

    fn dims(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    fn data(&self) -> Result<Vec<E>> {
        self.tensor.to_vec()
    }

    fn zero_grad(&self) {
        self.zero_grad();
    }
}

impl<S, E, B> ParameterAccessMut<E, B> for Parameter<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn grad(&self) -> Result<Option<Vec<E>>> {
        self.grad().map(|grad| grad.to_vec()).transpose()
    }

    fn set_grad(&mut self, grad: Vec<E>) -> Result<()> {
        self.tensor.set_grad_data(grad)
    }

    fn set_data(&mut self, data: Vec<E>) -> Result<()> {
        self.tensor.replace_data(data)
    }
}

pub trait Layer<Input: ?Sized> {
    type Output;
}

pub trait Module<Input: ?Sized, Context>: Layer<Input> {
    fn forward(&self, input: &Input, ctx: &mut Context) -> Result<Self::Output>;
}

pub trait HasParameters<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn visit_parameters<'a>(
        &'a self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRef<'a, E, B>),
    );

    fn visit_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRefMut<'a, E, B>),
    );

    fn parameters<'a>(&'a self, out: &mut Vec<ParameterRef<'a, E, B>>) {
        self.visit_parameters("", &mut |_, param| out.push(param));
    }

    fn parameters_mut<'a>(&'a mut self, out: &mut Vec<ParameterRefMut<'a, E, B>>) {
        self.visit_parameters_mut("", &mut |_, param| out.push(param));
    }
}

pub(crate) fn parameter_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_owned()
    } else {
        format!("{prefix}.{segment}")
    }
}

pub struct ParameterRef<'a, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    inner: &'a dyn ParameterAccess<E, B>,
}

impl<E, B> ParameterRef<'_, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn id(&self) -> ParameterId {
        self.inner.id()
    }

    pub fn dtype(&self) -> DTypeId {
        self.inner.dtype()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.inner.dims()
    }

    pub fn data(&self) -> Result<Vec<E>> {
        self.inner.data()
    }

    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}

pub struct ParameterRefMut<'a, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    inner: &'a mut dyn ParameterAccessMut<E, B>,
}

impl<E, B> ParameterRefMut<'_, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn id(&self) -> ParameterId {
        self.inner.id()
    }

    pub fn dtype(&self) -> DTypeId {
        self.inner.dtype()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.inner.dims()
    }

    /// Returns parameter data through the current host round-trip path.
    ///
    /// This data-access surface is intentionally unstable for external
    /// optimizer implementors until backend parity settles the device-resident
    /// optimizer kernel set. Built-in CPU optimizers may continue using this
    /// path in the interim.
    pub fn data(&self) -> Result<Vec<E>> {
        self.inner.data()
    }

    /// Returns gradient data through the current host round-trip path.
    ///
    /// See [`Self::data`] for the optimizer data-access stability note.
    pub fn grad(&self) -> Result<Option<Vec<E>>> {
        self.inner.grad()
    }

    /// Sets gradient data through the current host round-trip path.
    ///
    /// See [`Self::data`] for the optimizer data-access stability note.
    pub fn set_grad(&mut self, grad: Vec<E>) -> Result<()> {
        self.inner.set_grad(grad)
    }

    /// Sets parameter data through the current host round-trip path.
    ///
    /// See [`Self::data`] for the optimizer data-access stability note.
    pub fn set_data(&mut self, data: Vec<E>) -> Result<()> {
        self.inner.set_data(data)
    }

    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}
