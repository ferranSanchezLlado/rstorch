use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
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
    fn zero_grad(&self);
}

trait ParameterAccessMut<E, B>: ParameterAccess<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn data(&self) -> Result<Vec<E>>;
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
    fn data(&self) -> Result<Vec<E>> {
        self.tensor.to_vec()
    }

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

pub trait Layer<Input> {
    type Output;
}

pub trait Module<Input, Ctx>: Layer<Input> {
    fn forward(&self, input: &Input, ctx: &mut Ctx) -> Result<Self::Output>;
}

pub trait HasParameters<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn parameters<'a>(&'a self, out: &mut Vec<ParameterRef<'a, E, B>>);
    fn parameters_mut<'a>(&'a mut self, out: &mut Vec<ParameterRefMut<'a, E, B>>);
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

    pub fn data(&self) -> Result<Vec<E>> {
        self.inner.data()
    }

    pub fn grad(&self) -> Result<Option<Vec<E>>> {
        self.inner.grad()
    }

    pub fn set_grad(&mut self, grad: Vec<E>) -> Result<()> {
        self.inner.set_grad(grad)
    }

    pub fn set_data(&mut self, data: Vec<E>) -> Result<()> {
        self.inner.set_data(data)
    }

    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}
