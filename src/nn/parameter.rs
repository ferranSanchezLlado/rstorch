use crate::backend::{Backend, Cpu};
use crate::dtype::{DTypeId, FloatDType};
use crate::error::Result;
use crate::shape::{ShapeSpec, StaticShape};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

/// Host closure applied to a parameter's `(data, grad)` slices, returning the
/// updated data buffer. Used by the optimizer path to mutate parameter data
/// without a `data()`/`set_data()` host round trip.
type UpdateData<'a, E> = &'a mut dyn FnMut(&[E], &[E]) -> Vec<E>;

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
    fn update_data(&mut self, update: UpdateData<'_, E>) -> Result<bool>;
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

    fn update_data(&mut self, update: UpdateData<'_, E>) -> Result<bool> {
        let Some(grad) = self.grad() else {
            return Ok(false);
        };
        let data = self.tensor.host_values()?;
        let grad = grad.host_values()?;
        let next = update(&data, &grad);
        self.tensor.replace_data(next)?;
        Ok(true)
    }

    fn set_grad(&mut self, grad: Vec<E>) -> Result<()> {
        self.tensor.set_grad_data(grad)
    }

    fn set_data(&mut self, data: Vec<E>) -> Result<()> {
        self.tensor.replace_data(data)
    }
}

/// Non-trainable buffer with interior mutability (for running statistics).
pub struct Buffer<S, E = f32, B = Cpu>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    tensor: std::sync::Mutex<Tensor<S, E, B>>,
}

impl<S, E, B> Buffer<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(tensor: Tensor<S, E, B>) -> Self {
        tensor.set_requires_grad(false);
        Self {
            tensor: std::sync::Mutex::new(tensor),
        }
    }

    pub fn tensor(&self) -> std::sync::MutexGuard<'_, Tensor<S, E, B>> {
        self.tensor.lock().expect("buffer mutex poisoned")
    }

    pub(crate) fn as_ref(&self) -> BufferRef<'_, E, B>
    where
        S: StaticShape,
    {
        BufferRef { inner: self }
    }
}

trait BufferAccess<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn dtype(&self) -> DTypeId;
    fn dims(&self) -> Vec<usize>;
    fn data(&self) -> Result<Vec<E>>;
    fn set_data(&self, data: Vec<E>) -> Result<()>;
}

impl<S, E, B> BufferAccess<E, B> for Buffer<S, E, B>
where
    S: StaticShape,
    E: FloatDType,
    B: Backend<E>,
{
    fn dtype(&self) -> DTypeId {
        E::ID
    }

    fn dims(&self) -> Vec<usize> {
        self.tensor().shape().dims().to_vec()
    }

    fn data(&self) -> Result<Vec<E>> {
        self.tensor().to_vec()
    }

    fn set_data(&self, data: Vec<E>) -> Result<()> {
        let mut guard = self.tensor.lock().expect("buffer mutex poisoned");
        guard.replace_data(data)?;
        guard.set_requires_grad(false);
        Ok(())
    }
}

pub struct BufferRef<'a, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    inner: &'a dyn BufferAccess<E, B>,
}

impl<E, B> BufferRef<'_, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn dtype(&self) -> DTypeId {
        self.inner.dtype()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.inner.dims()
    }

    pub fn data(&self) -> Result<Vec<E>> {
        self.inner.data()
    }

    pub fn set_data(&self, data: Vec<E>) -> Result<()> {
        self.inner.set_data(data)
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

    fn visit_buffers<'a>(&'a self, prefix: &str, visit: &mut dyn FnMut(&str, BufferRef<'a, E, B>)) {
        let _ = prefix;
        let _ = &mut *visit;
    }

    fn buffers<'a>(&'a self, out: &mut Vec<BufferRef<'a, E, B>>) {
        self.visit_buffers("", &mut |_, buf| out.push(buf));
    }
}

pub(crate) fn parameter_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_owned()
    } else {
        format!("{prefix}.{segment}")
    }
}

macro_rules! has_parameters {
    // ── NEW arm: includes buffers section ────────────────────────────────────
    (
        impl[$($generics:tt)*] $ty:ty
        where { $($where_clause:tt)* }
        {
            params { $($params:tt)* }
            children { $($children:tt)* }
            transparent_children { $($transparent:tt)* }
            buffers { $($buffers:tt)* }
        }
    ) => {
        impl<$($generics)*> $crate::nn::HasParameters<E, B> for $ty
        where
            E: $crate::dtype::FloatDType,
            B: $crate::backend::Backend<E>,
            $($where_clause)*
        {
            fn visit_parameters<'a>(
                &'a self,
                prefix: &str,
                visit: &mut dyn FnMut(&str, $crate::nn::ParameterRef<'a, E, B>),
            ) {
                let _ = prefix;
                let _ = &mut *visit;
                $crate::nn::has_parameters!(@visit_params self prefix visit; $($params)*);
                $crate::nn::has_parameters!(@visit_children self prefix visit; $($children)*);
                $crate::nn::has_parameters!(@visit_transparent_children self prefix visit; $($transparent)*);
            }

            fn visit_parameters_mut<'a>(
                &'a mut self,
                prefix: &str,
                visit: &mut dyn FnMut(&str, $crate::nn::ParameterRefMut<'a, E, B>),
            ) {
                let _ = prefix;
                let _ = &mut *visit;
                $crate::nn::has_parameters!(@visit_params_mut self prefix visit; $($params)*);
                $crate::nn::has_parameters!(@visit_children_mut self prefix visit; $($children)*);
                $crate::nn::has_parameters!(@visit_transparent_children_mut self prefix visit; $($transparent)*);
            }

            fn visit_buffers<'a>(
                &'a self,
                prefix: &str,
                visit: &mut dyn FnMut(&str, $crate::nn::BufferRef<'a, E, B>),
            ) {
                let _ = prefix;
                let _ = &mut *visit;
                $crate::nn::has_parameters!(@visit_buffers self prefix visit; $($buffers)*);
                $crate::nn::has_parameters!(@visit_children_buffers self prefix visit; $($children)*);
                $crate::nn::has_parameters!(@visit_transparent_children_buffers self prefix visit; $($transparent)*);
            }
        }
    };

    // ── OLD arm: backward-compat — delegates to new arm with empty buffers ──
    (
        impl[$($generics:tt)*] $ty:ty
        where { $($where_clause:tt)* }
        {
            params { $($params:tt)* }
            children { $($children:tt)* }
            transparent_children { $($transparent:tt)* }
        }
    ) => {
        $crate::nn::has_parameters! {
            impl[$($generics)*] $ty
            where { $($where_clause)* }
            {
                params { $($params)* }
                children { $($children)* }
                transparent_children { $($transparent)* }
                buffers { }
            }
        }
    };

    (@visit_params $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_params $self:ident $prefix:ident $visit:ident; $field:ident ?, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_param $self $prefix $visit $field ?);
        $crate::nn::has_parameters!(@visit_params $self $prefix $visit; $($rest)*);
    };
    (@visit_params $self:ident $prefix:ident $visit:ident; $field:ident ?) => {
        $crate::nn::has_parameters!(@visit_param $self $prefix $visit $field ?);
    };
    (@visit_params $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_param $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_params $self $prefix $visit; $($rest)*);
    };
    (@visit_params $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_param $self $prefix $visit $field);
    };

    (@visit_params_mut $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_params_mut $self:ident $prefix:ident $visit:ident; $field:ident ?, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_param_mut $self $prefix $visit $field ?);
        $crate::nn::has_parameters!(@visit_params_mut $self $prefix $visit; $($rest)*);
    };
    (@visit_params_mut $self:ident $prefix:ident $visit:ident; $field:ident ?) => {
        $crate::nn::has_parameters!(@visit_param_mut $self $prefix $visit $field ?);
    };
    (@visit_params_mut $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_param_mut $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_params_mut $self $prefix $visit; $($rest)*);
    };
    (@visit_params_mut $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_param_mut $self $prefix $visit $field);
    };

    (@visit_param $self:ident $prefix:ident $visit:ident $field:ident) => {
        $visit(
            &$crate::nn::parameter_path($prefix, stringify!($field)),
            $self.$field.as_ref(),
        );
    };

    (@visit_param $self:ident $prefix:ident $visit:ident $field:ident ?) => {
        if let Some(param) = &$self.$field {
            $visit(
                &$crate::nn::parameter_path($prefix, stringify!($field)),
                param.as_ref(),
            );
        }
    };

    (@visit_param_mut $self:ident $prefix:ident $visit:ident $field:ident) => {
        $visit(
            &$crate::nn::parameter_path($prefix, stringify!($field)),
            $self.$field.as_mut(),
        );
    };

    (@visit_param_mut $self:ident $prefix:ident $visit:ident $field:ident ?) => {
        if let Some(param) = &mut $self.$field {
            $visit(
                &$crate::nn::parameter_path($prefix, stringify!($field)),
                param.as_mut(),
            );
        }
    };

    (@visit_children $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_children $self:ident $prefix:ident $visit:ident; $field:ident [], $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child $self $prefix $visit $field []);
        $crate::nn::has_parameters!(@visit_children $self $prefix $visit; $($rest)*);
    };
    (@visit_children $self:ident $prefix:ident $visit:ident; $field:ident []) => {
        $crate::nn::has_parameters!(@visit_child $self $prefix $visit $field []);
    };
    (@visit_children $self:ident $prefix:ident $visit:ident; $field:tt, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_children $self $prefix $visit; $($rest)*);
    };
    (@visit_children $self:ident $prefix:ident $visit:ident; $field:tt) => {
        $crate::nn::has_parameters!(@visit_child $self $prefix $visit $field);
    };

    (@visit_children_mut $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_children_mut $self:ident $prefix:ident $visit:ident; $field:ident [], $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child_mut $self $prefix $visit $field []);
        $crate::nn::has_parameters!(@visit_children_mut $self $prefix $visit; $($rest)*);
    };
    (@visit_children_mut $self:ident $prefix:ident $visit:ident; $field:ident []) => {
        $crate::nn::has_parameters!(@visit_child_mut $self $prefix $visit $field []);
    };
    (@visit_children_mut $self:ident $prefix:ident $visit:ident; $field:tt, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child_mut $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_children_mut $self $prefix $visit; $($rest)*);
    };
    (@visit_children_mut $self:ident $prefix:ident $visit:ident; $field:tt) => {
        $crate::nn::has_parameters!(@visit_child_mut $self $prefix $visit $field);
    };

    (@visit_child $self:ident $prefix:ident $visit:ident $field:ident []) => {
        for (idx, child) in $self.$field.iter().enumerate() {
            let segment = format!("{}.{idx}", stringify!($field));
            let path = $crate::nn::parameter_path($prefix, &segment);
            child.visit_parameters(&path, $visit);
        }
    };

    (@visit_child $self:ident $prefix:ident $visit:ident $field:tt) => {
        $self.$field
            .visit_parameters(&$crate::nn::parameter_path($prefix, stringify!($field)), $visit);
    };

    (@visit_child_mut $self:ident $prefix:ident $visit:ident $field:ident []) => {
        for (idx, child) in $self.$field.iter_mut().enumerate() {
            let segment = format!("{}.{idx}", stringify!($field));
            let path = $crate::nn::parameter_path($prefix, &segment);
            child.visit_parameters_mut(&path, $visit);
        }
    };

    (@visit_child_mut $self:ident $prefix:ident $visit:ident $field:tt) => {
        $self.$field.visit_parameters_mut(
            &$crate::nn::parameter_path($prefix, stringify!($field)),
            $visit,
        );
    };

    (@visit_transparent_children $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_transparent_children $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_transparent_child $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_transparent_children $self $prefix $visit; $($rest)*);
    };
    (@visit_transparent_children $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_transparent_child $self $prefix $visit $field);
    };

    (@visit_transparent_children_mut $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_transparent_children_mut $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_transparent_child_mut $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_transparent_children_mut $self $prefix $visit; $($rest)*);
    };
    (@visit_transparent_children_mut $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_transparent_child_mut $self $prefix $visit $field);
    };

    (@visit_transparent_child $self:ident $prefix:ident $visit:ident $field:ident) => {
        $self.$field.visit_parameters($prefix, $visit);
    };

    (@visit_transparent_child_mut $self:ident $prefix:ident $visit:ident $field:ident) => {
        $self.$field.visit_parameters_mut($prefix, $visit);
    };

    // ── Buffer helper rules ──────────────────────────────────────────────────
    (@visit_buffers $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_buffers $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_buffer $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_buffers $self $prefix $visit; $($rest)*);
    };
    (@visit_buffers $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_buffer $self $prefix $visit $field);
    };

    (@visit_buffer $self:ident $prefix:ident $visit:ident $field:ident) => {
        $visit(
            &$crate::nn::parameter_path($prefix, stringify!($field)),
            $self.$field.as_ref(),
        );
    };

    (@visit_children_buffers $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_children_buffers $self:ident $prefix:ident $visit:ident; $field:ident [], $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child_buffers $self $prefix $visit $field []);
        $crate::nn::has_parameters!(@visit_children_buffers $self $prefix $visit; $($rest)*);
    };
    (@visit_children_buffers $self:ident $prefix:ident $visit:ident; $field:ident []) => {
        $crate::nn::has_parameters!(@visit_child_buffers $self $prefix $visit $field []);
    };
    (@visit_children_buffers $self:ident $prefix:ident $visit:ident; $field:tt, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_child_buffers $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_children_buffers $self $prefix $visit; $($rest)*);
    };
    (@visit_children_buffers $self:ident $prefix:ident $visit:ident; $field:tt) => {
        $crate::nn::has_parameters!(@visit_child_buffers $self $prefix $visit $field);
    };

    (@visit_child_buffers $self:ident $prefix:ident $visit:ident $field:ident []) => {
        for (idx, child) in $self.$field.iter().enumerate() {
            let segment = format!("{}.{idx}", stringify!($field));
            let path = $crate::nn::parameter_path($prefix, &segment);
            child.visit_buffers(&path, $visit);
        }
    };

    (@visit_child_buffers $self:ident $prefix:ident $visit:ident $field:tt) => {
        $self.$field
            .visit_buffers(&$crate::nn::parameter_path($prefix, stringify!($field)), $visit);
    };

    (@visit_transparent_children_buffers $self:ident $prefix:ident $visit:ident;) => {};
    (@visit_transparent_children_buffers $self:ident $prefix:ident $visit:ident; $field:ident, $($rest:tt)*) => {
        $crate::nn::has_parameters!(@visit_transparent_child_buffers $self $prefix $visit $field);
        $crate::nn::has_parameters!(@visit_transparent_children_buffers $self $prefix $visit; $($rest)*);
    };
    (@visit_transparent_children_buffers $self:ident $prefix:ident $visit:ident; $field:ident) => {
        $crate::nn::has_parameters!(@visit_transparent_child_buffers $self $prefix $visit $field);
    };

    (@visit_transparent_child_buffers $self:ident $prefix:ident $visit:ident $field:ident) => {
        $self.$field.visit_buffers($prefix, $visit);
    };
}

pub(crate) use has_parameters;

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

    pub(crate) fn update_data(&mut self, update: UpdateData<'_, E>) -> Result<bool> {
        self.inner.update_data(update)
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
