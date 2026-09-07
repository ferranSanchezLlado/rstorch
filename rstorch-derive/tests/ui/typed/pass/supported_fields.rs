use rstorch::typed::nn::{Module, TypedBuffer, TypedParam, TypedVisitor, TypedVisitorMut};
use rstorch::typed::{Cpu, Tensor1};
use rstorch_derive::TypedModule;

type Value = Tensor1<2, f32, Cpu>;

struct Leaf;

impl Module for Leaf {
    fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}
    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}

#[derive(TypedModule)]
struct Generic<M>
where
    M: Module,
{
    weight: TypedParam<Value>,
    buffer: TypedBuffer<Value>,
    optional_weight: Option<TypedParam<Value>>,
    optional_buffer: Option<TypedBuffer<Value>>,
    child: M,
    optional_child: Option<Leaf>,
    blocks: Vec<Leaf>,
    #[typed_module(skip)]
    label: String,
}

#[derive(TypedModule)]
struct Tuple(TypedParam<Value>, TypedBuffer<Value>, Leaf);

#[derive(TypedModule)]
struct Empty;

fn assert_module<T: Module>() {}

fn main() {
    assert_module::<Generic<Leaf>>();
    assert_module::<Tuple>();
    assert_module::<Empty>();
}
