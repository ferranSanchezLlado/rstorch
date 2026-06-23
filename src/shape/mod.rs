use crate::error::{Result, ShapeError};
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;

mod layout;

pub use layout::Layout;

mod sealed {
    pub trait SealedDim {}
    pub trait SealedShape {}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Box<[usize]>,
}

impl Shape {
    pub fn known(dims: impl Into<Box<[usize]>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn numel(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                ShapeError::NumelOverflow {
                    dims: self.dims.clone(),
                }
                .into()
            })
        })
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::known(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::known(dims.into_boxed_slice())
    }
}

impl From<Box<[usize]>> for Shape {
    fn from(dims: Box<[usize]>) -> Self {
        Self::known(dims)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimId(TypeId);

pub trait DimSpec: sealed::SealedDim + Send + Sync + 'static {
    fn known() -> Option<usize>;
    fn symbol() -> Option<DimId>;
    fn symbol_name() -> Option<&'static str>;
}

#[derive(Debug, Clone, Copy)]
pub struct C<const N: usize>;

#[derive(Debug, Clone, Copy)]
pub struct Sym<Tag: 'static>(PhantomData<Tag>);

#[derive(Debug, Clone, Copy)]
pub struct AnyDim;

impl<const N: usize> sealed::SealedDim for C<N> {}

impl<const N: usize> DimSpec for C<N> {
    fn known() -> Option<usize> {
        Some(N)
    }

    fn symbol() -> Option<DimId> {
        Some(DimId(TypeId::of::<C<N>>()))
    }

    fn symbol_name() -> Option<&'static str> {
        Some(std::any::type_name::<C<N>>())
    }
}

impl<Tag: Send + Sync + 'static> sealed::SealedDim for Sym<Tag> {}

impl<Tag: Send + Sync + 'static> DimSpec for Sym<Tag> {
    fn known() -> Option<usize> {
        None
    }

    fn symbol() -> Option<DimId> {
        Some(DimId(TypeId::of::<Sym<Tag>>()))
    }

    fn symbol_name() -> Option<&'static str> {
        Some(std::any::type_name::<Tag>())
    }
}

impl sealed::SealedDim for AnyDim {}

impl DimSpec for AnyDim {
    fn known() -> Option<usize> {
        None
    }

    fn symbol() -> Option<DimId> {
        None
    }

    fn symbol_name() -> Option<&'static str> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimEntry {
    pub known: Option<usize>,
    pub symbol: Option<DimId>,
    pub symbol_name: Option<&'static str>,
    pub operand: usize,
    pub axis: usize,
}

impl DimEntry {
    pub fn of<D: DimSpec>(operand: usize, axis: usize) -> Self {
        Self {
            known: D::known(),
            symbol: D::symbol(),
            symbol_name: D::symbol_name(),
            operand,
            axis,
        }
    }
}

pub(crate) fn bind_and_check(
    op: &'static str,
    pairs: impl IntoIterator<Item = (DimEntry, usize)>,
) -> Result<()> {
    let mut seen = HashMap::<DimId, (usize, (usize, usize), &'static str)>::new();

    for (entry, size) in pairs {
        if let Some(expected) = entry.known
            && size != expected
        {
            return Err(ShapeError::DimMismatch {
                op,
                operand: entry.operand,
                axis: entry.axis,
                expected,
                found: size,
            }
            .into());
        }

        if let Some(symbol) = entry.symbol {
            let name = entry.symbol_name.unwrap_or("<unknown>");
            let pos = (entry.operand, entry.axis);
            if let Some(&(first_size, first_pos, first_name)) = seen.get(&symbol) {
                if first_size != size {
                    return Err(ShapeError::SymbolMismatch {
                        op,
                        symbol: first_name,
                        lhs: first_pos,
                        rhs: pos,
                        lhs_size: first_size,
                        rhs_size: size,
                    }
                    .into());
                }
            } else {
                seen.insert(symbol, (size, pos, name));
            }
        }
    }

    Ok(())
}

pub trait ShapeSpec: sealed::SealedShape + Send + Sync + 'static {
    const RANK: usize;

    fn known_shape() -> Option<Shape>;
    fn dim_entries(operand: usize) -> Vec<DimEntry>;

    fn validate(shape: &Shape) -> Result<()> {
        if shape.rank() != Self::RANK {
            return Err(ShapeError::RankMismatch {
                expected: Self::RANK,
                found: shape.rank(),
            }
            .into());
        }

        bind_and_check(
            "validate",
            Self::dim_entries(0)
                .into_iter()
                .zip(shape.dims().iter().copied()),
        )
    }
}

pub trait StaticShape: ShapeSpec {
    fn static_shape() -> Shape;
}

macro_rules! impl_dims {
    ($($dim:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy)]
        pub struct D0;

        impl sealed::SealedShape for D0 {}

        impl ShapeSpec for D0 {
            const RANK: usize = 0;

            fn known_shape() -> Option<Shape> {
                Some(Shape::known([]))
            }

            fn dim_entries(_operand: usize) -> Vec<DimEntry> {
                Vec::new()
            }
        }

        impl StaticShape for D0 {
            fn static_shape() -> Shape {
                Shape::known([])
            }
        }

        impl_dims!(@emit [D1 D2 D3 D4 D5 D6 D7 D8] [] ; $($dim),+);
    };

    (@emit [$rank:ident $($rest_ranks:ident)*] [$($prev:ident),*] ; $head:ident $(, $tail:ident)*) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $rank<$($prev,)* $head>(PhantomData<fn($($prev,)* $head)>);

        impl<$($prev,)* $head> sealed::SealedShape for $rank<$($prev,)* $head>
        where
            $($prev: DimSpec,)*
            $head: DimSpec,
        {
        }

        impl<$($prev,)* $head> ShapeSpec for $rank<$($prev,)* $head>
        where
            $($prev: DimSpec,)*
            $head: DimSpec,
        {
            const RANK: usize = impl_dims!(@count $($prev,)* $head);

            fn known_shape() -> Option<Shape> {
                Some(Shape::known([$($prev::known()?,)* $head::known()?]))
            }

            fn dim_entries(operand: usize) -> Vec<DimEntry> {
                let builders: [fn(usize, usize) -> DimEntry; impl_dims!(@count $($prev,)* $head)] = [
                    $(DimEntry::of::<$prev>,)*
                    DimEntry::of::<$head>,
                ];

                builders
                    .into_iter()
                    .enumerate()
                    .map(|(axis, build)| build(operand, axis))
                    .collect()
            }
        }

        impl<$(const $prev: usize,)* const $head: usize> StaticShape
            for $rank<$(C<$prev>,)* C<$head>>
        {
            fn static_shape() -> Shape {
                Shape::known([$($prev,)* $head])
            }
        }

        impl_dims!(@emit [$($rest_ranks)*] [$($prev,)* $head] ; $($tail),*);
    };

    (@emit [$($rank:ident)*] [$($prev:ident),*] ;) => {};

    (@count $($dim:ident),*) => {
        0usize $(+ impl_dims!(@one $dim))*
    };

    (@one $dim:ident) => {
        1usize
    };
}

impl_dims!(A, B, CC, D);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[derive(Debug)]
    struct Batch;

    #[derive(Debug)]
    struct Hidden;

    #[test]
    fn shape_numel_is_checked() {
        assert_eq!(Shape::known([2, 3]).numel().unwrap(), 6);
        assert!(matches!(
            Shape::known([usize::MAX, 2]).numel(),
            Err(Error::Shape(ShapeError::NumelOverflow { .. }))
        ));
    }

    #[test]
    fn sym_identity_uses_marker_type() {
        assert_eq!(Sym::<Batch>::symbol(), Sym::<Batch>::symbol());
        assert_ne!(Sym::<Batch>::symbol(), Sym::<Hidden>::symbol());
        assert_ne!(Sym::<Batch>::symbol(), Some(DimId(TypeId::of::<Batch>())));
    }
}
