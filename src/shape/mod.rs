use crate::error::{Result, ShapeError};
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;

mod layout;

pub(crate) use layout::Layout;

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
    const KNOWN: Option<usize>;

    fn known() -> Option<usize> {
        Self::KNOWN
    }

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
    const KNOWN: Option<usize> = Some(N);

    fn symbol() -> Option<DimId> {
        Some(DimId(TypeId::of::<C<N>>()))
    }

    fn symbol_name() -> Option<&'static str> {
        Some(std::any::type_name::<C<N>>())
    }
}

impl<Tag: Send + Sync + 'static> sealed::SealedDim for Sym<Tag> {}

impl<Tag: Send + Sync + 'static> DimSpec for Sym<Tag> {
    const KNOWN: Option<usize> = None;

    fn symbol() -> Option<DimId> {
        Some(DimId(TypeId::of::<Sym<Tag>>()))
    }

    fn symbol_name() -> Option<&'static str> {
        Some(std::any::type_name::<Tag>())
    }
}

impl sealed::SealedDim for AnyDim {}

impl DimSpec for AnyDim {
    const KNOWN: Option<usize> = None;

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
    const KNOWN_NUMEL: Option<usize>;

    fn known_shape() -> Option<Shape>;
    fn dim_entries(operand: usize) -> Vec<DimEntry>;

    fn validate(shape: &Shape) -> Result<()> {
        if shape.rank() != Self::RANK {
            return Err(ShapeError::RankMismatch {
                op: "validate",
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

pub trait LastAxis: ShapeSpec {
    type Last: DimSpec;
    type Reduced: ShapeSpec;
    type Row: ShapeSpec;
}

pub trait LeadingAxis: ShapeSpec {
    type Leading: DimSpec;
    type Reduced: ShapeSpec;
    type Col: ShapeSpec;
}

pub trait StaticShape: ShapeSpec {
    const DIMS: &'static [usize];
    const NUMEL: usize;

    fn static_shape() -> Shape {
        let _ = Self::NUMEL;
        Shape::known(Self::DIMS)
    }
}

pub(crate) const fn static_numel<const N: usize>(dims: [usize; N]) -> usize {
    let mut idx = 0usize;
    let mut numel = 1usize;
    while idx < N {
        match numel.checked_mul(dims[idx]) {
            Some(next) => numel = next,
            None => crate::error::const_check::ConstWriter::new()
                .str("static shape: element count overflow while multiplying by dimension ")
                .num(dims[idx])
                .panic(),
        }
        idx += 1;
    }
    numel
}

pub(crate) const fn known_numel<const N: usize>(dims: [Option<usize>; N]) -> Option<usize> {
    let mut idx = 0usize;
    let mut numel = 1usize;
    while idx < N {
        match dims[idx] {
            Some(dim) => match numel.checked_mul(dim) {
                Some(next) => numel = next,
                None => crate::error::const_check::ConstWriter::new()
                    .str("static shape: element count overflow while multiplying by dimension ")
                    .num(dim)
                    .panic(),
            },
            None => return None,
        }
        idx += 1;
    }
    Some(numel)
}

macro_rules! impl_dims {
    ($($dim:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy)]
        pub struct D0;

        impl sealed::SealedShape for D0 {}

        impl ShapeSpec for D0 {
            const RANK: usize = 0;
            const KNOWN_NUMEL: Option<usize> = Some(1);

            fn known_shape() -> Option<Shape> {
                Some(Shape::known([]))
            }

            fn dim_entries(_operand: usize) -> Vec<DimEntry> {
                Vec::new()
            }
        }

        impl StaticShape for D0 {
            const DIMS: &'static [usize] = &[];
            const NUMEL: usize = 1;
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
            const KNOWN_NUMEL: Option<usize> = known_numel([$($prev::KNOWN,)* $head::KNOWN]);

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
            const DIMS: &'static [usize] = &[$($prev,)* $head];
            const NUMEL: usize = static_numel([$($prev,)* $head]);
        }

        impl<$($prev,)* $head> LastAxis for $rank<$($prev,)* $head>
        where
            $($prev: DimSpec,)*
            $head: DimSpec,
        {
            type Last = $head;
            type Reduced = impl_dims!(@shape [$($prev),*]);
            type Row = D1<$head>;
        }

        impl<$($prev,)* $head> LeadingAxis for $rank<$($prev,)* $head>
        where
            $($prev: DimSpec,)*
            $head: DimSpec,
        {
            type Leading = impl_dims!(@first $($prev,)* $head);
            type Reduced = impl_dims!(@shape_tail $($prev,)* $head);
            type Col = D1<impl_dims!(@first $($prev,)* $head)>;
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

    (@first $head:ident $(, $tail:ident)*) => {
        $head
    };

    (@shape []) => {
        D0
    };

    (@shape [$head:ident]) => {
        D1<$head>
    };

    (@shape [$head:ident, $($tail:ident),+]) => {
        impl_dims!(@shape_nonempty [D2 D3 D4 D5 D6 D7 D8] [$head] ; $($tail),+)
    };

    (@shape_tail $head:ident) => {
        D0
    };

    (@shape_tail $head:ident, $($tail:ident),+) => {
        impl_dims!(@shape [$($tail),+])
    };

    (@shape_nonempty [$rank:ident $($rest_ranks:ident)*] [$($prev:ident),+] ; $head:ident) => {
        $rank<$($prev,)* $head>
    };

    (@shape_nonempty [$rank:ident $($rest_ranks:ident)*] [$($prev:ident),+] ; $head:ident, $($tail:ident),+) => {
        impl_dims!(@shape_nonempty [$($rest_ranks)*] [$($prev,)* $head] ; $($tail),+)
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

    #[test]
    fn static_numel_is_available_at_type_level() {
        assert_eq!(<D2<C<2>, C<3>> as ShapeSpec>::KNOWN_NUMEL, Some(6));
        assert_eq!(<D2<Sym<Batch>, C<3>> as ShapeSpec>::KNOWN_NUMEL, None);
        assert_eq!(<D2<C<2>, C<3>> as StaticShape>::NUMEL, 6);
    }
}
