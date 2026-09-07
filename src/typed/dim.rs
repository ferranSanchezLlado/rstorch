//! Dimension metadata and stable-Rust associated output mappings.

use super::ops::{
    ArgKeepDimOutput, ArgOutput, BroadcastOutput, ConcatOutput, Conv2dOutput, GatherOutput,
    IndexSelectOutput, InsertAxisOutput, KeepDimOutput, MatmulOutput, Pool2dOutput, RefinementOf,
    RemoveAxisOutput, ReplaceAxisOutput, StackOutput, TransposeOutput,
};
use super::{
    Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7, Tensor8,
    TypedTensor, typed_rank_table,
};
use crate::Element;

/// A type-level dimension whose value is retained only at runtime.
///
/// Each occurrence is independent. `usize::MAX` is reserved for this wildcard;
/// zero is a valid static dimension. Rust cannot distinguish another spelling
/// of the literal `usize::MAX` from [`DYN`].
pub const DYN: usize = usize::MAX;

macro_rules! impl_common_relations {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement, Target>
                BroadcastOutput<Target> for $name<$($dim,)* E, P>
            where
                Target: TypedTensor<Elem = E, Placement = P>,
            {
                type Output = Target;
            }
        )+
    };
}

typed_rank_table!(impl_common_relations);

macro_rules! impl_axis_outputs {
    (
        $name:ident, $all:tt, $removed_name:ident;
        $($axis:literal => remove [$($removed:ident),*], replace [$($replaced:tt),+];)+
    ) => {
        $(
            impl_axis_output_one!(
                $name, $all, $removed_name, $axis,
                [$($removed),*], [$($replaced),+]
            );
        )+
    };
    (@dim @, $value:expr) => { $value };
    (@dim $dim:ident, $value:expr) => { $dim };
}

macro_rules! impl_axis_output_one {
    ($name:ident, [$($all:ident),+], $removed_name:ident, $axis:literal,
        [$($removed:ident),*], [$($replaced:tt),+]
    ) => {
        impl<$(const $all: usize,)+ E: Element, P: Placement> RemoveAxisOutput<$axis>
            for $name<$($all,)+ E, P>
        {
            type Output = $removed_name<$($removed,)* E, P>;
        }

        impl<$(const $all: usize,)+ const DIM: usize, E: Element, P: Placement>
            ReplaceAxisOutput<$axis, DIM> for $name<$($all,)+ E, P>
        {
            type Output = $name<$({ impl_axis_outputs!(@dim $replaced, DIM) },)+ E, P>;
        }

        impl<$(const $all: usize,)+ E: Element, P: Placement> KeepDimOutput<$axis>
            for $name<$($all,)+ E, P>
        {
            type Output = $name<$({ impl_axis_outputs!(@dim $replaced, 1) },)+ E, P>;
        }

        impl<$(const $all: usize,)+ E: Element, P: Placement> ArgOutput<$axis>
            for $name<$($all,)+ E, P>
        {
            type Output = $removed_name<$($removed,)* i64, P>;
        }

        impl<$(const $all: usize,)+ E: Element, P: Placement> ArgKeepDimOutput<$axis>
            for $name<$($all,)+ E, P>
        {
            type Output = $name<$({ impl_axis_outputs!(@dim $replaced, 1) },)+ i64, P>;
        }

        impl<$(const $all: usize,)+ const LEN: usize, E: Element, P: Placement>
            IndexSelectOutput<$axis, Tensor1<LEN, i64, P>> for $name<$($all,)+ E, P>
        {
            type Output = $name<$({ impl_axis_outputs!(@dim $replaced, LEN) },)+ E, P>;
        }

        impl<$(const $all: usize,)+ E: Element, P: Placement> ConcatOutput<$axis>
            for $name<$($all,)+ E, P>
        {
            type Output = $name<$({ impl_axis_outputs!(@dim $replaced, DYN) },)+ E, P>;
        }
    };
}

impl_axis_outputs!(Tensor1, [D0], Tensor0;
    0 => remove [], replace [@];
);
impl_axis_outputs!(Tensor2, [D0, D1], Tensor1;
    0 => remove [D1], replace [@, D1];
    1 => remove [D0], replace [D0, @];
);
impl_axis_outputs!(Tensor3, [D0, D1, D2], Tensor2;
    0 => remove [D1, D2], replace [@, D1, D2];
    1 => remove [D0, D2], replace [D0, @, D2];
    2 => remove [D0, D1], replace [D0, D1, @];
);
impl_axis_outputs!(Tensor4, [D0, D1, D2, D3], Tensor3;
    0 => remove [D1, D2, D3], replace [@, D1, D2, D3];
    1 => remove [D0, D2, D3], replace [D0, @, D2, D3];
    2 => remove [D0, D1, D3], replace [D0, D1, @, D3];
    3 => remove [D0, D1, D2], replace [D0, D1, D2, @];
);
impl_axis_outputs!(Tensor5, [D0, D1, D2, D3, D4], Tensor4;
    0 => remove [D1, D2, D3, D4], replace [@, D1, D2, D3, D4];
    1 => remove [D0, D2, D3, D4], replace [D0, @, D2, D3, D4];
    2 => remove [D0, D1, D3, D4], replace [D0, D1, @, D3, D4];
    3 => remove [D0, D1, D2, D4], replace [D0, D1, D2, @, D4];
    4 => remove [D0, D1, D2, D3], replace [D0, D1, D2, D3, @];
);
impl_axis_outputs!(Tensor6, [D0, D1, D2, D3, D4, D5], Tensor5;
    0 => remove [D1, D2, D3, D4, D5], replace [@, D1, D2, D3, D4, D5];
    1 => remove [D0, D2, D3, D4, D5], replace [D0, @, D2, D3, D4, D5];
    2 => remove [D0, D1, D3, D4, D5], replace [D0, D1, @, D3, D4, D5];
    3 => remove [D0, D1, D2, D4, D5], replace [D0, D1, D2, @, D4, D5];
    4 => remove [D0, D1, D2, D3, D5], replace [D0, D1, D2, D3, @, D5];
    5 => remove [D0, D1, D2, D3, D4], replace [D0, D1, D2, D3, D4, @];
);
impl_axis_outputs!(Tensor7, [D0, D1, D2, D3, D4, D5, D6], Tensor6;
    0 => remove [D1, D2, D3, D4, D5, D6], replace [@, D1, D2, D3, D4, D5, D6];
    1 => remove [D0, D2, D3, D4, D5, D6], replace [D0, @, D2, D3, D4, D5, D6];
    2 => remove [D0, D1, D3, D4, D5, D6], replace [D0, D1, @, D3, D4, D5, D6];
    3 => remove [D0, D1, D2, D4, D5, D6], replace [D0, D1, D2, @, D4, D5, D6];
    4 => remove [D0, D1, D2, D3, D5, D6], replace [D0, D1, D2, D3, @, D5, D6];
    5 => remove [D0, D1, D2, D3, D4, D6], replace [D0, D1, D2, D3, D4, @, D6];
    6 => remove [D0, D1, D2, D3, D4, D5], replace [D0, D1, D2, D3, D4, D5, @];
);
impl_axis_outputs!(Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7], Tensor7;
    0 => remove [D1, D2, D3, D4, D5, D6, D7], replace [@, D1, D2, D3, D4, D5, D6, D7];
    1 => remove [D0, D2, D3, D4, D5, D6, D7], replace [D0, @, D2, D3, D4, D5, D6, D7];
    2 => remove [D0, D1, D3, D4, D5, D6, D7], replace [D0, D1, @, D3, D4, D5, D6, D7];
    3 => remove [D0, D1, D2, D4, D5, D6, D7], replace [D0, D1, D2, @, D4, D5, D6, D7];
    4 => remove [D0, D1, D2, D3, D5, D6, D7], replace [D0, D1, D2, D3, @, D5, D6, D7];
    5 => remove [D0, D1, D2, D3, D4, D6, D7], replace [D0, D1, D2, D3, D4, @, D6, D7];
    6 => remove [D0, D1, D2, D3, D4, D5, D7], replace [D0, D1, D2, D3, D4, D5, @, D7];
    7 => remove [D0, D1, D2, D3, D4, D5, D6], replace [D0, D1, D2, D3, D4, D5, D6, @];
);

macro_rules! impl_insert_outputs {
    ($source:ident, $input:tt, $target:ident;
        $($axis:literal => [$($output:tt),+];)+
    ) => {
        $(
            impl_insert_output_one!($source, $input, $target, $axis, [$($output),+]);
        )+
    };
}

macro_rules! impl_insert_output_one {
    ($source:ident, [$($input:ident),*], $target:ident, $axis:literal, [$($output:tt),+]) => {
        impl<$(const $input: usize,)* const DIM: usize, E: Element, P: Placement>
            InsertAxisOutput<$axis, DIM> for $source<$($input,)* E, P>
        {
            type Output = $target<$({ impl_axis_outputs!(@dim $output, DIM) },)+ E, P>;
        }

        impl<$(const $input: usize,)* E: Element, P: Placement> StackOutput<$axis>
            for $source<$($input,)* E, P>
        {
            type Output = $target<$({ impl_axis_outputs!(@dim $output, DYN) },)+ E, P>;
        }
    };
}

impl_insert_outputs!(Tensor0, [], Tensor1; 0 => [@];);
impl_insert_outputs!(Tensor1, [D0], Tensor2;
    0 => [@, D0]; 1 => [D0, @];
);
impl_insert_outputs!(Tensor2, [D0, D1], Tensor3;
    0 => [@, D0, D1]; 1 => [D0, @, D1]; 2 => [D0, D1, @];
);
impl_insert_outputs!(Tensor3, [D0, D1, D2], Tensor4;
    0 => [@, D0, D1, D2]; 1 => [D0, @, D1, D2];
    2 => [D0, D1, @, D2]; 3 => [D0, D1, D2, @];
);
impl_insert_outputs!(Tensor4, [D0, D1, D2, D3], Tensor5;
    0 => [@, D0, D1, D2, D3]; 1 => [D0, @, D1, D2, D3];
    2 => [D0, D1, @, D2, D3]; 3 => [D0, D1, D2, @, D3];
    4 => [D0, D1, D2, D3, @];
);
impl_insert_outputs!(Tensor5, [D0, D1, D2, D3, D4], Tensor6;
    0 => [@, D0, D1, D2, D3, D4]; 1 => [D0, @, D1, D2, D3, D4];
    2 => [D0, D1, @, D2, D3, D4]; 3 => [D0, D1, D2, @, D3, D4];
    4 => [D0, D1, D2, D3, @, D4]; 5 => [D0, D1, D2, D3, D4, @];
);
impl_insert_outputs!(Tensor6, [D0, D1, D2, D3, D4, D5], Tensor7;
    0 => [@, D0, D1, D2, D3, D4, D5]; 1 => [D0, @, D1, D2, D3, D4, D5];
    2 => [D0, D1, @, D2, D3, D4, D5]; 3 => [D0, D1, D2, @, D3, D4, D5];
    4 => [D0, D1, D2, D3, @, D4, D5]; 5 => [D0, D1, D2, D3, D4, @, D5];
    6 => [D0, D1, D2, D3, D4, D5, @];
);
impl_insert_outputs!(Tensor7, [D0, D1, D2, D3, D4, D5, D6], Tensor8;
    0 => [@, D0, D1, D2, D3, D4, D5, D6]; 1 => [D0, @, D1, D2, D3, D4, D5, D6];
    2 => [D0, D1, @, D2, D3, D4, D5, D6]; 3 => [D0, D1, D2, @, D3, D4, D5, D6];
    4 => [D0, D1, D2, D3, @, D4, D5, D6]; 5 => [D0, D1, D2, D3, D4, @, D5, D6];
    6 => [D0, D1, D2, D3, D4, D5, @, D6]; 7 => [D0, D1, D2, D3, D4, D5, D6, @];
);

macro_rules! impl_transpose_outputs {
    ($name:ident, $dim:tt; $($a:literal, $b:literal => [$($out:ident),+];)+) => {
        $(
            impl_transpose_output_one!($name, $dim, $a, $b, [$($out),+]);
        )+
    };
}

macro_rules! impl_transpose_output_one {
    ($name:ident, [$($dim:ident),+], $a:literal, $b:literal, [$($out:ident),+]) => {
        impl<$(const $dim: usize,)+ E: Element, P: Placement> TransposeOutput<$a, $b>
            for $name<$($dim,)+ E, P>
        {
            type Output = $name<$($out,)+ E, P>;
        }
    };
}

impl_transpose_outputs!(Tensor1, [D0]; 0, 0 => [D0];);
impl_transpose_outputs!(Tensor2, [D0, D1];
    0, 0 => [D0, D1]; 0, 1 => [D1, D0];
    1, 0 => [D1, D0]; 1, 1 => [D0, D1];
);
impl_transpose_outputs!(Tensor3, [D0, D1, D2];
    0, 0 => [D0, D1, D2]; 0, 1 => [D1, D0, D2]; 0, 2 => [D2, D1, D0];
    1, 0 => [D1, D0, D2]; 1, 1 => [D0, D1, D2]; 1, 2 => [D0, D2, D1];
    2, 0 => [D2, D1, D0]; 2, 1 => [D0, D2, D1]; 2, 2 => [D0, D1, D2];
);
impl_transpose_outputs!(Tensor4, [D0, D1, D2, D3];
    0, 0 => [D0,D1,D2,D3]; 0, 1 => [D1,D0,D2,D3]; 0, 2 => [D2,D1,D0,D3]; 0, 3 => [D3,D1,D2,D0];
    1, 0 => [D1,D0,D2,D3]; 1, 1 => [D0,D1,D2,D3]; 1, 2 => [D0,D2,D1,D3]; 1, 3 => [D0,D3,D2,D1];
    2, 0 => [D2,D1,D0,D3]; 2, 1 => [D0,D2,D1,D3]; 2, 2 => [D0,D1,D2,D3]; 2, 3 => [D0,D1,D3,D2];
    3, 0 => [D3,D1,D2,D0]; 3, 1 => [D0,D3,D2,D1]; 3, 2 => [D0,D1,D3,D2]; 3, 3 => [D0,D1,D2,D3];
);

// Stable Rust requires one direct associated output per axis pair. The row
// macro keeps the remaining exhaustive mappings declarative.
macro_rules! transpose_row {
    ($name:ident, $dim:tt, $a:literal; $($b:literal => [$($out:ident),+]),+ $(,)?) => {
        impl_transpose_outputs!($name, $dim; $($a, $b => [$($out),+];)+);
    };
}

transpose_row!(Tensor5,[D0,D1,D2,D3,D4],0;0=>[D0,D1,D2,D3,D4],1=>[D1,D0,D2,D3,D4],2=>[D2,D1,D0,D3,D4],3=>[D3,D1,D2,D0,D4],4=>[D4,D1,D2,D3,D0]);
transpose_row!(Tensor5,[D0,D1,D2,D3,D4],1;0=>[D1,D0,D2,D3,D4],1=>[D0,D1,D2,D3,D4],2=>[D0,D2,D1,D3,D4],3=>[D0,D3,D2,D1,D4],4=>[D0,D4,D2,D3,D1]);
transpose_row!(Tensor5,[D0,D1,D2,D3,D4],2;0=>[D2,D1,D0,D3,D4],1=>[D0,D2,D1,D3,D4],2=>[D0,D1,D2,D3,D4],3=>[D0,D1,D3,D2,D4],4=>[D0,D1,D4,D3,D2]);
transpose_row!(Tensor5,[D0,D1,D2,D3,D4],3;0=>[D3,D1,D2,D0,D4],1=>[D0,D3,D2,D1,D4],2=>[D0,D1,D3,D2,D4],3=>[D0,D1,D2,D3,D4],4=>[D0,D1,D2,D4,D3]);
transpose_row!(Tensor5,[D0,D1,D2,D3,D4],4;0=>[D4,D1,D2,D3,D0],1=>[D0,D4,D2,D3,D1],2=>[D0,D1,D4,D3,D2],3=>[D0,D1,D2,D4,D3],4=>[D0,D1,D2,D3,D4]);

// Rank six through eight use the same exhaustive rows. Keeping the invocations
// data-only makes omissions visible and avoids generic const expressions.
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],0;0=>[D0,D1,D2,D3,D4,D5],1=>[D1,D0,D2,D3,D4,D5],2=>[D2,D1,D0,D3,D4,D5],3=>[D3,D1,D2,D0,D4,D5],4=>[D4,D1,D2,D3,D0,D5],5=>[D5,D1,D2,D3,D4,D0]);
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],1;0=>[D1,D0,D2,D3,D4,D5],1=>[D0,D1,D2,D3,D4,D5],2=>[D0,D2,D1,D3,D4,D5],3=>[D0,D3,D2,D1,D4,D5],4=>[D0,D4,D2,D3,D1,D5],5=>[D0,D5,D2,D3,D4,D1]);
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],2;0=>[D2,D1,D0,D3,D4,D5],1=>[D0,D2,D1,D3,D4,D5],2=>[D0,D1,D2,D3,D4,D5],3=>[D0,D1,D3,D2,D4,D5],4=>[D0,D1,D4,D3,D2,D5],5=>[D0,D1,D5,D3,D4,D2]);
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],3;0=>[D3,D1,D2,D0,D4,D5],1=>[D0,D3,D2,D1,D4,D5],2=>[D0,D1,D3,D2,D4,D5],3=>[D0,D1,D2,D3,D4,D5],4=>[D0,D1,D2,D4,D3,D5],5=>[D0,D1,D2,D5,D4,D3]);
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],4;0=>[D4,D1,D2,D3,D0,D5],1=>[D0,D4,D2,D3,D1,D5],2=>[D0,D1,D4,D3,D2,D5],3=>[D0,D1,D2,D4,D3,D5],4=>[D0,D1,D2,D3,D4,D5],5=>[D0,D1,D2,D3,D5,D4]);
transpose_row!(Tensor6,[D0,D1,D2,D3,D4,D5],5;0=>[D5,D1,D2,D3,D4,D0],1=>[D0,D5,D2,D3,D4,D1],2=>[D0,D1,D5,D3,D4,D2],3=>[D0,D1,D2,D5,D4,D3],4=>[D0,D1,D2,D3,D5,D4],5=>[D0,D1,D2,D3,D4,D5]);

macro_rules! exhaustive_transpose_tail {
    ($name:ident, $dim:tt; $($a:literal => { $($b:literal => [$($out:ident),+]),+ }),+ $(,)?) => {
        $(exhaustive_transpose_row!($name, $dim, $a; $($b => [$($out),+]),+);)+
    };
}

macro_rules! exhaustive_transpose_row {
    ($name:ident, [$($dim:ident),+], $a:literal; $($b:literal => [$($out:ident),+]),+) => {
        transpose_row!($name, [$($dim),+], $a; $($b => [$($out),+]),+);
    };
}

exhaustive_transpose_tail!(Tensor7,[D0,D1,D2,D3,D4,D5,D6];
0=>{0=>[D0,D1,D2,D3,D4,D5,D6],1=>[D1,D0,D2,D3,D4,D5,D6],2=>[D2,D1,D0,D3,D4,D5,D6],3=>[D3,D1,D2,D0,D4,D5,D6],4=>[D4,D1,D2,D3,D0,D5,D6],5=>[D5,D1,D2,D3,D4,D0,D6],6=>[D6,D1,D2,D3,D4,D5,D0]},
1=>{0=>[D1,D0,D2,D3,D4,D5,D6],1=>[D0,D1,D2,D3,D4,D5,D6],2=>[D0,D2,D1,D3,D4,D5,D6],3=>[D0,D3,D2,D1,D4,D5,D6],4=>[D0,D4,D2,D3,D1,D5,D6],5=>[D0,D5,D2,D3,D4,D1,D6],6=>[D0,D6,D2,D3,D4,D5,D1]},
2=>{0=>[D2,D1,D0,D3,D4,D5,D6],1=>[D0,D2,D1,D3,D4,D5,D6],2=>[D0,D1,D2,D3,D4,D5,D6],3=>[D0,D1,D3,D2,D4,D5,D6],4=>[D0,D1,D4,D3,D2,D5,D6],5=>[D0,D1,D5,D3,D4,D2,D6],6=>[D0,D1,D6,D3,D4,D5,D2]},
3=>{0=>[D3,D1,D2,D0,D4,D5,D6],1=>[D0,D3,D2,D1,D4,D5,D6],2=>[D0,D1,D3,D2,D4,D5,D6],3=>[D0,D1,D2,D3,D4,D5,D6],4=>[D0,D1,D2,D4,D3,D5,D6],5=>[D0,D1,D2,D5,D4,D3,D6],6=>[D0,D1,D2,D6,D4,D5,D3]},
4=>{0=>[D4,D1,D2,D3,D0,D5,D6],1=>[D0,D4,D2,D3,D1,D5,D6],2=>[D0,D1,D4,D3,D2,D5,D6],3=>[D0,D1,D2,D4,D3,D5,D6],4=>[D0,D1,D2,D3,D4,D5,D6],5=>[D0,D1,D2,D3,D5,D4,D6],6=>[D0,D1,D2,D3,D6,D5,D4]},
5=>{0=>[D5,D1,D2,D3,D4,D0,D6],1=>[D0,D5,D2,D3,D4,D1,D6],2=>[D0,D1,D5,D3,D4,D2,D6],3=>[D0,D1,D2,D5,D4,D3,D6],4=>[D0,D1,D2,D3,D5,D4,D6],5=>[D0,D1,D2,D3,D4,D5,D6],6=>[D0,D1,D2,D3,D4,D6,D5]},
6=>{0=>[D6,D1,D2,D3,D4,D5,D0],1=>[D0,D6,D2,D3,D4,D5,D1],2=>[D0,D1,D6,D3,D4,D5,D2],3=>[D0,D1,D2,D6,D4,D5,D3],4=>[D0,D1,D2,D3,D6,D5,D4],5=>[D0,D1,D2,D3,D4,D6,D5],6=>[D0,D1,D2,D3,D4,D5,D6]}
);

exhaustive_transpose_tail!(Tensor8,[D0,D1,D2,D3,D4,D5,D6,D7];
0=>{0=>[D0,D1,D2,D3,D4,D5,D6,D7],1=>[D1,D0,D2,D3,D4,D5,D6,D7],2=>[D2,D1,D0,D3,D4,D5,D6,D7],3=>[D3,D1,D2,D0,D4,D5,D6,D7],4=>[D4,D1,D2,D3,D0,D5,D6,D7],5=>[D5,D1,D2,D3,D4,D0,D6,D7],6=>[D6,D1,D2,D3,D4,D5,D0,D7],7=>[D7,D1,D2,D3,D4,D5,D6,D0]},
1=>{0=>[D1,D0,D2,D3,D4,D5,D6,D7],1=>[D0,D1,D2,D3,D4,D5,D6,D7],2=>[D0,D2,D1,D3,D4,D5,D6,D7],3=>[D0,D3,D2,D1,D4,D5,D6,D7],4=>[D0,D4,D2,D3,D1,D5,D6,D7],5=>[D0,D5,D2,D3,D4,D1,D6,D7],6=>[D0,D6,D2,D3,D4,D5,D1,D7],7=>[D0,D7,D2,D3,D4,D5,D6,D1]},
2=>{0=>[D2,D1,D0,D3,D4,D5,D6,D7],1=>[D0,D2,D1,D3,D4,D5,D6,D7],2=>[D0,D1,D2,D3,D4,D5,D6,D7],3=>[D0,D1,D3,D2,D4,D5,D6,D7],4=>[D0,D1,D4,D3,D2,D5,D6,D7],5=>[D0,D1,D5,D3,D4,D2,D6,D7],6=>[D0,D1,D6,D3,D4,D5,D2,D7],7=>[D0,D1,D7,D3,D4,D5,D6,D2]},
3=>{0=>[D3,D1,D2,D0,D4,D5,D6,D7],1=>[D0,D3,D2,D1,D4,D5,D6,D7],2=>[D0,D1,D3,D2,D4,D5,D6,D7],3=>[D0,D1,D2,D3,D4,D5,D6,D7],4=>[D0,D1,D2,D4,D3,D5,D6,D7],5=>[D0,D1,D2,D5,D4,D3,D6,D7],6=>[D0,D1,D2,D6,D4,D5,D3,D7],7=>[D0,D1,D2,D7,D4,D5,D6,D3]},
4=>{0=>[D4,D1,D2,D3,D0,D5,D6,D7],1=>[D0,D4,D2,D3,D1,D5,D6,D7],2=>[D0,D1,D4,D3,D2,D5,D6,D7],3=>[D0,D1,D2,D4,D3,D5,D6,D7],4=>[D0,D1,D2,D3,D4,D5,D6,D7],5=>[D0,D1,D2,D3,D5,D4,D6,D7],6=>[D0,D1,D2,D3,D6,D5,D4,D7],7=>[D0,D1,D2,D3,D7,D5,D6,D4]},
5=>{0=>[D5,D1,D2,D3,D4,D0,D6,D7],1=>[D0,D5,D2,D3,D4,D1,D6,D7],2=>[D0,D1,D5,D3,D4,D2,D6,D7],3=>[D0,D1,D2,D5,D4,D3,D6,D7],4=>[D0,D1,D2,D3,D5,D4,D6,D7],5=>[D0,D1,D2,D3,D4,D5,D6,D7],6=>[D0,D1,D2,D3,D4,D6,D5,D7],7=>[D0,D1,D2,D3,D4,D7,D6,D5]},
6=>{0=>[D6,D1,D2,D3,D4,D5,D0,D7],1=>[D0,D6,D2,D3,D4,D5,D1,D7],2=>[D0,D1,D6,D3,D4,D5,D2,D7],3=>[D0,D1,D2,D6,D4,D5,D3,D7],4=>[D0,D1,D2,D3,D6,D5,D4,D7],5=>[D0,D1,D2,D3,D4,D6,D5,D7],6=>[D0,D1,D2,D3,D4,D5,D6,D7],7=>[D0,D1,D2,D3,D4,D5,D7,D6]},
7=>{0=>[D7,D1,D2,D3,D4,D5,D6,D0],1=>[D0,D7,D2,D3,D4,D5,D6,D1],2=>[D0,D1,D7,D3,D4,D5,D6,D2],3=>[D0,D1,D2,D7,D4,D5,D6,D3],4=>[D0,D1,D2,D3,D7,D5,D6,D4],5=>[D0,D1,D2,D3,D4,D7,D6,D5],6=>[D0,D1,D2,D3,D4,D5,D7,D6],7=>[D0,D1,D2,D3,D4,D5,D6,D7]}
);

macro_rules! impl_refinement_and_gather {
    ($name:ident, [$($target:ident),*], [$($source:ident),*]) => {
        impl<$(const $target: usize,)* $(const $source: usize,)* E: Element, P: Placement>
            RefinementOf<$name<$($source,)* E, P>> for $name<$($target,)* E, P>
        {}

        impl<$(const $target: usize,)* $(const $source: usize,)* E: Element, P: Placement>
            GatherOutput<$name<$($source,)* i64, P>> for $name<$($target,)* E, P>
        {
            type Output = $name<$($source,)* E, P>;
        }
    };
}

impl_refinement_and_gather!(Tensor0, [], []);
impl_refinement_and_gather!(Tensor1, [D0], [I0]);
impl_refinement_and_gather!(Tensor2, [D0, D1], [I0, I1]);
impl_refinement_and_gather!(Tensor3, [D0, D1, D2], [I0, I1, I2]);
impl_refinement_and_gather!(Tensor4, [D0, D1, D2, D3], [I0, I1, I2, I3]);
impl_refinement_and_gather!(Tensor5, [D0, D1, D2, D3, D4], [I0, I1, I2, I3, I4]);
impl_refinement_and_gather!(Tensor6, [D0, D1, D2, D3, D4, D5], [I0, I1, I2, I3, I4, I5]);
impl_refinement_and_gather!(
    Tensor7,
    [D0, D1, D2, D3, D4, D5, D6],
    [I0, I1, I2, I3, I4, I5, I6]
);
impl_refinement_and_gather!(
    Tensor8,
    [D0, D1, D2, D3, D4, D5, D6, D7],
    [I0, I1, I2, I3, I4, I5, I6, I7]
);

impl<const M: usize, const K1: usize, const K2: usize, const N: usize, E: Element, P: Placement>
    MatmulOutput<Tensor2<K2, N, E, P>> for Tensor2<M, K1, E, P>
{
    type Output = Tensor2<M, N, E, P>;
}

macro_rules! impl_batched_matmul {
    ($name:ident, [$($batch:ident),+]) => {
        impl<$(const $batch: usize,)+ const M: usize, const K1: usize, const K2: usize,
            const N: usize, E: Element, P: Placement>
            MatmulOutput<Tensor2<K2, N, E, P>> for $name<$($batch,)+ M, K1, E, P>
        {
            type Output = $name<$($batch,)+ M, N, E, P>;
        }

        impl<$(const $batch: usize,)+ const M: usize, const K1: usize, const K2: usize,
            const N: usize, E: Element, P: Placement>
            MatmulOutput<$name<$($batch,)+ K2, N, E, P>> for $name<$($batch,)+ M, K1, E, P>
        {
            type Output = $name<$($batch,)+ M, N, E, P>;
        }
    };
}

impl_batched_matmul!(Tensor3, [B0]);
impl_batched_matmul!(Tensor4, [B0, B1]);
impl_batched_matmul!(Tensor5, [B0, B1, B2]);
impl_batched_matmul!(Tensor6, [B0, B1, B2, B3]);
impl_batched_matmul!(Tensor7, [B0, B1, B2, B3, B4]);
impl_batched_matmul!(Tensor8, [B0, B1, B2, B3, B4, B5]);

impl<
    const B: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const OUT: usize,
    const WC: usize,
    const KH: usize,
    const KW: usize,
    E: Element,
    P: Placement,
> Conv2dOutput<Tensor4<OUT, WC, KH, KW, E, P>> for Tensor4<B, C, H, W, E, P>
{
    type Output = Tensor4<B, OUT, DYN, DYN, E, P>;
}

impl<const B: usize, const C: usize, const H: usize, const W: usize, E: Element, P: Placement>
    Pool2dOutput for Tensor4<B, C, H, W, E, P>
{
    type Output = Tensor4<B, C, DYN, DYN, E, P>;
}

#[cfg(test)]
mod tests {
    use super::*;

    trait Same<T> {}
    impl<T> Same<T> for T {}

    fn assert_output<T, Expected>()
    where
        T: TypedTensor + Same<Expected>,
        Expected: TypedTensor,
    {
    }

    fn assert_transpose<T, Expected, const A: usize, const B: usize>()
    where
        T: TransposeOutput<A, B, Output = Expected>,
        Expected: TypedTensor,
    {
    }

    fn assert_remove<T, Expected, const A: usize>()
    where
        T: RemoveAxisOutput<A, Output = Expected>,
        Expected: TypedTensor,
    {
    }

    #[test]
    fn rank_table_preserves_every_marker_occurrence() {
        assert_eq!(
            <Tensor0 as super::super::sealed::TypedTensor>::MARKERS,
            &[] as &[usize]
        );
        assert_eq!(
            <Tensor4<DYN, DYN, 0, 7> as super::super::sealed::TypedTensor>::MARKERS,
            &[DYN, DYN, 0, 7]
        );
        assert_eq!(
            <Tensor8<0, 1, 2, 3, 4, 5, 6, DYN> as super::super::sealed::TypedTensor>::MARKERS.len(),
            8
        );
    }

    #[test]
    fn representative_axis_outputs_are_exact() {
        assert_transpose::<Tensor8<0, 1, 2, 3, 4, 5, 6, 7>, Tensor8<7, 1, 2, 3, 4, 5, 6, 0>, 0, 7>(
        );
        assert_remove::<Tensor8<0, 1, 2, 3, 4, 5, 6, 7>, Tensor7<0, 1, 2, 3, 4, 5, 7>, 6>();
        assert_output::<
            <Tensor7<1, 2, 3, 4, 5, 6, 7> as InsertAxisOutput<3, 0>>::Output,
            Tensor8<1, 2, 3, 0, 4, 5, 6, 7>,
        >();
        assert_output::<
            <Tensor7<1, 2, 3, 4, 5, 6, 7> as StackOutput<7>>::Output,
            Tensor8<1, 2, 3, 4, 5, 6, 7, DYN>,
        >();
    }

    #[test]
    fn operation_outputs_retain_only_promised_metadata() {
        assert_output::<<Tensor4<2, 3, 8, 9> as Pool2dOutput>::Output, Tensor4<2, 3, DYN, DYN>>();
        assert_output::<
            <Tensor4<2, 3, 8, 9> as Conv2dOutput<Tensor4<5, 3, 3, 3>>>::Output,
            Tensor4<2, 5, DYN, DYN>,
        >();
        assert_output::<
            <Tensor3<DYN, 4, 6> as MatmulOutput<Tensor2<DYN, 7>>>::Output,
            Tensor3<DYN, 4, 7>,
        >();
        assert_output::<
            <Tensor3<2, 3, 4> as GatherOutput<Tensor3<DYN, 0, 8, i64>>>::Output,
            Tensor3<DYN, 0, 8>,
        >();
    }
}
