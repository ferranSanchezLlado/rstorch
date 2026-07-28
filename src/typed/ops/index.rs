//! Typed index construction and indexed reads.

use super::{GatherOutput, IndexSelectOutput, ReplaceAxisOutput};
use crate::typed::const_check::assert_gather_dimension;
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{DYN, DeviceBinding, DeviceCtx, Placement, Tensor1, Tensor2, TypedTensor};
use crate::{Element, Error, Result, Shape, Tensor};
use std::sync::Arc;

fn validate_operand<P: Placement>(
    source: &Arc<DeviceBinding>,
    indices: &Arc<DeviceBinding>,
    op: &'static str,
) -> Result<()> {
    validate_binding::<P>(source, op)?;
    validate_binding::<P>(indices, op)?;
    if !Arc::ptr_eq(source, indices) {
        return Err(Error::InvalidArg {
            op,
            msg: "operands do not share the canonical placement binding".into(),
        });
    }
    Ok(())
}

const fn assert_gather_geometry(source: &[usize], indices: &[usize], axis: usize) {
    let mut current = 0;
    while current < source.len() {
        if current != axis {
            assert_gather_dimension(source[current], indices[current]);
        }
        current += 1;
    }
}

impl<const LEN: usize, P: Placement> Tensor1<LEN, i64, P> {
    /// Constructs the half-open unit-step index range `[start, end)`.
    ///
    /// The resulting runtime length must equal `LEN`, unless `LEN` is [`DYN`].
    pub fn arange(start: i64, end: i64, ctx: &DeviceCtx<P>) -> Result<Self> {
        validate_binding::<P>(ctx.binding(), "arange")?;
        let count = (i128::from(end) - i128::from(start)).max(0);
        let count = usize::try_from(count).map_err(|_| Error::InvalidArg {
            op: "arange",
            msg: format!("range {start}..{end} has too many elements"),
        })?;
        if LEN != DYN && LEN != count {
            return Err(Error::ShapeMismatch {
                op: "arange",
                lhs: Shape::from([count]),
                rhs: Shape::from([LEN]),
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::InvalidArg {
                op: "arange",
                msg: format!("range {start}..{end} has too many elements"),
            })?;
        values.extend(start..end);
        let tensor = Tensor::from_vec(values, [count], &ctx.device())?;
        checked_wrap(tensor, Arc::clone(ctx.binding()), "arange")
    }

    /// Constructs an index tensor from exact host `i64` values.
    ///
    /// The number of values must equal `LEN`, unless `LEN` is [`DYN`].
    pub fn from_indices(values: Vec<i64>, ctx: &DeviceCtx<P>) -> Result<Self> {
        validate_binding::<P>(ctx.binding(), "from_indices")?;
        let len = values.len();
        let tensor = Tensor::from_vec(values, [len], &ctx.device())?;
        checked_wrap(tensor, Arc::clone(ctx.binding()), "from_indices")
    }
}

impl<const S: usize, P: Placement> Tensor2<S, S, bool, P> {
    /// Constructs a square causal mask, with `true` above the diagonal.
    ///
    /// `seq_len` must equal static `S`; [`DYN`] retains the supplied size.
    pub fn causal_mask(seq_len: usize, ctx: &DeviceCtx<P>) -> Result<Self> {
        validate_binding::<P>(ctx.binding(), "causal_mask")?;
        let tensor = Tensor::causal_mask(seq_len, &ctx.device())?;
        checked_wrap(tensor, Arc::clone(ctx.binding()), "causal_mask")
    }
}

macro_rules! impl_typed_indexing {
    ($(($name:ident, [$($dim:ident),+])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)+ E: Element, P: Placement>
                crate::typed::$name<$($dim,)+ E, P>
            {
                /// Selects whole slices at a typed one-dimensional index set.
                ///
                /// `AXIS` must name an axis of this rank. The selected source
                /// marker is replaced by the index tensor's `LEN` marker.
                pub fn index_select<const AXIS: usize, const LEN: usize>(
                    &self,
                    indices: &Tensor1<LEN, i64, P>,
                ) -> Result<<Self as IndexSelectOutput<AXIS, Tensor1<LEN, i64, P>>>::Output>
                where
                    Self: IndexSelectOutput<AXIS, Tensor1<LEN, i64, P>>,
                    <Self as IndexSelectOutput<AXIS, Tensor1<LEN, i64, P>>>::Output:
                        TypedTensor<Elem = E, Placement = P>,
                {
                    validate_operand::<P>(self.binding(), indices.binding(), "index_select")?;
                    let tensor = self
                        .dynamic()
                        .index_select(AXIS as isize, indices.dynamic())?;
                    checked_wrap(tensor, Arc::clone(self.binding()), "index_select")
                }

                /// Gathers elements through a same-rank typed index grid.
                ///
                /// The output has the index grid's geometry and the source's
                /// element type. Known non-selected dimensions must match;
                /// relationships involving [`DYN`] defer to the runtime op.
                pub fn gather<const AXIS: usize, Indices>(
                    &self,
                    indices: &Indices,
                ) -> Result<<Self as GatherOutput<Indices>>::Output>
                where
                    Self: GatherOutput<Indices> + ReplaceAxisOutput<AXIS, DYN>,
                    Indices: TypedTensor<Elem = i64, Placement = P>,
                    <Self as GatherOutput<Indices>>::Output:
                        TypedTensor<Elem = E, Placement = P>,
                {
                    const {
                        assert_gather_geometry(Self::MARKERS, Indices::MARKERS, AXIS);
                    }
                    validate_operand::<P>(self.binding(), indices.binding(), "gather")?;
                    let tensor = self.dynamic().gather(AXIS as isize, indices.dynamic())?;
                    checked_wrap(tensor, Arc::clone(self.binding()), "gather")
                }
            }
        )+
    };
}

impl_typed_indexing! {
    (Tensor1, [D0]),
    (Tensor2, [D0, D1]),
    (Tensor3, [D0, D1, D2]),
    (Tensor4, [D0, D1, D2, D3]),
    (Tensor5, [D0, D1, D2, D3, D4]),
    (Tensor6, [D0, D1, D2, D3, D4, D5]),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6]),
    (Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::typed::{Cpu, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7, Tensor8};

    fn cpu() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    fn weighted_sum(tensor: &Tensor) -> Tensor {
        let weights = (0..tensor.num_elements())
            .map(|index| 0.25 + index as f32 * 0.5)
            .collect::<Vec<_>>();
        tensor
            .mul(&Tensor::from_vec(weights, tensor.dims().to_vec(), &Device::Cpu).unwrap())
            .unwrap()
            .sum_all()
            .unwrap()
    }

    #[test]
    fn constructors_and_causal_mask_match_runtime_values() {
        let ctx = cpu();
        let range = Tensor1::<3, i64>::arange(2, 5, &ctx).unwrap();
        assert_eq!(range.as_dynamic().to_vec::<i64>().unwrap(), vec![2, 3, 4]);

        let values = Tensor1::<3, i64>::from_indices(vec![4, -1, 4], &ctx).unwrap();
        assert_eq!(values.as_dynamic().to_vec::<i64>().unwrap(), vec![4, -1, 4]);

        let typed = Tensor2::<3, 3, bool>::causal_mask(3, &ctx).unwrap();
        let dynamic = Tensor::causal_mask(3, &Device::Cpu).unwrap();
        assert_eq!(typed.dims(), [3, 3]);
        assert_eq!(
            typed.as_dynamic().to_vec::<bool>().unwrap(),
            dynamic.to_vec::<bool>().unwrap()
        );

        let dynamic = Tensor2::<DYN, DYN, bool>::causal_mask(2, &ctx).unwrap();
        assert_eq!(dynamic.dims(), [2, 2]);
    }

    #[test]
    fn static_constructor_geometry_errors_are_structured() {
        let ctx = cpu();
        assert!(matches!(
            Tensor1::<2, i64>::arange(0, 3, &ctx),
            Err(Error::ShapeMismatch { op: "arange", .. })
        ));
        assert!(matches!(
            Tensor1::<2, i64>::from_indices(vec![0], &ctx),
            Err(Error::ShapeMismatch {
                op: "from_indices",
                ..
            })
        ));
        assert!(matches!(
            Tensor2::<2, 2, bool>::causal_mask(3, &ctx),
            Err(Error::ShapeMismatch {
                op: "causal_mask",
                ..
            })
        ));
    }

    #[test]
    fn index_select_and_gather_match_dynamic_and_preserve_geometry() {
        let ctx = cpu();
        let source = Tensor2::<2, 3>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3], &ctx).unwrap();
        let selected_ids = Tensor1::<2, i64>::from_indices(vec![2, 0], &ctx).unwrap();
        let selected: Tensor2<2, 2> = source.index_select::<1, 2>(&selected_ids).unwrap();
        let expected = source
            .as_dynamic()
            .index_select(1, selected_ids.as_dynamic())
            .unwrap();
        assert_eq!(selected.dims(), [2, 2]);
        assert_eq!(
            selected.as_dynamic().to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );

        let gather_ids = Tensor2::<2, 1, i64>::from_vec(vec![2, 0], [2, 1], &ctx).unwrap();
        let gathered: Tensor2<2, 1> = source.gather::<1, _>(&gather_ids).unwrap();
        let expected = source
            .as_dynamic()
            .gather(1, gather_ids.as_dynamic())
            .unwrap();
        assert_eq!(gathered.dims(), [2, 1]);
        assert_eq!(
            gathered.as_dynamic().to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn arange_preserves_i64_values_above_f64_integer_precision() {
        let ctx = cpu();
        let start = (1i64 << 53) + 1;
        let range = Tensor1::<3, i64>::arange(start, start + 3, &ctx).unwrap();
        assert_eq!(
            range.as_dynamic().to_vec::<i64>().unwrap(),
            vec![start, start + 1, start + 2]
        );
    }

    #[test]
    fn arange_rejects_static_length_mismatch_before_allocation() {
        let ctx = cpu();
        assert!(matches!(
            Tensor1::<1, i64>::arange(i64::MIN, i64::MAX, &ctx),
            Err(Error::ShapeMismatch { op: "arange", .. })
        ));
    }

    #[test]
    fn index_values_bounds_and_deferred_geometry_are_runtime_errors() {
        let ctx = cpu();
        let source = Tensor2::<DYN, 3>::from_vec(vec![1.; 6], [2, 3], &ctx).unwrap();
        let bad_select = Tensor1::<1, i64>::from_indices(vec![2], &ctx).unwrap();
        assert!(matches!(
            source.index_select::<0, 1>(&bad_select),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: 2,
                axis: 0,
                size: 2
            })
        ));

        let bad_gather = Tensor2::<DYN, 1, i64>::from_vec(vec![3, 0], [2, 1], &ctx).unwrap();
        assert!(matches!(
            source.gather::<1, _>(&bad_gather),
            Err(Error::IndexOutOfBounds {
                op: "gather",
                index: 3,
                axis: 1,
                size: 3
            })
        ));

        let too_tall = Tensor2::<DYN, 1, i64>::from_vec(vec![0; 3], [3, 1], &ctx).unwrap();
        assert!(matches!(
            source.gather::<1, _>(&too_tall),
            Err(Error::ShapeMismatch { op: "gather", .. })
        ));
    }

    #[test]
    fn indexed_reads_retain_runtime_gradients() {
        let ctx = cpu();
        let leaf = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], [3], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let source = Tensor1::<3>::try_from_dynamic(leaf.clone(), &ctx).unwrap();
        let ids = Tensor1::<3, i64>::from_indices(vec![1, 1, 2], &ctx).unwrap();
        let loss = weighted_sum(source.index_select::<0, 3>(&ids).unwrap().as_dynamic());
        assert_eq!(
            loss.backward()
                .unwrap()
                .wrt_input(&leaf)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0, 1.0, 1.25]
        );

        let leaf = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [1, 3], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let source = Tensor2::<1, 3>::try_from_dynamic(leaf.clone(), &ctx).unwrap();
        let ids = Tensor2::<1, 3, i64>::from_vec(vec![2, 0, 2], [1, 3], &ctx).unwrap();
        let loss = weighted_sum(source.gather::<1, _>(&ids).unwrap().as_dynamic());
        assert_eq!(
            loss.backward()
                .unwrap()
                .wrt_input(&leaf)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.75, 0.0, 1.5]
        );
    }

    #[test]
    fn indexed_methods_cover_every_supported_rank() {
        let ctx = cpu();
        let one = Tensor1::<1, i64>::from_indices(vec![0], &ctx).unwrap();

        macro_rules! rank_case {
            ($ty:ty, $dims:expr, $indices:expr) => {{
                let source = <$ty>::from_vec(vec![7.0f32], $dims, &ctx).unwrap();
                assert_eq!(source.index_select::<0, 1>(&one).unwrap().dims(), $dims);
                let indices = $indices;
                assert_eq!(source.gather::<0, _>(&indices).unwrap().dims(), $dims);
            }};
        }

        rank_case!(
            Tensor1<1>,
            [1],
            Tensor1::<1, i64>::from_indices(vec![0], &ctx).unwrap()
        );
        rank_case!(Tensor2<1, 1>, [1, 1], Tensor2::<1, 1, i64>::from_vec(vec![0], [1, 1], &ctx).unwrap());
        rank_case!(Tensor3<1, 1, 1>, [1, 1, 1], Tensor3::<1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1], &ctx).unwrap());
        rank_case!(Tensor4<1, 1, 1, 1>, [1, 1, 1, 1], Tensor4::<1, 1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1, 1], &ctx).unwrap());
        rank_case!(Tensor5<1, 1, 1, 1, 1>, [1, 1, 1, 1, 1], Tensor5::<1, 1, 1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1, 1, 1], &ctx).unwrap());
        rank_case!(Tensor6<1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1], Tensor6::<1, 1, 1, 1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1, 1, 1, 1], &ctx).unwrap());
        rank_case!(Tensor7<1, 1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1, 1], Tensor7::<1, 1, 1, 1, 1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1, 1, 1, 1, 1], &ctx).unwrap());
        rank_case!(Tensor8<1, 1, 1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1, 1, 1], Tensor8::<1, 1, 1, 1, 1, 1, 1, 1, i64>::from_vec(vec![0], [1, 1, 1, 1, 1, 1, 1, 1], &ctx).unwrap());
    }

    #[test]
    fn output_contracts_encode_selected_and_index_geometry() {
        fn selected<T: IndexSelectOutput<1, Tensor1<4, i64>, Output = Tensor2<2, 4>>>() {}
        fn gathered<T: GatherOutput<Tensor2<1, 2, i64>, Output = Tensor2<1, 2>>>() {}
        selected::<Tensor2<2, 3>>();
        gathered::<Tensor2<2, 3>>();
    }
}
