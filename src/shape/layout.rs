use super::Shape;
use crate::error::{Result, ShapeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    shape: Shape,
    strides: Box<[usize]>,
    offset: usize,
}

impl Layout {
    pub(crate) fn contiguous(shape: Shape) -> Result<Self> {
        let mut strides = vec![0; shape.rank()];
        let mut stride = 1usize;
        for (idx, &dim) in shape.dims().iter().enumerate().rev() {
            strides[idx] = stride;
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| ShapeError::NumelOverflow {
                    dims: shape.dims().into(),
                })?;
        }

        Ok(Self {
            shape,
            strides: strides.into_boxed_slice(),
            offset: 0,
        })
    }

    pub(crate) fn from_parts(
        shape: Shape,
        strides: impl Into<Box<[usize]>>,
        offset: usize,
    ) -> Result<Self> {
        let strides = strides.into();
        if strides.len() != shape.rank() {
            return Err(ShapeError::RankMismatch {
                expected: shape.rank(),
                found: strides.len(),
            }
            .into());
        }

        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    #[cfg(test)]
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[cfg(test)]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        let Ok(contiguous) = Self::contiguous(self.shape.clone()) else {
            return false;
        };
        self.offset == 0 && self.strides == contiguous.strides
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape
            .numel()
            .expect("constructed layouts have valid numel")
    }

    pub(crate) fn storage_positions(&self) -> Result<Vec<usize>> {
        let numel = self.shape.numel()?;
        if numel == 0 {
            return Ok(Vec::new());
        }

        let rank = self.shape.rank();
        if rank == 0 {
            return Ok(vec![self.offset]);
        }

        let mut positions = Vec::with_capacity(numel);
        for linear in 0..numel {
            let mut remaining = linear;
            let mut position = self.offset;
            for axis in (0..rank).rev() {
                let dim = self.shape.dims()[axis];
                let coord = if dim == 0 { 0 } else { remaining % dim };
                if let Some(next) = remaining.checked_div(dim) {
                    remaining = next;
                }
                let contribution = coord.checked_mul(self.strides[axis]).ok_or_else(|| {
                    ShapeError::NumelOverflow {
                        dims: self.shape.dims().into(),
                    }
                })?;
                position = position.checked_add(contribution).ok_or_else(|| {
                    ShapeError::NumelOverflow {
                        dims: self.shape.dims().into(),
                    }
                })?;
            }
            positions.push(position);
        }

        Ok(positions)
    }

    pub(crate) fn storage_span_len(&self) -> Result<usize> {
        let positions = self.storage_positions()?;
        let Some(max) = positions.into_iter().max() else {
            return Ok(0);
        };
        max.checked_add(1).ok_or_else(|| {
            ShapeError::NumelOverflow {
                dims: self.shape.dims().into(),
            }
            .into()
        })
    }

    pub(crate) fn validate_in_storage(&self, storage_len: usize) -> Result<()> {
        if self.numel() == 0 {
            return Ok(());
        }
        if self.offset >= storage_len {
            return Err(ShapeError::InvalidOffset {
                offset: self.offset,
                storage_len,
            }
            .into());
        }
        let span_len = self.storage_span_len()?;
        if span_len > storage_len {
            return Err(ShapeError::LayoutOutOfBounds {
                offset: span_len - 1,
                storage_len,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn reshape_contiguous(&self, shape: Shape) -> Result<Self> {
        let expected = shape.numel()?;
        if expected != self.numel() {
            return Err(ShapeError::LengthMismatch {
                expected,
                found: self.numel(),
            }
            .into());
        }
        if !self.is_contiguous() {
            return Err(ShapeError::ViewIncompatible {
                op: "reshape",
                reason: "input layout is not contiguous",
            }
            .into());
        }
        Self::contiguous(shape)
    }

    pub(crate) fn transpose2(&self) -> Result<Self> {
        if self.shape.rank() != 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                found: self.shape.rank(),
            }
            .into());
        }
        Self::from_parts(
            Shape::known([self.shape.dims()[1], self.shape.dims()[0]]),
            vec![self.strides[1], self.strides[0]],
            self.offset,
        )
    }

    pub(crate) fn transpose_axes(&self, lhs: usize, rhs: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if lhs >= rank || rhs >= rank {
            return Err(ShapeError::RankMismatch {
                expected: lhs.max(rhs) + 1,
                found: rank,
            }
            .into());
        }

        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.to_vec();
        dims.swap(lhs, rhs);
        strides.swap(lhs, rhs);
        Self::from_parts(Shape::known(dims), strides, self.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn layout_has_row_major_strides() {
        let layout = Layout::contiguous(Shape::known([2, 3, 4])).unwrap();
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.offset(), 0);
    }

    #[test]
    fn layout_validates_storage_bounds() {
        let layout = Layout::from_parts(Shape::known([2, 2]), vec![3, 1], 0).unwrap();
        assert!(layout.validate_in_storage(5).is_ok());

        let err = layout.validate_in_storage(4).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LayoutOutOfBounds { .. })
        ));

        let layout = Layout::from_parts(Shape::known([1]), vec![1], 2).unwrap();
        let err = layout.validate_in_storage(2).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::InvalidOffset { .. })
        ));
    }
}
