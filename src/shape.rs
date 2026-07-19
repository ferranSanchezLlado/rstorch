//! Runtime shapes. Shapes are data, not types (exploration §2): a
//! [`Shape`] is a list of dimension sizes, axes are `isize` with negative
//! indexing, and every shape check is a loud structured error at runtime.

use crate::error::{Error, Result};

/// The dimensions of a tensor, outermost first (row-major convention).
///
/// Constructed via `impl Into<Shape>` conveniences in argument position:
/// arrays (`[64, 784]`), slices, `Vec<usize>`, and `()` for the rank-0
/// scalar shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// The dimension sizes, outermost first.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of axes. A scalar has rank 0.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (the product of all dimensions; 1 for a
    /// scalar shape).
    ///
    /// Note: tensor constructors validate this product against overflow at
    /// construction time (via the layout), so for any shape taken from an
    /// existing tensor this cannot overflow. For hand-built pathological
    /// shapes the product may wrap in release builds; do not use raw
    /// `Shape` values as a validation boundary.
    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    /// Resolve a possibly-negative axis into `[0, rank)`.
    ///
    /// `-1` is the last axis, `-rank` the first; anything outside
    /// `[-rank, rank)` is [`Error::InvalidAxis`] carrying `op`.
    pub fn resolve_axis(&self, axis: isize, op: &'static str) -> Result<usize> {
        let rank = self.rank() as isize;
        let resolved = if axis < 0 { axis + rank } else { axis };
        if resolved < 0 || resolved >= rank {
            return Err(Error::InvalidAxis {
                op,
                axis,
                rank: self.rank(),
            });
        }
        Ok(resolved as usize)
    }

    /// Resolve a possibly-negative *insertion* axis into `[0, rank]` — the
    /// variant axis-inserting ops (`unsqueeze`, `stack`) need, where
    /// `rank` itself (append position) is valid and negative axes count
    /// from the end of the *output* rank: valid inputs are
    /// `[-rank-1, rank]`.
    pub fn resolve_insert_axis(&self, axis: isize, op: &'static str) -> Result<usize> {
        let out_rank = self.rank() as isize + 1;
        let resolved = if axis < 0 { axis + out_rank } else { axis };
        if resolved < 0 || resolved >= out_rank {
            return Err(Error::InvalidAxis {
                op,
                axis,
                rank: self.rank(),
            });
        }
        Ok(resolved as usize)
    }

    /// Compute the NumPy/PyTorch broadcast of two shapes (exploration
    /// §4.2): align ranks to the right, pad the shorter with 1s; each axis
    /// pair must be equal or contain a 1, producing the max.
    ///
    /// Returns [`Error::ShapeMismatch`] carrying `op` when incompatible.
    pub fn broadcast_with(&self, other: &Shape, op: &'static str) -> Result<Shape> {
        let (a, b) = (self.dims(), other.dims());
        let rank = a.len().max(b.len());
        let mut out = vec![0usize; rank];
        for i in 0..rank {
            // Right-aligned: axis `rank-1-i` of the output.
            let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
            out[rank - 1 - i] = if da == db {
                da
            } else if da == 1 {
                db
            } else if db == 1 {
                da
            } else {
                return Err(Error::ShapeMismatch {
                    op,
                    lhs: self.clone(),
                    rhs: other.clone(),
                });
            };
        }
        Ok(Shape(out))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        shape.clone()
    }
}

/// The rank-0 scalar shape.
impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape(Vec::new())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(Shape::from([2, 3]).dims(), &[2, 3]);
        assert_eq!(Shape::from(vec![4]).dims(), &[4]);
        assert_eq!(Shape::from(&[5, 6][..]).dims(), &[5, 6]);
        let scalar = Shape::from(());
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.num_elements(), 1);
    }

    #[test]
    fn axis_resolution() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.resolve_axis(0, "t").unwrap(), 0);
        assert_eq!(s.resolve_axis(2, "t").unwrap(), 2);
        assert_eq!(s.resolve_axis(-1, "t").unwrap(), 2);
        assert_eq!(s.resolve_axis(-3, "t").unwrap(), 0);
        assert!(matches!(
            s.resolve_axis(3, "t"),
            Err(Error::InvalidAxis {
                op: "t",
                axis: 3,
                rank: 3
            })
        ));
        assert!(matches!(
            s.resolve_axis(-4, "t"),
            Err(Error::InvalidAxis { .. })
        ));
    }

    #[test]
    fn insert_axis_resolution() {
        let s = Shape::from([2, 3]);
        assert_eq!(s.resolve_insert_axis(2, "unsqueeze").unwrap(), 2);
        assert_eq!(s.resolve_insert_axis(-1, "unsqueeze").unwrap(), 2);
        assert_eq!(s.resolve_insert_axis(-3, "unsqueeze").unwrap(), 0);
        assert!(s.resolve_insert_axis(3, "unsqueeze").is_err());
        assert!(s.resolve_insert_axis(-4, "unsqueeze").is_err());
    }

    #[test]
    fn broadcasting() {
        let a = Shape::from([8, 1, 6, 1]);
        let b = Shape::from([7, 1, 5]);
        assert_eq!(a.broadcast_with(&b, "add").unwrap().dims(), &[8, 7, 6, 5]);

        let a = Shape::from([5, 4]);
        let b = Shape::from([1]);
        assert_eq!(a.broadcast_with(&b, "add").unwrap().dims(), &[5, 4]);

        let scalar = Shape::from(());
        assert_eq!(a.broadcast_with(&scalar, "add").unwrap().dims(), &[5, 4]);

        let a = Shape::from([3]);
        let b = Shape::from([4]);
        assert!(matches!(
            a.broadcast_with(&b, "add"),
            Err(Error::ShapeMismatch { op: "add", .. })
        ));
    }

    #[test]
    fn display() {
        assert_eq!(Shape::from([2, 3]).to_string(), "[2, 3]");
        assert_eq!(Shape::from(()).to_string(), "[]");
    }
}
