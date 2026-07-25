//! Op families (shape, elementwise, reductions, matmul, indexing, conv,
//! losses), one file per family per the implementation-plan §3 grid.

pub(crate) mod conv;
pub(crate) mod elementwise;
pub(crate) mod index;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod shape;
pub(crate) mod sugar;
