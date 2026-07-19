//! CPU backend: the reference implementation every other backend is
//! conformance-tested against.
//!
//! Per-family kernel modules (ownership per implementation-plan §3):
//! `host` (T10a), `elementwise` (T10b), `reduce`/`matmul` (T11),
//! `index` (T25), `conv` (T26), `fused` (T48). T10b additionally owns
//! this file and `backend/parallel.rs` (the rayon switch).

pub(crate) mod conv;
pub(crate) mod elementwise;
pub(crate) mod fused;
pub(crate) mod host;
pub(crate) mod index;
pub(crate) mod matmul;
pub(crate) mod reduce;
