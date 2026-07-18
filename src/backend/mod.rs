//! Crate-private backend layer: the `BackendOps` trait, op enums, and the
//! single dispatch point (exploration §4.5). No public surface here; the
//! public face of a backend is the `Device` enum.

pub(crate) mod cpu;
