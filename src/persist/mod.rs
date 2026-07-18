//! Persistence: safetensors read/write with the 16.14 behaviors — atomic
//! temp-and-rename saves, staged all-or-nothing restore, reader limits,
//! and a versioned envelope for non-tensor state.
