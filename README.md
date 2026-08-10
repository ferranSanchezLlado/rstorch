# RsTorch

**This branch (`restart-v3`) is a from-zero rebuild in progress.** The v0.x
published library has been removed from the tree; rstorch is being rebuilt
around one concrete tensor type with zero generic parameters and linear
gradients — a safer PyTorch-inspired design that spends Rust's type system on
*state* (ownership, linearity) rather than *shapes*.

The last published v0.x release remains available on
[crates.io](https://crates.io/crates/rstorch); its source is on the `master`
branch history (`ce27c8c` and earlier).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
