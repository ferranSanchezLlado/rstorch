#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let _model = Sequential::new()
        .add_module(Linear::<3, 4>::new())
        .add_module(Linear::<5, 2>::new());
}
