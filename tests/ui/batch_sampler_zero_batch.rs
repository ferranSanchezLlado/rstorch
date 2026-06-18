#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let sampler = SequentialSampler::new(4);
    let _ = BatchSampler::<_, 0>::new(sampler);
}
