#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let dataset = Basic::new(vec![([1.0_f32], 0_u8)]);
    let loader = DataLoader::new(dataset)
        .collate::<OneHotClassification<1, 0>>()
        .batch_size::<1>();
    let _ = loader.iter();
}
