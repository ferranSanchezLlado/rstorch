#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let dataset = Basic::new(vec![1_u8]);
    let _ = DataLoader::new(dataset).batch_size::<0>();
}
