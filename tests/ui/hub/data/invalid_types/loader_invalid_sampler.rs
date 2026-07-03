use rstorch::data::{self, VecDataset};

fn main() {
    let dataset = VecDataset::new(vec![vec![1.0f32]]);
    let _ = data::loader(dataset).sampler(123usize);
}
