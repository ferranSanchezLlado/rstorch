use rstorch::data::{SequentialSampler, StaticDataLoader, VecDataset};

fn iter_with_invalid_collate(
    loader: StaticDataLoader<VecDataset<Vec<f32>>, SequentialSampler, usize, 2>,
) {
    let _ = loader.iter();
}

fn main() {}
