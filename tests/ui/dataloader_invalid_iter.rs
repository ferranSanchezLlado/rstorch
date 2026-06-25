use rstorch::data::{DataLoader, SequentialSampler, VecDataset};

fn iter_with_invalid_collate(loader: DataLoader<VecDataset<Vec<f32>>, SequentialSampler, usize>) {
    let _ = loader.iter();
}

fn main() {}
