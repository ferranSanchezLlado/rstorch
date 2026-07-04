use rstorch::prelude::*;

fn main() {
    let guard = no_grad();
    std::thread::spawn(move || drop(guard));
}
