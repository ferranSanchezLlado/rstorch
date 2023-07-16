use rstorch::{sequential, Identity, Linear, ReLU, Sequential, Softmax};

#[test]
fn test_macro() {
    let _module = sequential!(
        Identity(),
        Linear(3, 10),
        ReLU(),
        Linear(10, 100),
        ReLU(),
        Linear(100, 2),
        Softmax()
    );

    let _module_2 = sequential!(
        Identity::new(),
        Linear::new(3, 10),
        ReLU::new(),
        Linear::new(10, 100),
        ReLU::new(),
        Linear::new(100, 2),
        Softmax::new(),
    );
}
