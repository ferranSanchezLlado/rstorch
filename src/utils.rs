use std::ops::{Div, Sub};

use ndarray::prelude::*;

pub fn normalize_zero_one<A, D, R>(x: Array<A, D>) -> Array<R, D>
where
    D: Dimension,
    A: Ord + Clone,
    R: From<A> + Clone + Div<Output = R> + Sub<Output = R>,
{
    // TODO: Find min and max in single loop
    let max: R = x.iter().max().unwrap().clone().into();
    let min: R = x.iter().min().unwrap().clone().into();

    let max_min = max - min.clone();

    x.mapv(|v| (R::from(v) - min.clone()) / max_min.clone())
}

pub fn normalize_zero_max<A, D>(x: Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: Ord + Clone + Sub<Output = A>,
{
    let min = x.iter().min().unwrap().clone();

    x.mapv(|v| v - min.clone())
}

pub fn flatten<A, D>(x: Array<A, D>) -> Array1<A>
where
    D: Dimension,
{
    x.into_iter().collect()
}

// assumes x value is normalized between 0 and n-1
pub fn one_hot<A: Into<usize>>(x: Array0<A>, n: usize) -> Array1<f64> {
    let mut encoded = Array1::zeros(n);
    encoded[x.into_scalar().into()] = 1.0;
    encoded
}

pub fn argmax(arr: Array2<f64>, axis: Axis) -> Option<Array1<usize>> {
    arr.axis_iter(axis)
        .map(|v| {
            v.into_iter()
                .enumerate()
                .reduce(|acc, e| if e.1 > acc.1 { e } else { acc })
                .map(|v| v.0)
        })
        .collect()
}

pub fn accuracy(pred: Array2<f64>, truth: Array2<f64>) -> f64 {
    let pred = argmax(pred, Axis(0)).unwrap();
    let truth = argmax(truth, Axis(0)).unwrap();

    let n = pred.len() as f64;
    pred.into_iter()
        .zip(truth)
        .map(|(a, b)| usize::from(a == b))
        .sum::<usize>() as f64
        / n
}
