use polars::prelude::*;
use polars_io::csv::CsvReader;
use ndarray::*;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

fn main() {
    let mut b1: Array1<f32> = Array1::<f32>::zeros(10);
    let mut w1: Array2<f32> = Array2::<f32>::zeros((10, 784));
    let mut w1: Array2<f32> = Array2::<f32>::zeros((784, 41000));

    for elem in &b1.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    for elem in &w1.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    for elem in &x.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }

    w1.dot(x) + b1.broadcast((10, x.shape()[1])).unwrap();
}