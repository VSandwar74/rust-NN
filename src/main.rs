mod util;
mod ml;

use crate::util::*;
use crate::ml::gradient_descent;
use polars::prelude::*;
use ndarray::*;


fn main() { 

    let df: DataFrame = read_csv();
    let (mut ndarray,
        _m, 
        _n
    ): (
        Array2<f32>, 
        usize, 
        usize
    ) = parse_data(df);

    shuffle::<f32>(&mut ndarray); 

    let (
        _data_dev,
        _y_dev,
        _x_dev,
        _data_train,
        y_train,
        x_train,
        _m_train,
    ): (
        Array2<f32>,
        Array1<f32>,
        Array2<f32>,
        Array2<f32>,
        Array1<f32>,
        Array2<f32>,
        usize,
    ) = split_data(ndarray);

    let alpha: f32 = 0.10;
    let iterations: i32 = 500;

    let (_w1, _b1, _w2, _b2) = gradient_descent(x_train, y_train, alpha, iterations);

    println!("w1: {:?}", _w1);
    println!("b1: {:?}", _b1);
    println!("w2: {:?}", _w2);
    println!("b2: {:?}", _b2);

}