use polars::prelude::*;
use polars_io::csv::CsvReader;
use ndarray::*;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

pub fn read_csv() -> DataFrame {
    // Read a CSV file into a DataFrame
    let df: DataFrame = CsvReader::from_path("train.csv")
        .unwrap()
        .has_header(true)
        .finish() 
        .unwrap();

    // Print the DataFrame
    println!("DataFrame:\n{}", df);

    df
}

pub fn parse_data(df: DataFrame) -> (Array2<f32>, usize, usize) {
    let ndarray: Array2<f32> = df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

    let shape: &[usize] = ndarray.shape();
    let m: usize = shape[0];
    let n: usize = shape[1];

    return (ndarray, m, n)
}

pub fn swap_rows(array: &mut Array2<f32>, row1: usize, row2: usize) {
    let row1_data: ArrayView1<f32> = array.row(row1);
    let row2_data: ArrayView1<f32> = array.row(row2);

    // Create owned copies of the rows
    let row1_owned: Array1<f32> = row1_data.to_owned();
    let row2_owned: Array1<f32> = row2_data.to_owned();

    // Assign the rows
    array.row_mut(row1).assign(&row2_owned);
    array.row_mut(row2).assign(&row1_owned);
}

pub fn shuffle<T>(array: &mut Array2<f32>) {
    // Fisher-Yates shuffle
    let mut rng: ThreadRng = thread_rng();
    let n_rows = array.shape()[0];
    
    for i in (0..n_rows).rev() {
        let j = rng.gen_range(0..=i);
        swap_rows(array, i, j);
    }
}

pub fn split_data(ndarray: Array2<f32>) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array2<f32>, Array1<f32>, Array2<f32>, usize){
    // Split data into training and testing sets
    let _pre_t = &ndarray.slice(s![0..1000, ..]);
    let data_dev = _pre_t.t();
    let y_dev = &data_dev.slice(s![0, ..]);
    let x_dev = data_dev.slice(s![1.., ..]); // Remove the & before mut x_dev
    let x_dev = x_dev.mapv(|x: f32| x / 255.0);

    let _pre_t = &ndarray.slice(s![1000.., ..]);
    let data_train = _pre_t.t();
    let y_train = &data_train.slice(s![0, ..]);
    let x_train = data_train.slice(s![1.., ..]); 
    let x_train = x_train.mapv(|x: f32| x / 255.0);
    let m_train = x_train.shape()[1];   


    return (data_dev.to_owned(), y_dev.to_owned(), x_dev.to_owned(), data_train.to_owned(), y_train.to_owned(), x_train.to_owned(), m_train)
}
