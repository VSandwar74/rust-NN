use polars::prelude::*;
use polars_io::csv::CsvReader;
use ndarray::*;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

fn read_csv() -> DataFrame {
    // Read a CSV file into a DataFrame
    let df: DataFrame = CsvReader::from_path("example.csv")
        .unwrap()
        .has_header(true)
        .finish() 
        .unwrap();

    // Print the DataFrame
    println!("DataFrame:\n{}", df);

    df
}

fn parse_data(df: DataFrame) -> (Array2<u32>, usize, usize) {
    let ndarray: Array2<u32> = df.to_ndarray::<UInt32Type>(IndexOrder::C).unwrap();

    let shape: &[usize] = ndarray.shape();
    let m: usize = shape[0];
    let n: usize = shape[1];

    return (ndarray, m, n)
}

pub fn shuffle<T>(row: &mut Array1<u32>) {
    // Fisher-Yates shuffle
    let mut rng: ThreadRng = thread_rng();
    for i in (1..row.len()).rev() {
        row.swap_axes(i, rng.gen_range(0..(i + 1)));
    }
}


fn main() {
    let df: DataFrame = read_csv();
    let (mut ndarray, m, n): (Array2<u32>, usize, usize) = parse_data(df);
    shuffle::<u32>(&mut ndarray.row_mut(0).to_owned());

    println!("Shape: ({}, {})", m, n);
    println!("Data: {:?}", ndarray);

}