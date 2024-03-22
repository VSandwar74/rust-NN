// use polars::prelude::*;
// use polars_io::csv::CsvReader;
// use ndarray::*;

// fn read_csv() -> DataFrame {
//     // Read a CSV file into a DataFrame
//     let df: DataFrame = CsvReader::from_path("example.csv")
//         .unwrap()
//         .has_header(true)
//         .finish() 
//         .unwrap();

//     // Print the DataFrame
//     println!("DataFrame:\n{}", df);

//     df
// }


// fn main() {
//     let df: DataFrame = read_csv();    
//     let _data = df.to_ndarray::<f64>(IndexOrder::Fortran).unwrap();
//     let _shape: (usize, usize) = df.shape(); 
// }


use polars::prelude::*;

fn main() {
    let a: Series = UInt32Chunked::new("a", &[1, 2, 3]).into_series();
    let b: Series = Float64Chunked::new("b", &[10., 8., 6.]).into_series();
    
    let df: DataFrame = DataFrame::new(vec![a, b]).unwrap();
    let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    println!("{:?}", ndarray);
}