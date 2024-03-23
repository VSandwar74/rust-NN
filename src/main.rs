use polars::prelude::*;
use polars_io::csv::CsvReader;
use ndarray::*;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

fn read_csv() -> DataFrame {
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

fn parse_data(df: DataFrame) -> (Array2<u32>, usize, usize) {
    let ndarray: Array2<u32> = df.to_ndarray::<UInt32Type>(IndexOrder::C).unwrap();

    let shape: &[usize] = ndarray.shape();
    let m: usize = shape[0];
    let n: usize = shape[1];

    return (ndarray, m, n)
}

fn swap_rows(array: &mut Array2<u32>, row1: usize, row2: usize) {
    let row1_data: ArrayView1<u32> = array.row(row1);
    let row2_data: ArrayView1<u32> = array.row(row2);

    // Create owned copies of the rows
    let row1_owned: Array1<u32> = row1_data.to_owned();
    let row2_owned: Array1<u32> = row2_data.to_owned();

    // Assign the rows
    array.row_mut(row1).assign(&row2_owned);
    array.row_mut(row2).assign(&row1_owned);
}

fn shuffle<T>(array: &mut Array2<u32>) {
    // Fisher-Yates shuffle
    let mut rng: ThreadRng = thread_rng();
    let n_rows = array.shape()[0];
    
    for i in (0..n_rows).rev() {
        let j = rng.gen_range(0..=i);
        swap_rows(array, i, j);
        // row.swap(i, rng.gen_range(0..=(i + 1)));
        // row.swap_axes(i, rng.gen_range(0..(i + 1)));
    }
}

// fn init_params() -> (f32, f32, f32, f32) {
//     // Hyper params
//     let w1: f32 = 392.1; // 10 - 784
//     let b1: f32 = 5.0; // 10 - 1
//     let w2: f32 = 10.0; // 10 - 10
//     let b2: f32 = 4.9; // 10 - 1
    
//     (w1, b1, w2, b2)
// }

// fn re_lu(mut z: &Array2<f32>) -> Array2<f32> {
//     return z.mapv(|x| x.max(0.0));
// }

// fn softmax(mut z: &Array2<f32>) -> Array2<f32> {
//     let tot = z.mapv(|x| x.exp()).sum();
//     let a: Array2<f32> = z.mapv(|x| x.exp() / tot);
//     return a
// }

// fn forward_prop(w1: Array2<f32>, b1: Array2<f32>, w2: Array2<f32>, b2: Array2<f32>, x: Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
//     let z1: Array2<f32> = w1.dot(&x) + &b1;
//     let a1: Array2<f32> = re_lu(&z1);
//     let z2: Array2<f32> = w2.dot(&a1) + &b2;
//     let a2: Array2<f32> = softmax(&z2);
//     return (z1, a1, z2, a2)
// }

// fn re_lu_deriv(Z: Array2<f32>) -> Array2<f32> {
//     return Z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
// }

// fn one_hot(Y: Array1<f32>) -> Array2<f32> {
//     // Find the maximum value in Y
//     let max_value = Y.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

//     // Create a one-hot encoded matrix with the appropriate dimensions
//     let mut one_hot_y: Array2<f32> = Array2::<f32>::zeros((Y.len(), max_value as usize + 1));

//     // Populate the one-hot encoded matrix
//     for i in 0..Y.len() {
//         one_hot_y[[i, Y[i] as usize]] = 1.0;
//     }

//     one_hot_y.t().to_owned()
// }

// fn backward_prop(
//     z1: &Array2<f32>,
//     a1: &Array2<f32>,
//     // z2: &Array2<f32>,
//     a2: &Array2<f32>,
//     // w1: &Array2<f32>,
//     w2: &Array2<f32>,
//     x: &Array2<f32>,
//     y: &Array1<f32>,
//     m: usize,
// ) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
//     let one_hot_y = one_hot(y.clone());
//     let dz2 = a2 - &one_hot_y;
//     let dw2 = 1.0 / m as f32 * dz2.dot(&a1.t());
//     let db2 = 1.0 / m as f32 * dz2.sum_axis(Axis(1));
//     let dz1 = w2.t().dot(&dz2) * re_lu_deriv(z1.clone());
//     let dw1 = 1.0 / m as f32 * dz1.dot(&x.t());
//     let db1 = 1.0 / m as f32 * dz1.sum_axis(Axis(1));
//     (dw1, db1, dw2, db2)
// }

// fn update_params(mut w1: u32, mut b1: u32, mut w2: u32, mut b2: u32, dw1: u32, db1: u32, dw2: u32, db2: u32, alpha: u32) -> (u32, u32, u32, u32){
//     w1 = w1 - alpha * dw1;
//     b1 = b1 - alpha * db1;
//     w2 = w2 - alpha * dw2;
//     b2 = b2 - alpha * db2;
//     return (w1, b1, w2, b2)
// }

// fn _split_data(ndarray: Array2<u32>) -> (Array2<u32>, Array1<u32>, Array2<u32>, Array2<u32>, Array1<u32>, Array2<u32>, usize){
//     // Split data into training and testing sets
//     let _pre_t = &ndarray.slice(s![0..1000, ..]);
//     let data_dev = _pre_t.t();
//     let y_dev = &data_dev.slice(s![0, ..]);
//     let x_dev = &data_dev.slice(s![1.., ..]);

//     let _pre_t = &ndarray.slice(s![1000.., ..]);
//     let data_train = _pre_t.t();
//     let y_train = &data_train.slice(s![0, ..]);
//     let x_train = &data_train.slice(s![1.., ..]);
//     let m_train = x_train.shape()[1];   


//     return (data_dev.to_owned(), y_dev.to_owned(), x_dev.to_owned(), data_train.to_owned(), y_train.to_owned(), x_train.to_owned(), m_train)
// }

// // def get_predictions(A2):
// //     return np.argmax(A2, 0)

// // def get_accuracy(predictions, Y):
// //     print(predictions, Y)
// //     return np.sum(predictions == Y) / Y.size

// // def gradient_descent(X, Y, alpha, iterations):
// //     W1, b1, W2, b2 = init_params()
// //     for i in range(iterations):
// //         Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
// //         dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
// //         W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
// //         if i % 10 == 0:
// //             print("Iteration: ", i)
// //             predictions = get_predictions(A2)
// //             print(get_accuracy(predictions, Y))
// //     return W1, b1, W2, b2

fn main() {
    let df: DataFrame = read_csv();

    let (mut ndarray, m, n): (Array2<u32>, usize, usize) = parse_data(df);
        
    shuffle::<u32>(&mut ndarray);
    
    println!("ndarray: {:?}", ndarray);

    
    // // let (data_dev, y_dev, x_dev, data_train, y_train, x_train, m_train):
    // //     (Array2<u32>, Array1<u32>, Array2<u32>, Array2<u32>, Array1<u32>, Array2<u32>, usize) = _split_data(ndarray);
    
    
    println!("Shape: ({}, {})", m, n);
    // println!("Data: {:?}", ndarray);

}