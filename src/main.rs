use polars::prelude::*;
use polars_io::csv::CsvReader;
use ndarray::*;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

fn argmax(matrix: Array2<f32>) -> Array1<f32> {
    let mut output = Array1::zeros(matrix.shape()[1]);
    for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
        let (max_idx, max_val) =
            row.iter()
                .enumerate()
                .fold((0, row[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        output[i] = max_idx as f32;
    }
    output
}


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

fn parse_data(df: DataFrame) -> (Array2<f32>, usize, usize) {
    let ndarray: Array2<f32> = df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

    let shape: &[usize] = ndarray.shape();
    let m: usize = shape[0];
    let n: usize = shape[1];

    return (ndarray, m, n)
}

fn swap_rows(array: &mut Array2<f32>, row1: usize, row2: usize) {
    let row1_data: ArrayView1<f32> = array.row(row1);
    let row2_data: ArrayView1<f32> = array.row(row2);

    // Create owned copies of the rows
    let row1_owned: Array1<f32> = row1_data.to_owned();
    let row2_owned: Array1<f32> = row2_data.to_owned();

    // Assign the rows
    array.row_mut(row1).assign(&row2_owned);
    array.row_mut(row2).assign(&row1_owned);
}

fn shuffle<T>(array: &mut Array2<f32>) {
    // Fisher-Yates shuffle
    let mut rng: ThreadRng = thread_rng();
    let n_rows = array.shape()[0];
    
    for i in (0..n_rows).rev() {
        let j = rng.gen_range(0..=i);
        swap_rows(array, i, j);
    }
}

fn split_data(ndarray: Array2<f32>) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array2<f32>, Array1<f32>, Array2<f32>, usize){
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

fn init_params() -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let mut rng = thread_rng();

    // Initialize arrays with zeros
    let mut w1: Array2<f32> = Array2::<f32>::zeros((10, 784));
    let mut b1: Array1<f32> = Array1::<f32>::zeros(10);
    let mut w2: Array2<f32> = Array2::<f32>::zeros((10, 10));
    let mut b2: Array1<f32> = Array1::<f32>::zeros(10);

    // Fill arrays with random values
    for elem in w1.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    for elem in b1.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    for elem in w2.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    for elem in b2.iter_mut() {
        *elem = rng.gen::<f32>() - 0.5;
    }
    // println!("w1: {:?}", w1);
    // println!("b1: {:?}", b1);
    // println!("w2: {:?}", w2);
    // println!("b2: {:?}", b2);
    (w1, b1, w2, b2)
}

fn re_lu(mut z: &Array2<f32>) -> Array2<f32> {
    return z.mapv(|x| x.max(0.0));
}

// TO DO: Fix the softmax function

fn softmax(mut z: &Array2<f32>) -> Array2<f32> {
    let tot = z.mapv(|x| x.exp());
    // println!("{:?}", tot);
    // let a: Array2<f32> = z.mapv(|x| x.exp() / tot);
    return tot;
}

fn forward_prop(w1: &Array2<f32>, b1: &Array1<f32>, w2: &Array2<f32>, b2: &Array1<f32>, x: &mut Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let z1: Array2<f32> = w1.dot(x) + b1.broadcast((x.shape()[1], 10)).unwrap().t();
    let a1: Array2<f32> = re_lu(&z1);
    let z2: Array2<f32> = w2.dot(&a1) + b2.broadcast((a1.shape()[1], 10)).unwrap().t();
    let a2: Array2<f32> = softmax(&z2);
    // println!("x: {:?}", x);
    // println!("w1: {:?}", w1);
    // println!("x_1: {:?}", x.slice_axis(Axis(1), Slice::new(0,Some(1),1)));
    // println!("w1_1: {:?}", w1.slice_axis(Axis(0), Slice::new(0,Some(1),1)));
    // let ax = w1.slice_axis(Axis(0), Slice::new(0,Some(1),1)).to_owned().dot(&x.slice_axis(Axis(1), Slice::new(0,Some(1),1)).to_owned());
    // println!("dot: {:?}",ax);
    // println!("w1: {:?}", w1.dot(x));
    // println!("b1: {:?}", b1.broadcast((x.shape()[1], 10)).unwrap().t());
    // println!("z1: {:?}", z1);
    // println!("a1: {:?}", a1);
    // println!("z2: {:?}", z2);
    // println!("a2: {:?}", a2);
    return (z1, a1, z2, a2)
}

fn re_lu_deriv(Z: Array2<f32>) -> Array2<f32> {
    return Z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
}

fn one_hot(Y: Array1<f32>) -> Array2<f32> {
    // Find the maximum value in Y
    let max_value = Y.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    // Create a one-hot encoded matrix with the appropriate dimensions
    let mut one_hot_y: Array2<f32> = Array2::<f32>::zeros((Y.len(), max_value as usize + 1));

    // Populate the one-hot encoded matrix
    for i in 0..Y.len() {
        one_hot_y[[i, Y[i] as usize]] = 1.0;
    }

    one_hot_y.t().to_owned()
}

fn backward_prop(
    z1: &Array2<f32>,
    a1: &Array2<f32>,
    // z2: &Array2<f32>,
    a2: &Array2<f32>,
    // w1: &Array2<f32>,
    w2: &Array2<f32>,
    x: &Array2<f32>,
    y: &Array1<f32>,
    m: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let one_hot_y = one_hot(y.clone());
    let dz2 = a2 - &one_hot_y;
    let dw2 = 1.0 / m as f32 * dz2.dot(&a1.t());
    let db2 = 1.0 / m as f32 * dz2.sum_axis(Axis(1));
    let dz1 = w2.t().dot(&dz2) * re_lu_deriv(z1.clone());
    let dw1 = 1.0 / m as f32 * dz1.dot(&x.t());
    let db1 = 1.0 / m as f32 * dz1.sum_axis(Axis(1));
    (dw1, db1, dw2, db2)
}
fn update_params(
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    dw1: Array2<f32>,
    db1: Array1<f32>,
    dw2: Array2<f32>,
    db2: Array1<f32>,
    alpha: f32,
) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let updated_w1 = &w1 - &(alpha * &dw1);
    let updated_b1 = &b1 - &(alpha * &db1);
    let updated_w2 = &w2 - &(alpha * &dw2);
    let updated_b2 = &b2 - &(alpha * &db2);
    (updated_w1, updated_b1, updated_w2, updated_b2)
}

fn get_predictions(a2: Array2<f32>) -> Array1<f32> {
    println!("{:?}", argmax(a2));
    return argmax(a2);
}

fn get_accuracy(predictions: Array1<f32>, y: &Array1<f32>) -> f32 {
    let mut correct = 0;
    for i in 0..y.len() {
        if predictions[i] == y[i] {
            correct += 1;
        }
    }
    return correct as f32 / y.len() as f32
}

fn gradient_descent(mut x: Array2<f32>, y: Array1<f32>, alpha: f32, iterations: i32) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>){
    let (mut w1, mut b1, mut w2, mut b2) = init_params();
    for i in 0..iterations {
        println!("Iteration: {}", i);
        let (z1, a1, z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &mut x);
        // println!("z1 {:?}", z1);
        // println!("z2 {:?}", z2);
        // println!("a1 {:?}", a1);
        // println!("a2 {:?}", a2);
        let (dw1, db1, dw2, db2) = backward_prop(&z1, &a1, &a2, &w2, &x, &y, x.shape()[1]);
        (w1, b1, w2, b2) = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha);
        if i % 10 == 0 {
            println!("Iteration: {}", i);
            let predictions = get_predictions(a2);
            println!("{}", get_accuracy(predictions, &y));
        }
    }
    return (w1, b1, w2, b2)

}

fn main() { 
    // Example matrices
    let w1: Array2<f32> = Array::zeros((10, 784));
    let x: Array2<f32> = Array::zeros((784, 41000));
    let b1: Array2<f32> = Array::ones((10, 1)); // Initialize b1 as a matrix of ones

    // Broadcast b1 to match the shape of x along the second axis (column dimension)
    let b1_broadcasted = b1.broadcast((10, 41000)).unwrap();

    // Perform element-wise addition
    let z1: Array2<f32> = w1.dot(&x) + &b1_broadcasted;

    // Print the resulting shape of z1
    println!("Shape of z1: {:?}", z1.shape());

    let df: DataFrame = read_csv();
    let (mut ndarray, _m, _n): (Array2<f32>, usize, usize) = parse_data(df);

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

}