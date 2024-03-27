// use polars::prelude::*;
// use polars_io::csv::CsvReader;
use ndarray::*;
// use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;


fn argmax(matrix: Array2<f32>) -> Array1<f32> {
    let mut output = Array1::zeros(matrix.shape()[1]);
    
    
    for (i, row) in matrix.axis_iter(Axis(1)).enumerate() {
        let (max_idx, _max_val) =
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

fn re_lu(z: &Array2<f32>) -> Array2<f32> {
    return z.mapv(|x| x.max(0.0));
}

// TO DO: Fix the softmax function

fn softmax(z: &Array2<f32>) -> Array2<f32> {
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

fn re_lu_deriv(z: Array2<f32>) -> Array2<f32> {
    return z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
}

fn one_hot(y: Array1<f32>) -> Array2<f32> {
    // Find the maximum value in Y
    let max_value = y.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    // Create a one-hot encoded matrix with the appropriate dimensions
    let mut one_hot_y: Array2<f32> = Array2::<f32>::zeros((y.len(), max_value as usize + 1));

    // Populate the one-hot encoded matrix
    for i in 0..y.len() {
        one_hot_y[[i, y[i] as usize]] = 1.0;
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
    let x = argmax(a2);
    println!("{:?}", x);
    x
    // return argmax(a2);
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

pub fn gradient_descent(mut x: Array2<f32>, y: Array1<f32>, alpha: f32, iterations: i32) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>){
    let (mut w1, mut b1, mut w2, mut b2) = init_params();
    for i in 0..iterations {
        println!("Iteration: {}", i);
        let (z1, a1, _z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &mut x);
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