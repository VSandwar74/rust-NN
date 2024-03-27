use ndarray::*;
use rand::thread_rng;
use rand::Rng;


fn argmax(matrix: &Array2<f32>) -> Array1<f32> {
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

fn forward_prop(
    w1: &Array2<f32>, 
    b1: &Array1<f32>, 
    w2: &Array2<f32>, 
    b2: &Array1<f32>, 
    x: &mut Array2<f32>,
    z1: &mut Array2<f32>,
    a1: &mut Array2<f32>,
    z2: &mut Array2<f32>,
    a2: &mut Array2<f32>,
) {
    *z1 = w1.dot(x) + b1.broadcast((x.shape()[1], 10)).unwrap().t();
    *a1 = re_lu(&z1);
    *z2 = w2.dot(a1) + b2.broadcast((a1.shape()[1], 10)).unwrap().t();
    *a2 = softmax(z2);
    // println!("x: {:?}", x);
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
    dw1: &mut Array2<f32>,
    db1: &mut Array1<f32>,
    dw2: &mut Array2<f32>,
    db2: &mut Array1<f32>
) {
    let one_hot_y = one_hot(y.clone());
    let dz2 = a2 - &one_hot_y;
    *dw2 = 1.0 / m as f32 * dz2.dot(&a1.t());
    *db2 = 1.0 / m as f32 * dz2.sum_axis(Axis(1));
    let dz1 = w2.t().dot(&dz2) * re_lu_deriv(z1.clone());
    *dw1 = 1.0 / m as f32 * dz1.dot(&x.t());
    *db1 = 1.0 / m as f32 * dz1.sum_axis(Axis(1));
}
fn update_params(
    w1: &mut Array2<f32>,
    b1: &mut Array1<f32>,
    w2: &mut Array2<f32>,
    b2: &mut Array1<f32>,
    dw1: &Array2<f32>,
    db1: &Array1<f32>,
    dw2: &Array2<f32>,
    db2: &Array1<f32>,
    alpha: f32,
) {
    *w1 -= &(alpha * dw1);
    *w2 -= &(alpha * dw2);
    *b1 -= &(alpha * db1);
    *b2 -= &(alpha * db2);
}

fn get_predictions(a2: &Array2<f32>) -> Array1<f32> {
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

pub fn gradient_descent(mut x: Array2<f32>, y: Array1<f32>, alpha: f32, iterations: i32) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let (
        mut z1,
        mut a1,
        mut z2,
        mut a2
    ) = (
        Array2::<f32>::zeros((10, x.shape()[1])), 
        Array2::<f32>::zeros((10, x.shape()[1])), 
        Array2::<f32>::zeros((10, x.shape()[1])), 
        Array2::<f32>::zeros((10, x.shape()[1]))
    );
    let (
        mut dw1,
        mut db1,
        mut dw2,
        mut db2
    ) = (
        Array2::<f32>::zeros((10, x.shape()[1])), 
        Array1::<f32>::zeros(10), 
        Array2::<f32>::zeros((10, 10)), 
        Array1::<f32>::zeros(10)
    );

    let (mut w1, mut b1, mut w2, mut b2) = init_params();

    for i in 0..iterations {
        println!("Iteration: {}", i);
        forward_prop(&w1, &b1, &w2, &b2, &mut x, &mut z1, &mut a1, &mut z2, &mut a2);
        backward_prop(&z1, &a1, &a2, &w2, &x, &y, x.shape()[1], &mut dw1, &mut db1, &mut dw2, &mut db2);
        update_params(&mut w1, &mut b1, &mut w2, &mut b2, &dw1, &db1, &dw2, &db2, alpha);
        if i % 10 == 0 {
            println!("Iteration: {}", i);
            let predictions = get_predictions(&a2);
            println!("{}", get_accuracy(predictions, &y));
        }
    }

    return (w1, b1, w2, b2);
}