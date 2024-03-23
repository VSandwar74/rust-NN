# Neural Network Made from Scratch in Rust

## Only used polars for io, ndarray for linear algebra/math, and rng for random

inspired by Samson Zhang, transpiled to Rust

## Trained on MNIST dataset
Accessible at https://www.kaggle.com/competitions/digit-recognizer or Kaggle API

## Steps:
1. Read CSV using polars-io
2. Parse data with util library and Fisher-Yates shuffle dataframe and train test validation split
3. Create initial params by creating matrices and vectors and initializing using thread rng
4. Create forward propogation function using activation layer calculus, softmax function
5. Create backward propogation function through one hot encoding, and basic linear algebra
6. Create prediction accuracy function by replication numpy argmax in Rust
7. Perform gradient descent by selecting hyperparameters and iterating through data set using function.

## Final Accuracy
74%.