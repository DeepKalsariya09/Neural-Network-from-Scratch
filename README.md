# Neural-Network-from-Scratch
This repository demonstrate the basics of neural networks, including forward pass, backpropagation, and cross-validation, all coded from scratch using Python and popular libraries like NumPy, Pandas, and Matplotlib.

## Overview
This project implements a simple neural network to approximate the exponential decay function 𝑓(𝑥)=𝑒
−
0.1
𝑥
f(x)=e 
−0.1x. The network is trained and validated using K-Fold cross-validation, providing insights into its performance across different data splits. Visualization tools are employed to illustrate both the dataset and the network's predictions.

## Features
- **Data Generation:** Synthetic dataset creation for 
𝑓
(𝑥)=
𝑒
−
0.1
𝑥
f(x)=e 
−0.1x
 .
- **Visualization:** Plotting of the dataset and model predictions.
- **K-Fold Cross-Validation:** Implementation of 5-fold cross-validation to assess model robustness.
- **Neural Network Architecture:** Simple feedforward network with sigmoid activation functions.
- **Training:** Gradient descent optimization with detailed tracking of weights, biases, and RMSE.

## Results
- **Weights and Biases:** Updated weights and biases after each epoch.
- **RMSE:** Root Mean Squared Error for each epoch to assess the model's performance.
- **Predicted vs Actual:** Visualization of the predicted function against the actual function.
