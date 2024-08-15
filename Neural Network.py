import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# dataset generation
X = np.linspace(0, 10, 100)
y = np.exp(-0.1 * X)

# displaying data set
plt.xlabel("x")
plt.ylabel("f(x)=e^-0.1x")
plt.title("Overall Dataset")
plt.plot(X, y)

# cross validation method
kf = KFold(n_splits=5)  # default is 5 folds
train_indices = np.empty((5, 80))
test_indices = np.empty((5, 20))
plt.figure(figsize=(12, 12))

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    train_indices[i, :] = train_index.T
    test_indices[i, :] = test_index.T
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    plt.subplot(320 + 1 + i)
    plt.plot(X_train, y_train, 'r')
    plt.plot(X_test, y_test, 'b')
    plt.xlabel("x")
    plt.ylabel("f(x)=e^-0.1x")
    plt.title('Train and Test Data Sets')
    plt.legend(['train', 'test'])
plt.show()


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# forward pass function
def forward_pass(x, weight, bias):
    a1 = x * weight[0] + bias[0]
    z1 = sigmoid(a1)
    a2 = z1 * weight[1] + bias[1]
    z2 = sigmoid(a2)
    return z1, z2


# gradient function
def gradient(x, y, weight, z1, z2):
    temp = -1 * (y - z2) * (z2) * (1 - z2)
    grad_b1 = temp * weight[1] * (z1) * (1 - z1)
    grad_w1 = grad_b1 * x
    grad_b2 = temp
    grad_w2 = temp * z1
    return grad_w1, grad_b1, grad_w2, grad_b2


# initializing weight and biases to random values
weight = np.array([0.1, 0.1])
bias = np.array([0.1, 0.1])
weight.shape, bias.shape
epochs = 5
lr = 1  # learning rate
train_indices = train_indices.astype('i')
test_indices = test_indices.astype('i')

# X_train,y_train=X[train_indices[0,:]],y[train_indices[0,:]]
# print(X_train.T)
# X_train.shape,y_train.shape

for j in range(5):
    X_train, y_train = X[train_indices[j, :]], y[train_indices[j, :]]
    X_test, y_test = X[test_indices[j, :]], y[test_indices[j, :]]
    print("Cross Validation:", j + 1)
    for epoch in range(epochs):
        for i in range(80):  # loop for all samples
            z1, z2 = forward_pass(X_train[i], weight=weight, bias=bias)
            grad_w1, grad_b1, grad_w2, grad_b2 = gradient(X_train[i], y_train[i], weight=weight, z1=z1, z2=z2)
            weight[0] -= lr * grad_w1
            bias[0] -= lr * grad_b1
            weight[1] -= lr * grad_w2
            bias[1] -= lr * grad_b2

        # testing
        y_predicted = np.empty(y_test.shape)
        for i in range(20):
            z1, z2 = forward_pass(X_test[i], weight=weight, bias=bias)
            y_predicted[i] = z2

        # root means sqaured error on testing dataset
        rmse = mean_squared_error(y_true=y_test, y_pred=y_predicted)
        data = {'Epoch': epoch + 1, 'Weight 1': weight[0], 'Weight 2': weight[1], 'Bias 1': bias[0], 'Bias 2':bias[1],'RMSE':rmse}
        df = pd.DataFrame(data, index=[0])
        print(df)
        print()

        # displaying data set
        plt.xlabel("x")
        plt.ylabel("f(x)=e^-0.1x")
        plt.title("Overall Dataset")
        plt.plot(X, y, label="Training+Testing")
        y_predicted = np.empty(len(y))

        for i in range(len(X)):
            z1, z2 = forward_pass(X[i], weight=weight, bias=bias)
            y_predicted[i] = z2
        plt.plot(X, y_predicted, label="Predicted")
        plt.legend()
        plt.show()