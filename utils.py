import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivation_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def derivation_relu(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=1).reshape(x.shape[0], 1)


def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat + 1e-6)) / y.shape[0]
