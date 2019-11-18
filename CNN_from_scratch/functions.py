import numpy as np


def sigmoid(x):
    return np.true_divide(1, 1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

def d_sigmoid(x):
    sigma = np.true_divide(1, 1 + np.exp(-x))
    return sigma * (1 - sigma)

def d_relu(x):
    return 1. * (x > 0)

def softmax(x):
    n = np.exp(x - np.max(x))
    return np.true_divide(n, np.sum(n))
