import numpy as np

ACTIVATION = {"sigmoid": lambda z: sigmoid(z),
              "relu": lambda z: relu(z)}


def initialize_parameters(layer_dims):
    parameters = {}
    for index in range(1,len(layer_dims)):
        dim1 = layer_dims[index]
        dim2 = layer_dims[index - 1]
        parameters["W{}".format(index)] = np.random.randn(dim1, dim2)
        parameters["b{}".format(index)] = np.zeros((dim1, 1))

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    return Z, {"A": A,"W": W, "b": b, "Z": Z}


def sigmoid(z):
    return 1 / (1 + np.exp(-z)), z


def relu(z):
    return np.maximum(z, 0), z


def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    A_current, _ = ACTIVATION[activation](Z)
    return A_current, linear_cache


def L_model_forward(X, parameters):
    caches = []
    layers_amount = len(parameters.keys()) / 2

    A = X
    for layer in range(1, layers_amount):
        W = parameters["W{}".format(layer)]
        b = parameters["b{}".format(layer)]
        A, linear_cache = linear_activation_forward(A, W, b, "relu")
        caches.append(linear_cache)

    W = parameters["W{}".format(layers_amount)]
    b = parameters["b{}".format(layers_amount)]
    AL, linear_cache = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(linear_cache)

    return AL, caches


def compute_cost(AL, Y):
    Y_tran = Y.transpose()
    cost_arr = (Y_tran.dot(np.log(AL)) + (1 - Y_tran).dot(1 - AL)) / len(AL)
    return -1 * cost_arr[0][0]
