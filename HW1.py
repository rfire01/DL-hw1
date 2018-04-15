import numpy as np

ACTIVATION_FORWARD = {"sigmoid": lambda z: sigmoid(z),
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
    A_current, _ = ACTIVATION_FORWARD[activation](Z)
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




ACTIVATION_BACKWARD = {"sigmoid": lambda dA, ac: sigmoid_backward(dA, ac),
              "relu": lambda dA, ac: relu_backward(dA, ac)}


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']

    m = len(dZ.transpose())
    dW = dZ.dot(A_prev.transpose()) / m
    db = dZ / m
    dA = W.transpose().dot(dZ)

    return dA, dW, db


def relu_backward (dA, activation_cache):
    return dA * (activation_cache > 0)


def sigmoid_der(z, derivative=True):
    # Calculate sigmoid derivative
    return z * (1 - z) if derivative else 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, activation_cache):
    return dA * sigmoid_der(activation_cache)


def linear_activation_backward(dA, cache, activation):
    activation_cache = cache['Z']
    return linear_backward(ACTIVATION_BACKWARD[activation](dA, activation_cache), cache)


def L_model_backward(AL, Y, caches):
    layers_amount = len(caches)
    grads = {}

    # This line is problematic when AL ~ 0 or 1 (taken from the assignment)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA{}".format(layers_amount)] = dAL
    dA, dW, db = linear_activation_backward(dAL, caches[-1], "sigmoid")
    grads["dW{}".format(layers_amount)] = dW
    grads["db{}".format(layers_amount)] = db

    for layer in range(layers_amount - 1, 0, -1):
        grads["dA{}".format(layer)] = dA
        dA, dW, db = linear_activation_backward(dA, caches[layer - 1], "relu")
        grads["dW{}".format(layer)] = dW
        grads["db{}".format(layer)] = db

    return grads


def Update_parameters(parameters, grads, learning_rate):
    layers_amount = len(parameters.keys()) / 2

    for layer in range(1, layers_amount + 1):
        parameters["W{}".format(layer)] -= learning_rate * grads[
            "dW{}".format(layer)]
        parameters["b{}".format(layer)] -= learning_rate * grads[
            "db{}".format(layer)]

    return parameters

