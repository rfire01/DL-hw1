import numpy as np

ACTIVATION = {"sigmoid": lambda dA, ac: sigmoid_backward(dA, ac),
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
    return linear_backward(ACTIVATION[activation](dA, activation_cache), cache)


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
