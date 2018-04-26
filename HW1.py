import numpy as np
import idx2numpy as idx2np

ACTIVATION_FORWARD = {"sigmoid": lambda z: sigmoid(z),
              "relu": lambda z: relu(z)}

ACTIVATION_BACKWARD = {"sigmoid": lambda dA, ac: sigmoid_backward(dA, ac),
              "relu": lambda dA, ac: relu_backward(dA, ac)}

EPSILON = 0.000001

def initialize_parameters(layer_dims):
    parameters = {}
    for index in range(1,len(layer_dims)):
        dim1 = layer_dims[index]
        dim2 = layer_dims[index - 1]
        parameters["W{}".format(index)] = np.random.randn(dim1, dim2) * 0.02
        parameters["b{}".format(index)] = np.zeros((dim1, 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    return Z, {"A": A,"W": W, "b": b}


def sigmoid(z):
    # print('z = ', z)
    return 1.0 / (1 + np.exp(1.0 * z)), z


def relu(z):
    return np.maximum(z, 0), z


def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    A_current, activation_cache = ACTIVATION_FORWARD[activation](Z)
    # print('A = ', A_current)
    return A_current, linear_cache, activation_cache


def normalize_sigmoid_res(AL):
    activations = AL[0]
    for i in range(0, len(activations)):
        if activations[i] == 0:
            activations[i] == 0 + EPSILON
        if activations[i] == 1:
            activations[i] = 1 - EPSILON
    return activations


def L_model_forward(X, parameters):
    caches = []
    layers_amount = int(len(parameters.keys()) / 2)

    A = X
    for layer in range(1, layers_amount):
        W = parameters["W{}".format(layer)]
        b = parameters["b{}".format(layer)]
        A, linear_cache, activation_cache = linear_activation_forward(A, W, b, "relu")
        caches.append((linear_cache, activation_cache))

    W = parameters["W{}".format(layers_amount)]
    b = parameters["b{}".format(layers_amount)]
    AL, linear_cache, activation_cache = linear_activation_forward(A, W, b, "sigmoid")
    # AL = normalize_sigmoid_res(AL)
    caches.append((linear_cache, activation_cache))
    # print('AL', AL)

    return AL, caches


def compute_cost(AL, Y):
    AL_trans = AL.transpose()
    cost_arr = (Y.dot(np.log(AL_trans)) +
                (1 - Y).dot(np.log(1 - AL_trans))) / Y.shape[1]
    return -1 * cost_arr[0][0]


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']

    m = len(dZ.transpose())
    dW = dZ.dot(A_prev.transpose()) / m
    db = np.sum(dZ, axis=1) / m
    db = db.reshape(len(db), 1)
    dA = W.transpose().dot(dZ)

    return dA, dW, db


def relu_backward (dA, activation_cache):
    return np.multiply(dA, (activation_cache > 0))


def sigmoid_der(z, derivative=True):
    # Calculate sigmoid derivative
    return np.multiply(z, (1 - z)) if derivative else 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, activation_cache):
    return np.multiply(dA, sigmoid_der(activation_cache))


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    return linear_backward(ACTIVATION_BACKWARD[activation](dA, activation_cache),
                           linear_cache)


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
    layers_amount = int(len(parameters.keys()) / 2)

    for layer in range(1, layers_amount + 1):
        parameters["W{}".format(layer)] -= learning_rate * grads[
            "dW{}".format(layer)]
        parameters["b{}".format(layer)] -= learning_rate * grads[
            "db{}".format(layer)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    costs = []
    parameters = initialize_parameters(layers_dims)
    for iteration in range(1, num_iterations + 1):
        print('iteration ', iteration)
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if (iteration % 100) == 0:
            costs.append(cost[0][0])

        grads = L_model_backward(AL, Y, caches)
        parameters = Update_parameters(parameters, grads, learning_rate)

    return parameters, costs


def get_digit_indices(Y, digits):
    if (digits == '3,8'):
        first_digit = Y == 3
        second_digit = Y == 8
    else:
        first_digit = Y == 7
        second_digit = Y == 9

    indices_first = np.array(range(len(first_digit)))
    indices_first = indices_first[first_digit]
    indices_second = np.array(range(len(second_digit)))
    indices_second = indices_second[second_digit]
    indices = np.append(indices_first, indices_second)
    indices = np.sort(indices)
    return indices


def get_filtered_X(X, Y, digits):
    indices = get_digit_indices(Y, digits)
    filtered = X[indices]
    X_refactored = []
    for index in range(0, filtered.shape[0]):
        exm_matrix = filtered[index]
        exm_vect = np.ndarray.flatten(exm_matrix)
        X_refactored.append(exm_vect)

    X_refactored = np.asanyarray(X_refactored)
    X_refactored = X_refactored.transpose()
    return X_refactored


def transform_digits(indices, digits):
    labels = indices.getA()
    res = []
    for i in range(0, labels.shape[1]):
        if digits == '3,8':
            if labels[0][i] == 3:
                res.append(1)
            else:
                res.append(0)
        else:
            if labels[0][i] == 7:
                res.append(1)
            else:
                res.append(0)
    return res


def get_filtered_Y(Y, digits):
    indices = get_digit_indices(Y, digits)
    temp = np.matrix(Y[indices])
    res = transform_digits(temp, digits)
    return np.matrix(res)


def get_predictions(probabilities):
    res = []
    for i in range(0, probabilities.size):
        if probabilities[0][i] > 0.5:
            res.append(1)
        else:
            res.append(0)
    return res


def get_accuracy(predictions, Y):
    num_of_correct = 0
    Y = Y.getA()[0]
    for i in range(0, len(predictions)):
        if(predictions[i] == Y[i]):
            num_of_correct += 1
    return float(num_of_correct) / len(predictions)


def predict(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters)
    predictions = get_predictions(AL)
    accuracy = get_accuracy(predictions, Y)
    return accuracy

# -------------------- training data --------------------

X = idx2np.convert_from_file('train-images.idx3-ubyte')
Y = idx2np.convert_from_file('train-labels.idx1-ubyte')

X_3_8 = get_filtered_X(X, Y, '3,8')
X_7_9 = get_filtered_X(X, Y, '7,9')

Y_3_8 = get_filtered_Y(Y, '3,8')
Y_7_9 = get_filtered_Y(Y, '7,9')

# -------------------- test data --------------------

X_test = idx2np.convert_from_file('t10k-images.idx3-ubyte')
Y_test = idx2np.convert_from_file('t10k-labels.idx1-ubyte')

X_3_8_test = get_filtered_X(X_test, Y_test, '3,8') / 255
X_7_9_test = get_filtered_X(X_test, Y_test, '7,9') / 255

Y_3_8_test = get_filtered_Y(Y_test, '3,8')
Y_7_9_test = get_filtered_Y(Y_test, '7,9')


parameters_3_8,costs_3_8 = L_layer_model(X_3_8, Y_3_8, [784, 20, 7, 5, 1], 0.02, 100)
# parameters_7_9,costs_7_9 = L_layer_model(X_7_9, Y_7_9, [784, 20, 7, 5, 1], 0.05, 1000)

print('parameters 3,8 : ', parameters_3_8)
print('costs 3,8 :', costs_3_8)

# print('parameters 7,9 : ', parameters_7_9)
# print('costs 7,9 :', costs_7_9)


accuracy_3_8 = predict(X_3_8_test, Y_3_8_test, parameters_3_8)
# accuracy_7_9 = predict(X_7_9_test, Y_7_9_test, parameters_7_9)

print('accuracy for 3,8 = ', accuracy_3_8)
# print('accuracy for 7,9 = ', accuracy_7_9)









