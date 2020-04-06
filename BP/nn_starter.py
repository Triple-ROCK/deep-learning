import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x):
    """
    Compute softmax function for input.
    Use tricks from previous assignment to avoid overflow
    """
    # YOUR CODE HERE
    x -= np.max(x, axis=0, keepdims=True)  # normalize to avoid overflow
    tmp = np.exp(x)
    s = tmp / np.sum(tmp, axis=0, keepdims=True)
    # END YOUR CODE
    return s


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    # YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    # END YOUR CODE
    return s


def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # YOUR CODE HERE
    m = data.shape[1]
    z1 = W1 @ data + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = softmax(z2)
    cost = -np.sum(labels * np.log(a2)) / m
    ret = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'cost': cost}
    for key in params:
        ret[key] = params[key]
    # END YOUR CODE
    return ret


def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    a1 = params['a1']
    a2 = params['a2']

    # YOUR CODE HERE
    m = data.shape[1]
    delta2 = -(labels - a2)
    delta1 = W2.T @ delta2 * a1 * (1 - a1)
    gradW2 = delta2 @ a1.T / m
    gradW1 = delta1 @ data.T / m
    gradb2 = np.sum(delta2, axis=1, keepdims=True) / m
    gradb1 = np.sum(delta1, axis=1, keepdims=True) / m
    # END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad


def init_weight(layer_sizes):
    params = {}
    n, num_hidden, output_dim = layer_sizes
    s1 = np.sqrt(6) / np.sqrt(n + num_hidden)
    s2 = np.sqrt(6) / np.sqrt(num_hidden + output_dim)
    # params['W1'] = np.random.rand(num_hidden, n)*2*s1 -s1
    params['W1'] = np.random.randn(num_hidden, n)
    # params['W2'] = np.random.rand(output_dim, num_hidden)*2*s2 - s2
    params['W2'] = np.random.randn(output_dim, num_hidden)
    params['b1'] = np.zeros((num_hidden, 1))
    params['b2'] = np.zeros((output_dim, 1))
    return params


def nn_train(trainData, trainLabels, devData, devLabels):
    (n, m) = trainData.shape  # n: demension of each example,  m: number of examples
    num_hidden = 300
    learning_rate = 5
    num_epoch = 30
    B = 1000

    # YOUR CODE HERE
    layer_sizes = (n, num_hidden, 10)
    params = init_weight(layer_sizes)
    cost = np.zeros(num_epoch)
    for epoch in range(num_epoch):
        for i in range(m // B):
            X = trainData[:, i * B:(i + 1) * B]
            y = trainLabels[:, i * B:(i + 1) * B]
            fprop_cache = forward_prop(X, y, params)
            grad = backward_prop(X, y, fprop_cache)
            params['W1'] -= learning_rate * grad['W1']
            params['W2'] -= learning_rate * grad['W2']
            params['b1'] -= learning_rate * grad['b1']
            params['b2'] -= learning_rate * grad['b2']
        cost[epoch] = forward_prop(trainData, trainLabels, params)['cost']
        print('epoch: {}   cost: {}\n'.format(epoch, cost[epoch]))
    # END YOUR CODE

    return params, cost


def nn_test(data, labels, params):
    output = forward_prop(data, labels, params)['a2']
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (
        np.argmax(
            output,
            axis=0) == np.argmax(
            labels,
            axis=0)).sum() * 1. / labels.shape[1]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    # transpose to get n by m shape, where m = #examples
    devData = trainData[0:10000, :].T
    devLabels = trainLabels[0:10000, :].T
    trainData = trainData[10000:, :].T
    trainLabels = trainLabels[10000:, :].T

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels).T
    testData = testData.T
    testData = (testData - mean) / std

    params, cost = nn_train(trainData, trainLabels, devData, devLabels)
    plt.plot(cost)

    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print('Test accuracy: %f' % accuracy)


if __name__ == '__main__':
    main()
