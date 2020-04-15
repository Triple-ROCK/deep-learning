# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:39:07 2020

@author: 欧阳磊
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


class optimizer(object):
    def __init__(self, optimizer_params):
        """
        Arguments:
        optimizer_params -- parameters of your optimizer
        """
        self.learning_rate = optimizer_params['learning_rate']
    
    
    def initialize(self):
        """
        Initialize the optimizer for Momentum or Adam
        """
        raise NotImplementedError
        
    
    def update_parameters(self, parameters, grads):
        """
        Arguments:
        parameters -- to be optimized by this optimizer: A dict
        grads -- used to optimize parameters: A dict
        """
        L = len(parameters) // 2  # number of layers in the neural networks

        # Update rule for each parameter
        for l in range(L):
            parameters["W" + str(l+1)] -= self.learning_rate * grads['dW' + str(l+1)]
            parameters["b" + str(l+1)] -= self.learning_rate * grads['db' + str(l+1)]
            
        return parameters
    

class Momentum(optimizer):
    def __init__(self, optimizer_params):
        optimizer.__init__(self, optimizer_params)
        self.beta = optimizer_params['beta']
        self.v = {}
        
    
    def initialize(self, parameters):
        """
        Initialize the optimizer for Momentum
        """
        L = len(parameters) // 2 # number of layers in the neural networks
        
        # Initialize velocity
        for l in range(L):
            self.v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
            self.v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

        
    def update_parameters(self, parameters, grads):
        """
        Update parameters using Momentum
        
        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients for each parameters
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(parameters) // 2 # number of layers in the neural networks
    
        # Momentum update for each parameter
        for l in range(L):
            
            # compute velocities
            self.v["dW" + str(l+1)] = self.beta * self.v['dW' + str(l+1)] + (1-self.beta) * grads['dW' + str(l+1)]
            self.v["db" + str(l+1)] = self.beta * self.v['db' + str(l+1)] + (1-self.beta) * grads['db' + str(l+1)]
            # update parameters
            parameters["W" + str(l+1)] -= self.learning_rate * self.v["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= self.learning_rate * self.v["db" + str(l+1)]
            
        return parameters
    
    
    
class Adam(optimizer):
    def __init__(self, optimizer_params):
        optimizer.__init__(self, optimizer_params)
        self.beta1 = optimizer_params['beta1']
        self.beta2 = optimizer_params['beta2']
        self.epsilon = 1e-8
        self.v = {}
        self.s = {}
        
    
    def initialize(self, parameters):
        """
        Initialize the optimizer for Momentum
        """
        L = len(parameters) // 2 # number of layers in the neural networks
        
        # Initialize velocity
        for l in range(L):
            self.v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
            self.v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
            self.s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            self.s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

        
    def update_parameters(self, parameters, grads, t):
        """
        Update parameters using Momentum
        
        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients for each parameters
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        
        # Alias for optimizer parameters 
        v, s, beta1, beta2, learning_rate, epsilon = \
            self.v, self.s, self.beta1, self.beta2, self.learning_rate, self.epsilon
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]
    
            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
    
            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * (grads['dW' + str(l+1)]**2)
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * (grads['db' + str(l+1)]**2)
    
            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
    
            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] -= learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    
        return parameters, v, s
    
  

def load_data():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


    
def standardize_data(train_x_orig, train_y, test_x_orig, test_y, classes):
    
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    return (train_x, test_x) 
    
  
    
def one_hot_labels(labels, num_classes = 10):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels.T


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        
        
