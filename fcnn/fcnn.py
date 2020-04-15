# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:40:16 2020

@author: 欧阳磊
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from fcnn_utils import *


class fully_connected_nn(object):
    def __init__(self, hyperparams):
        """
        Initializes an fully connected neural network(FCNN)
        An FCNN has following attributes:
            self.hidden_activations
            self.output_activations
            self.loss_function
            self.layers_size
            self.learning_rate
            self.parameters
        """
        self.hidden_activations = hyperparams['hidden_activations']
        self.output_activations = hyperparams['output_activations']
        self.loss_function = hyperparams['loss_function']
        self.layers_size = hyperparams['layers_size']
        self.learning_rate = hyperparams['learning_rate']
        self.parameters = hyperparams['init_params']


    def init_params(self):
        """
        initialize parameters for the neural network
        """
        
        np.random.seed(1)
        parameters = self.parameters
        layer_dims = self.layers_size
        L = len(layer_dims)            # number of layers in the network
    
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    def forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        
        
        def linear_forward(A, W, b):
            """
            Implement the linear part of a layer's forward propagation.
            """
            Z = W.dot(A) + b
            
            assert(Z.shape == (W.shape[0], A.shape[1]))
            cache = (A, W, b)
            
            return Z, cache
        
        
        def activation_forward(Z, activation):
            if activation == 'relu':
                A = np.maximum(0,Z)
            elif activation == 'logistic':
                A = 1/(1+np.exp(-Z))
                
            assert(A.shape == Z.shape)
            cache = Z 
            
            return A, cache
        
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activation_forward(Z, activation)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
    
        return A, cache


    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """
    
        caches = []
        A = X
        parameters = self.parameters
        L = len(parameters) // 2    # number of layers in the neural network
        
        # Implement [LINEAR -> activation]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self.forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], self.hidden_activations)
            caches.append(cache)
        
        # Implement LINEAR -> output_activation. Add "cache" to the "caches" list.
        AL, cache = self.forward(A, parameters['W' + str(L)], parameters['b' + str(L)], self.output_activations)
        caches.append(cache)
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).
    
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]
    
        # Compute loss from aL and y.
        if self.loss_function == 'cross entropy':
            cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
        elif self.loss_function == 'mean squared':
            a = -1/(2*m) * np.sum((Y-AL)**2)
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ()) 
        return cost


    def backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        
        def linear_backward(dZ, cache):
            """
            Implement the linear portion of backward propagation for a single layer (layer l)
            """
            A_prev, W, b = cache
            m = A_prev.shape[1]
        
            dW = 1./m * np.dot(dZ,A_prev.T)
            db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
            dA_prev = np.dot(W.T,dZ)
            
            assert (dA_prev.shape == A_prev.shape)
            assert (dW.shape == W.shape)
            assert (db.shape == b.shape)
            
            return dA_prev, dW, db
        
        
        def activation_backward(dA, cache, activation):
            """
            Implement the backward propagation for a single hidden/output unit.
            """
            
            Z = cache # retrive Z from cache
            
            if activation == 'relu':
                dZ = np.array(dA, copy=True) # just converting dz to a correct object.
                dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0 as well. 
            elif activation == 'logistic':
                s = 1/(1+np.exp(-Z))
                dZ = dA * s * (1-s)
    
            assert (dZ.shape == Z.shape)
            
            return dZ
        
        
        linear_cache, activation_cache = cache
        
        dZ = activation_backward(dA, activation_cache, activation)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
        return dA_prev, dW, db


    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        if self.loss_function == 'cross entropy':
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # it should be modified
        elif self.loss_function == 'mean squared':
            dAL = Y - AL
        
        # Lth layer (Activation -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.backward(
            dAL, current_cache, self.output_activations)
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backward( 
                grads["dA" + str(l + 2)], current_cache, self.hidden_activations)
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
    
        return grads
    

    def gradient_check(self, X, Y, epsilon = 1e-7):
        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
        
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """
        
        def dictionary_to_vector(parameters, dict_type):
            """
            Roll all our parameters dictionary into a single vector satisfying our specific required shape.
            Arguments:
            parameters -- dict to be transformed
            dict_type: -- 'parameters' or 'grads'
            """
            keys = []
            count = 0
            L = len(parameters) // 2 if dict_type == 'parameters' else len(parameters) // 3
            for l in range(L):
                if dict_type == 'parameters':
                    key_W = 'W' + str(l+1)
                    key_b = 'b' + str(l+1)
                elif dict_type == 'grads':
                    key_W = 'dW' + str(l+1)
                    key_b = 'db' + str(l+1)
                
                # flatten parameter
                new_vector = np.reshape(parameters[key_W], (-1,1))
                keys = keys + [key_W]*new_vector.shape[0]
                
                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector), axis=0)
                count = count + 1
                
                new_vector = np.reshape(parameters[key_b], (-1,1))
                keys = keys + [key_b]*new_vector.shape[0]
                
                theta = np.concatenate((theta, new_vector), axis=0)

            return theta, keys
        
        
        def vector_to_dictionary(theta, layers_size):
            """
            Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
            """
            parameters = {}
            L = len(layers_size)
            ind_start = 0
            ind_end = 0
            for l in range(1, L):
                ind_start, ind_end = ind_end, ind_end + layers_size[l] * layers_size[l-1]
                parameters["W" + str(l)] = theta[ind_start:ind_end].reshape((layers_size[l], layers_size[l-1]))
                ind_start, ind_end = ind_end, ind_end + layers_size[l]
                parameters["b" + str(l)] = theta[ind_start:ind_end]
        
            return parameters
    
    
        # BP gradients
        parameters_values, _ = dictionary_to_vector(self.parameters, 'parameters')
        AL, caches = self.L_model_forward(X)
        gradients = self.L_model_backward(AL, Y, caches)
        grad, _ = dictionary_to_vector(gradients, 'grads')
        
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        
        # Compute gradapprox
        for i in range(num_parameters):
            
            thetaplus = np.copy(parameters_values)                                     
            thetaplus[i][0] += epsilon
            thetaplus = vector_to_dictionary(thetaplus, self.layers_size)
            self.parameters = thetaplus
            ALplus, _ = self.L_model_forward(X)
            J_plus[i] = self.compute_cost(ALplus, Y)
    
            thetaminus = np.copy(parameters_values)
            thetaminus[i][0] -= epsilon
            thetaminus = vector_to_dictionary(thetaminus, self.layers_size)
            self.parameters = thetaminus
            ALminus, _ = self.L_model_forward(X)
            J_minus[i] = self.compute_cost(ALminus, Y)
            
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        
        # Compare gradapprox to backward propagation gradients by computing difference.
        numerator = np.linalg.norm(gradapprox - grad)
        denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
        difference = numerator / denominator
    
        if difference > 1e-7:
            print ("There is a mistake in the backward propagation! difference = " + str(difference))
        else:
            print ("Your backward propagation works perfectly fine! difference = " + str(difference))
        
        return difference



    def update_parameters(self, grads):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        """
        
        L = len(self.parameters) // 2 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] -= self.learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= self.learning_rate * grads["db" + str(l+1)]
            
    
    def train(self, X, Y, num_iterations = 2500, print_cost = True):
        
        np.random.seed(1)
        costs = []  # keep track of cost
        self.init_params()
        
        for i in range(num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y) # Compute cost.
            grads = self.L_model_backward(AL, Y, caches)   # Backward propagation.
     
            self.update_parameters(grads)

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()


    def predict(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X)
    
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p



def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    (train_x, test_x) = standardize_data(train_x_orig, train_y, test_x_orig, test_y, classes)
    
    layers_size = [train_x.shape[0], 20, 7, 5, 1]
    
    hyperparams = {
            'hidden_activations':'relu',
            'output_activations':'logistic',
            'loss_function':'cross entropy',
            'layers_size': layers_size,
            'learning_rate': 0.0075,
            'init_params' : {}}
    
    net = fully_connected_nn(copy.deepcopy(hyperparams))
    net.init_params()
    
    # gradient check
    hyperparams['hidden_activations'] = 'logistic' # use logistic instead of relu to do gradient check!
    hyperparams['layers_size'] = [500, 2, 15, 10, 5, 5, 15, 5, 5, 5, 5, 15, 1]
    test_net = fully_connected_nn(hyperparams)
    test_net.init_params()
    test_net.gradient_check(train_x[:500, :], train_y[:, :])
    
    # net.train(train_x, train_y)
    
    # pred_train = net.predict(train_x, train_y)
    # pred_test = net.predict(test_x, test_y)
    # print_mislabeled_images(classes, test_x, test_y, pred_test)
    
    
if __name__ == '__main__':
    main()