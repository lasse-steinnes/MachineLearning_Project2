# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:18:17 2019

@author: James
"""
import numpy as np

def cross_entropy_cost_function (a, y):
    
    return np.sum(-y * np.log(a) + (1 - y) * np.log(1 - a))


def SGD(cost_function, parameter, learning_rate = 0.5 ,  tolerance = 1):
    """
    Stocastic gradient descent (SGD) should be run
    in each batch. It should be used by picking a few points
    at random and sending them through the network. At 
    which point in the last layer the gradients are found 
    of these points and a new parameter (weight and bias)
    is found to minimise the cost function.
    
    This function takes the cost function, learning rate and 
    the parameter.
    This function outputs the new updated parameter and a 
    boolean refering to if the tolerance has been reached.
    'True' we are within the tolerance, 'False' we have not reached the 
    max tolerance.
    """
    training_data = np.arange(100.0)
    epochs = 10
    mini_batch_size = 10
    num_mini_batches = training_data / mini_batch_size
    
     
    
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        mini_batches = np.array(np.array_split(training_data, num_mini_batches))
        for mini_batch in mini_batches:
            update_mini_batch_WandB(mini_batch, cost_function, learning_rate, parameter)
        
        # evaluate the accuracy of mini_batch
        # and store the best result
        #return a boolean if the tolerance is reached for early stopping
        
    return

def update_mini_batch_WandB(mini_batch, cost_function,learning_rate, parameter):
    
    #delta_grad_w = [np.zeros(w.shape) for w in self.weights]
    #delta_grad_b = [np.zeros(b.shape) for b in self.biases]
    
    for data in mini_batch:
        #calls backpropagation to find the new gradient or change
        # in gradient
        grad_w , grad_b = backpropagation(data)
        
        self.weights = self.weights - learning_rate * grad_w
        self.biases = self.biases - learning_rate * grad_b
     


