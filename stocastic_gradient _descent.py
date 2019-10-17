# -*- coding: utf-8 -*-

import numpy as np
import autograd as auto


class SGD:
    '''
    class which performs Stocastic Gradient Descent for a Neural Network
    and for Logistic regression. 
    '''
    
    def __init__(self, cost_function, training_data = np.arange(100.0), epochs =10, mini_batch_size = 10, learning_rate = 0.5, tolerance = 1):
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
        'True' we are within the tolerance, 'False' we have not 
        reached the max tolerance.
        """
        self.cost_function = cost_function
        self.training_data = training_data
        self.epochs = epochs
        self.num_mini_batches = training_data / mini_batch_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        
    def run_batch(self):
        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            mini_batches = np.array(np.array_split(self.training_data, self.num_mini_batches))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
        
    # evaluate the accuracy of mini_batch
    # and store the best result
    #return a boolean if the tolerance is reached for early stopping   
    
    def update_mini_batch(self, mini_batch):
    
        #delta_grad_w = [np.zeros(w.shape) for w in self.weights]
        #delta_grad_b = [np.zeros(b.shape) for b in self.biases]
    
        for data in mini_batch:
            #calls backpropagation to find the new gradient or change
            # in gradient
            grad_w , grad_b = backpropagation(data)
            
            self.weights = self.weights - learning_rate * grad_w
            self.biases = self.biases - learning_rate * grad_b
      
class SGD_Log(SGD):
    '''
    This is a child class of the SGD class used to run Stocastic
    Gradient Descent for Logistic regression.
    '''
    def __init__(self, cost_function, parameter, training_data = np.arange(100.0), epochs =10, mini_batch_size = 10, learning_rate = 0.5, tolerance = 1):
        SGD.__init__(self, cost_function, training_data = np.arange(100.0), epochs =10, mini_batch_size = 10, learning_rate = 0.5, tolerance = 1)
        
        self.parameter = parameter
        self.der_cost_function = auto(self.cost_function)
    
    def run_batch(self):
        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            mini_batches = np.array(np.array_split(self.training_data, self.num_mini_batches))
            
            for mini_batch in mini_batches:
                gradient = self.der_cost_function(self.parameter)
                self.parameter = self.parameter - self.learning_rate * gradient
                
def cross_entropy_cost_function (a, y):
    return np.sum(-y * np.log(a) + (1 - y) * np.log(1 - a))


   


