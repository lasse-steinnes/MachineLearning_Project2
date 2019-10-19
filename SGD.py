# -*- coding: utf-8 -*-
import numpy as np
from autograd import grad


class SGD:
    '''
    class which performs Stocastic Gradient Descent for a Neural Network
    and for Logistic regression. 
    '''
    
    def __init__(self, cost_function, epochs =10, mini_batch_size = 10, 
                learning_rate = 0.5, adaptive_learning_rate = 'const', tolerance = 1, max_iter = 1000):
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
        #cost_function(self, Weights, data, target) 
        self.deriv_cost_function = grad(cost_function, 1)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.weights = None
        self.mini_batch_size  = mini_batch_size
        self.time  = 0
        
        
        try:
            self.learning_rate_adaption = {'const': False, 'decay': SGD.__decay, 'momentum': SGD.__momentum}[adaptive_learning_rate]
        except:
            self.learning_rate_adaption = adaptive_learning_rate
        
    def run_SGD(self, X, t):
        samples = X.shape[0]
        num_mini_batches = samples // self.mini_batch_size
        self.iteration = 0
        self.__old_weights = 100
        current_epoch = 0
       
        while SGD.__check(self, self.iteration, self.__old_weights, current_epoch):
            SGD.run_epoch(self,X, t,num_mini_batches)
            current_epoch += 1
        return self.weights

    def run_minibatch(self, minibatch):
        """
        runs a minibatch of from (X,t)
        """
        try:
            self.gamma = self.learning_rate_adaption(self, self.learning_rate, self.time)
        except:
            self.gamma = self.learning_rate
    
        self.delta = self.deriv_cost_function(self, self.weights, minibatch[0], minibatch[1])
        self.__old_weights = self.weights
        self.weights = self.__old_weights - self.gamma * self.delta

        

    def run_epoch(self,X, t, num_mini_batches):
        """
        runs a whole epoch
        shuffles data and splits it
        """
        for mini_batch in SGD.creat_mini_batch(self, X, t, num_mini_batches):
            SGD.run_minibatch(self, mini_batch)
            self.iteration += 1
        self.time += 1
        

    def creat_mini_batch(self,X, t, num_mini_batches):
        """
        returns array with [... (X_n, t_n) ...] for 0 < n < num_mini_batches
        """
        cur_seed = np.random.randint(self.epochs*self.max_iter + self.iteration)
        np.random.seed(cur_seed)
        np.random.shuffle(X)
        np.random.seed(cur_seed)
        np.random.shuffle(t)

        mini_batches_X = np.array_split(X, num_mini_batches)
        mini_batches_T = np.array_split(t, num_mini_batches)

        return zip(mini_batches_X, mini_batches_T)
    
    #halting condition
    def __check(self, iteration, old_weights, current_epoch):
        if iteration >= self.max_iter: 
            print("Max. iter. reached")
            return False
        if np.linalg.norm(old_weights - self.weights) < self.tolerance:
            print("tolerance reached")
            return False
        if current_epoch >= self.epochs: return False
        return True

    #functions for adaptive learning rate
    def __decay(self, gamma0, t):
        return gamma0 / ( gamma0*t +1)
    def __momentum(self, m0, t):
        pass
