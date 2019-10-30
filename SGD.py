# -*- coding: utf-8 -*-
import numpy as np
from autograd import grad


class SGD:
    '''
    class which performs Stocastic Gradient Descent for a Neural Network
    and for Logistic regression. 
    '''
    
    def __init__(self, cost_function, epochs =10, mini_batch_size = 10, 
                learning_rate = 0.5, adaptive_learning_rate = 'const',
                momentum = True, m0 = 1e-2,
                tolerance = 1):
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
        self.mini_batch_size  = mini_batch_size
        self.learning_rate = learning_rate

        self.tolerance = tolerance
        #tracks number of epochs
        self. time = 0
        self.momentum = momentum
        if momentum:
            self.v = 0
            self.m0 = m0

        self.weights = None

        if adaptive_learning_rate =='cyclic':
            self.maxm0= m0
            self.maxgamma = learning_rate
        self.curr_iter = 0
        try:
            self.learning_rate_adaption = {'const': False, 'decay': SGD.__decay, 'momentum': SGD.__momentum, 'cyclic':SGD.__cyclic}[adaptive_learning_rate]
        except:
            self.learning_rate_adaption = adaptive_learning_rate
        
    def run_SGD(self, X, t):
        samples = X.shape[0]
        num_mini_batches = samples // self.mini_batch_size
        self.__old_weights = 100
       
        while SGD.__check(self, self.__old_weights):
            SGD.run_epoch(self,X, t,num_mini_batches)
            self.time += 1
        return self.weights

    def run_minibatch(self, minibatch, update_weight = True):
        """
        runs a minibatch of from (X,t)
        """
        #try:
        self.gamma = self.learning_rate_adaption(self, self.learning_rate, self.time)
        #except:
            #self.gamma = self.learning_rate

        if type(self.weights) == tuple:#when wheigths are set to be all inputs
            self.delta = self.deriv_cost_function( *self.weights)
        else:
            self.delta = self.deriv_cost_function(self, self.weights, minibatch[0], minibatch[1])
        
        if self.momentum:
            SGD.__momentum(self)
            self.delta = self.v
        self.delta *= self.gamma 
        self.curr_iter += 1
        if update_weight:
            self.__old_weights = self.weights
            self.weights = self.__old_weights - self.delta

        

    def run_epoch(self,X, t, num_mini_batches):
        """
        runs a whole epoch
        shuffles data and splits it
        """
        for mini_batch in SGD.creat_mini_batch(self, X, t, num_mini_batches):
            SGD.run_minibatch(self, mini_batch)
        self.time += 1
        

    def creat_mini_batch(self,X, t, num_mini_batches):
        X = np.copy(X)
        t = np.copy(t) 
        self.iterations = self.epochs*num_mini_batches
        """
        returns array with [... (X_n, t_n) ...] for 0 < n < num_mini_batches
        """
        cur_seed = np.random.randint(self.epochs*10**4)
        np.random.seed(cur_seed)
        np.random.shuffle(X)
        np.random.seed(cur_seed)
        np.random.shuffle(t)

        mini_batches_X = np.array_split(X, num_mini_batches)
        mini_batches_T = np.array_split(t, num_mini_batches)

        return zip(mini_batches_X, mini_batches_T)
    
    #halting condition
    def __check(self, iteration, old_weights):
        if np.linalg.norm(old_weights - self.weights) < self.tolerance:
            print("tolerance reached")
            return False
        if self.time >= self.epochs: return False
        return True

    #functions for adaptive learning rate
    def __cyclic(self, dummy1, dummy2):
        x = self.curr_iter/self.iterations
        change = 10 #==0.1
        frac = 0.8
        x0 = frac/2
        if x < x0:
            self.m0 = self.maxm0*(1 - x/(x0*change))
            self.gamma = self.maxgamma*(0.9 + x/(x0*change))
        elif (x > x0) and (x < frac):
            self.m0 = self.maxm0*(0.9 + (x - x0)/(x0*change))
            self.gamma = self.maxgamma*(1 - (x - x0)/(x0*change))
        else:
            self.m0 = self.maxm0
            self.gamma = 0.9*self.maxgamma*(1 - 4*(x-frac))
        return self.gamma

    def __decay(self, gamma0, t):
        return gamma0 / ( gamma0*t +1)
        

    def __momentum(self):
        self.v = self.v * self.m0 + self.gamma*self.delta


