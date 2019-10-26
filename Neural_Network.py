# -*- coding: utf-8 -*-
import autograd.numpy as np
import autograd 

class Neural_Network:

    def __init__(self, training_data, training_target,test_data, test_target,
                 validation_data, validation_target, number_of_nodes, active_fn,
                 eta = 0.01, lmbd = 0.0):

        self.training_data = training_data
        self.training_target = training_target
        self.test_data = test_data
        self.test_target = test_target
        self.validation_data = validation_data
        self.validation_target = validation_target
        self.eta = eta
        self.lmbd = lmbd
        self.nodes = number_of_nodes
        self.layers = len(number_of_nodes)
        #initialise the biases and weights with a random number
        ''' biases is a list of matrices, one matrix for each layer. the size
        is the number of nodesx1
        '''
        self.biases = [np.random.randn(i, 1)*1e-2 for i in self.nodes[1:]]
        #print('init biases', self.biases[0].shape)
        '''weights is a list of matrices, one matrix for each layer.
        e.g if the layers have 10,5,2 nodes, then it creates 5x10 and 2x5 matrices
        to contain the weights
        '''
        self.weights = [np.random.randn(i, j)*1e-2 for j, i in zip(self.nodes[:-1], self.nodes[1:])]
        #print ('length of init weights', len(self.weights))
       # print ('initialised weights', self.weights[2])
        # setup up a list of activation functions
        if active_fn == 'sigmoid':
            self.functions = [Neural_Network.sigmoid_act for i in range(0, self.layers)]
            self.functions_prime = [autograd.elementwise_grad(l, 1) for l in self.functions]
        if active_fn == 'tanh': 
            self.functions = [Neural_Network.tanh_act for i in range(0, self.layers)]
            self.functions_prime = [autograd.elementwise_grad(l, 1) for l in self.functions]
            
        #print('functions_prime', self.functions_prime)
        #self.activations = [np.random.randn(i, 1) for i in self.nodes[1:]]

    def feedforward(self, f_z):
        '''
        Feed an initial input f_z, this is feed to calculate the
        activation also called f_z, this is then feed in again
        as an input for the next layer, and so on for each layer,
        till we reach the output layer L.
        '''
        self.activations = [f_z]
        self.z = [0]
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            #print('f_z shape', f_z.shape)
            #print('weight', weight.shape)
            z = np.dot(weight, f_z) + bias
            self.z.append(z)
            f_z = function(self, z)
            #print ('f_z shape after feeding', f_z.shape)
            self.activations.append(f_z)
        
        self.probabilities =  np.exp(f_z)/np.sum(np.exp(f_z), keepdims = True) #, axis = 1, keepdims = True)
        #print('probabilities', self.probabilities)
        return self.probabilities
    

    def backpropagation(self, f_z, target):
        '''
        Description:
        Backpropagation minimise the error and
        calculates the gradient for each layer,
        working backwards from last layer L. In
        this way, weights which contribute to large
        errors can be updated by a feed forward.

        (Need to work differently on hidden layers and output
        How to do this on different layers depend on dimensions of f_z)
        ---------------------------------------
        Parameters:
        - data (corresponding to Y)
        - X
        - f_z: activation (function a^l?)
        - prob: probabilities
        - lambda is penalty for weigths
        ----------------------------------------
        '''
        
       
        self.probabilities = Neural_Network.feedforward(self, f_z)
        
        # setting the first layer self, W, b, a_h, y):
        '''
        error_W = autograd.grad(Neural_Network.cross_entropy, 1)(self, self.weights[self.layers-2],
                               self.biases[self.layers-2] , self.activations[self.layers-2], target)
        error_b = autograd.grad(Neural_Network.cross_entropy, 2)(self, self.weights[self.layers-2],
                               self.biases[self.layers-2] , self.activations[self.layers-2], target)
        '''
        
        #print ('activation of last hidden layer', self.activations[self.layers-2].shape)
        
        delta = autograd.grad(Neural_Network.cross_entropy, 2)(self, self.weights[self.layers-2],
                               self.biases[self.layers-2] , self.activations[self.layers-2], target)
        
        self.current_weights = self.weights #current weights before adjustment
        
        #print ('delta shape', delta.shape)
        
        # looping through layers
        for i in reversed(range(1, self.layers)): # f_z: (batch,nodes)
            
            #print('activations', self.activations[i-1].T.shape)
            #print ('layers', self.layers)
            #print ('i layer', i)
            #print ('self.weights[i] shape', self.weights[i].shape)
            #print('error now shape', error_now.shape)
            #print('activations shape', self.activations[i].T.shape)
           
            self.activations[i-1] = np.mean(self.activations[i-1], axis = 1, keepdims = True)
            #print ('delta shape', delta.shape)
            #print('activations after mean', self.activations[i-1].T.shape)
            delta_W = np.matmul(delta, self.activations[i-1].T)
            
            
            #self.now_bias_gradient = np.sum(error_now * prime, axis = 0, keepdims = True).T
            #print ('bias grad shape', self.now_bias_gradient.shape)
            #print('weights grad shape', self.now_weights_gradient.shape)
            
            
            if self.lmbd > 0.0:
                #print ('now_weights shape', self.now_weights[i].shape)
                delta_W += self.lmbd * self.current_weights[i-1] # or 1/n taking the mean, lambda is penalty on weights

            #initialise the velocity to zero
            v_dw = 0
            v_db = 0
            
            if (self.momentum == True & i == self.layers-1):
                v_dw = v_dw * self.gamma + (1-self.gamma)* delta_W
                v_db = v_db * self.gamma + (1-self.gamma)* delta
                
                #print ('momentum', v_dw)
                #print ('momentumb', v_db)
                #print ('biases i shape before momentum', self.biases[i].shape)
                self.weights[i-1] -=  self.eta * v_dw
                self.biases[i-1] -= self.eta * v_db
                #print ('biases i shape', self.biases[i].shape)
            else:    
                self.weights[i-1] -= self.eta * delta_W
                self.biases[i-1] -= self.eta * delta
                
            #print ('weights transp ', self.current_weights[i-2].T.shape)
            #print ('error W', error_W.shape)
            #error_W = np.matmul(self.now_weights[i-2].T, error_W)
            #error_b = np.matmul(self.now_weights[i-2].T, error_b)
            if i > 1:
                a_prime = (self.functions_prime[i-1](self, self.z[i-1]))
                a_prime = a_prime.mean(axis = 1, keepdims = True)
                #print('prime', a_prime)
                delta = np.matmul(self.current_weights[i-1].T, delta) * a_prime
                      
        return 

# must calculate these in backpropagation: dC_dw , dC_db

    def SGD(self, epochs, mini_batch_size, tolerance = 1, momentum = True):
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
        
        #self.cost_function = cost_function
        self.epochs = epochs
        self.momentum = momentum
        self.num_mini_batches = len(self.training_data) / mini_batch_size
        #print('len of data', len(self.training_data))
        #print ('mini batch size', mini_batch_size)
        #print ('num of mini batches' , self.num_mini_batches)
        self.tol_reached = False
        self.tolerance = tolerance
        self.training_cost = np.zeros(epochs) + 10
        self.training_accuracy = np.zeros(epochs)
        self.test_cost = np.zeros(epochs)
        self.test_accuracy = np.zeros(epochs)
        self.gamma = 0.9
        
        for epoch in range(self.epochs):
            print ('epoch is:', epoch)
            np.random.seed(0)
            np.random.shuffle(self.training_data)
            np.random.seed(0)
            np.random.shuffle(self.training_target)
            mini_batches_data = np.array(np.array_split(self.training_data, self.num_mini_batches))
            mini_batches_target = np.array(np.array_split(self.training_target, self.num_mini_batches))
            
            #print ('mini batches data', mini_batches_data)
            #print ('mini batches target', mini_batches_target)
            
            for mini_batch_data, mini_batch_target in zip(mini_batches_data,mini_batches_target):
                
                #print ('mini batch shape', mini_batch_data.shape)
                #print ('len mb target', len(mini_batch_target))
                #print('mini batch data', mini_batch_data)
                #a = Neural_Network.feedforward(self, mini_batch_data.T)
                #calls backpropagation to find the new gradient
                Neural_Network.backpropagation(self, mini_batch_data.T, mini_batch_target.T)
                #print ('bias after back prop', self.biases)
            #print ('a is: ', a)
            # calculate the cost of the epoch
            cost, a = Neural_Network.epoch_cost(self, self.training_data.T, self.training_target.T)
            print ('a shape', a.shape)
            print('The training cost is:', cost)
            self.training_cost[epoch] = cost
            
            accuracy = Neural_Network.classification_accuracy(self, a, self.training_target)
            print('The training accuracy is :', accuracy)
            self.training_accuracy[epoch] = accuracy
            
            cost, a = Neural_Network.epoch_cost(self, self.test_data.T, self.test_target.T)
            print('Test cost is:', cost)
            self.test_cost[epoch] = cost
            
            accuracy = Neural_Network.classification_accuracy(self, a, self.test_target)
            print('The test accuracy is :', accuracy)
            self.test_accuracy[epoch] = accuracy
        
        self.validation_cost, a = Neural_Network.epoch_cost(self, self.validation_data.T, self.validation_target.T)
        self.validation_accuracy = Neural_Network.classification_accuracy(self, a, self.validation_target)
        
            #if np.min(self.test_cost) < self.tolerance:
            #    return
    def classification_accuracy(self, prediction, y):
        
        prediction = prediction.T
        #prediction = np.where(prediction < 0.5, 0 , 1)
        prediction = np.argmax(prediction, axis =1)
        y = np.argmax(y, axis =1)
        return len(prediction[prediction == y])/len(y)
        #print(pred)
        #return np.sum(np.where(np.all(prediction==y, axis = 1), 1, 0))/ len(prediction)
        

        
    '''    
    def classification_accuracy(self, a , target):
        accuracy = 0
        #print('the len of target is:', len(target))
        a = np.where(a < 0.5, 0 , 1) # set the output to 0 or 1 depending if the input is less or greater than 0.5
        for x, y in zip(a,target):
            print ('x', x)
            print ('y', y)
            if x == y:
                accuracy += 1
        return accuracy / len(target)
    '''
    def sigmoid_act(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def tanh_act(self, z):
        e_z = np.exp(z)
        e_neg_z = np.exp(-z)
        return (e_z - e_neg_z) / (e_z + e_neg_z)
    
    def epoch_cost(self, f_z, target):
        cost = 0.0
        #for f_z, t in zip(data, target):
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            #print('weights shape', weight.shape)
            #print('bias shape', bias.shape)
            #print('f_z shape', f_z.shape)
            z = np.dot(weight, f_z) + bias
            f_z = function(self, z)
            
        a =  np.exp(f_z)/np.sum(np.exp(f_z), keepdims = True)
        cost += Neural_Network.cross_entropy_cost_function(self, a, target)
        return cost, a
    
    def cross_entropy_cost_function (self, p, y):
        return - np.sum(np.where(y==1, np.log(p), 0))/y.shape[1]
    
    def cross_entropy(self, W, b, a_h, y):
        z = np.matmul(W, a_h) + b
        a = self.functions[self.layers-1](self, z)
        p =  np.exp(a)/np.sum(np.exp(a), keepdims = True)
        return  - np.sum(np.where(y==1, np.log(p), 0))/y.shape[1] #np.sum(-y * np.log(p) + (1 - y) * np.log(1 - p))
    

        
