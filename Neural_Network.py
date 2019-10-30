# -*- coding: utf-8 -*-
import autograd.numpy as np
import autograd
from SGD import SGD
import pandas as pd
class Neural_Network:

    def __init__(self, number_of_nodes, active_fn, cost_function,regularization =('none', 1e-15), log = True):
        """
        Initialize a NN
        number of nodes: -list of number of nodes including input and output layer
                         - at least number of inputs and output nodes
        active_fn:       - list of activation functions
                         - strings 'sigmoid', 'tanh', 'relu' and 'softmax' are supported
        cost_function:   - either str 'mse' or 'classification' using cross entropy
        regularization:  - regularization schme for cost function, either 'l1' or 'l2' and strenght
        log:             - creats table of information and keeps track of evolution of NN during training

        Methods:
        -feedforward:   claculate output of NN based on data shape (#features, #samples)
        -training:      trains the NN, usage of SGD class and backpropagation
        """

        self.nodes = number_of_nodes
        self.layers = len(number_of_nodes)

        #initalze biases shape (#nodes l, 1)
        self.biases = [np.random.randn(i, 1) for i in self.nodes[1:]]
        #initalize weights shape (#nodes l+1, #nodes l)
        self.weights = [np.random.randn(i, j)/np.sqrt(j) for j, i in zip(self.nodes[:-1], self.nodes[1:])]

        # setup up a list of activation functions only one literal
        if active_fn == 'sigmoid':
            self.functions = [Neural_Network.sigmoid_act for i in range(0, self.layers-1)]
        elif active_fn == 'tanh':
            self.functions = [Neural_Network.tanh_act for i in range(0, self.layers-1)]
        else:
            d = {'sigmoid': Neural_Network.sigmoid_act, 'tanh': Neural_Network.tanh_act, 'softmax':Neural_Network.softmax_act, 'relu': Neural_Network.relu_act}
            self.functions = [d[name] for name in active_fn]
        #derivative of layer activation functions
        self.functions_prime = [autograd.elementwise_grad(l, 1) for l in self.functions]

        self.reg = regularization

        # set up cost function
        if cost_function == 'classification':
            self.cost_function = Neural_Network.cross_entropy
            self.functions[self.layers - 2] = Neural_Network.softmax_act
            self.has_acc = True
        if cost_function == 'mse':
            self.cost_function = Neural_Network.mse
            self.has_acc = False

        self.log = False
        if log:
            self.log = True
            self.call =0
            #creat topology mapping
            self.mapping = str(self.nodes[0])
            for i in range(1, self.layers):
                self.mapping += ' : ' + str(self.nodes[i])
                if type (active_fn) == list :
                    self.mapping += '_' + active_fn[i-1]
                else:
                    self.mapping += '_' + active_fn
            
            self.toi = pd.DataFrame(columns=["number of layers", "nodes per layer", 
                                        "epoch", "batch size",
                                        "learning rate","momentum parameter","lambda", "stopping tol",
                                         "cost", "accuracy", "data set"])


    def feedforward(self, data):
        '''
        Feed an initial input data, this is feed to calculate the
        activation a, this is then feed in again
        as an input for the next layer, and so on for each layer,
        till we reach the output layer L.
        '''
        data = np.copy(data)
        self.activations = [data]
        self.z = [0]
        a = data
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            z = np.matmul(weight, a) + bias
            self.z.append(z)
            a = function(self, z)
            self.activations.append(a)
        return a


    def __backpropagation(self, f_z, target):
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
        f_z = np.copy(f_z)
        target = np.copy(target)
        Neural_Network.feedforward(self, f_z)
        #set all inputs for cost function
        self.gradient.weights = (self, self.biases[self.layers -2], target)
        self.gradient.run_minibatch((f_z, target), update_weight= False)
        delta = self.gradient.delta# contains learning rate and momentum

        current_weights = np.copy(self.weights) #current weights before adjustment
        current_biases = np.copy(self.biases)

        # looping through layers
        for i in reversed(range(1, self.layers)):
            self.activations[i-1] = np.mean(self.activations[i-1], axis = 1, keepdims = True)
            delta_W = np.matmul(delta, self.activations[i-1].T)
            if self.lmbd > 0.0:
                delta_W += self.lmbd * current_weights[i-1] # or 1/n taking the mean, lambda is penalty on weights

            self.weights[i-1] = current_weights[i-1] - delta_W
            self.biases[i-1]  = current_biases[i-1] -  delta

            if i > 1:
                a_prime = (self.functions_prime[i-1](self, self.z[i-1])).mean(axis = 1, keepdims = True)
                delta = np.matmul(current_weights[i-1].T, delta) * a_prime

    def training(self, data, target, epochs, mini_batch_size,
            eta = 0.5, eta_schedule = ('decay',0.1),
            momentum = True, gamma = 0.1,
            lmbd = 0.1, tolerance = 1e-3,
            test_data = None,
            validation_data = None):
        """
        training NN
        data shape (#samples, #features)
        target shape (#samples, #output nodes)
        eta: learning rate
        eta_schedule: (scheme, cycles) 'decay' or 'const', if 'decay' the time is multiplied with cycles
        momentum, gamma, set momentum to true, gamma strength of momentum (gamma=0 ==momentum =False)
        lmbd fraction of old weights taken into change
        test_data/validation_data  (inut, outpur ); input shape (#samples, #features), output shape (#samples, #output nodes)
        """
        data = np.copy(data)
        target = np.copy(target)
        self.gradient = SGD( self.cost_function, epochs = epochs, mini_batch_size = mini_batch_size,
                learning_rate = eta, adaptive_learning_rate = eta_schedule[0],
                momentum = momentum, m0 = gamma)

        self.lmbd = lmbd

        samples = data.shape[0]
        num_mini_batches = samples // mini_batch_size

        self.init_eta = eta
        self.tolerance = tolerance

        for self.epoch in range(epochs):
            #run minibatches
            for mini_batch_data, mini_batch_target in self.gradient.creat_mini_batch(data, target, num_mini_batches):
                Neural_Network.feedforward(self, mini_batch_data.T)
                #calls backpropagation to find the new gradient
                Neural_Network.__backpropagation(self, mini_batch_data.T, mini_batch_target.T)

            self.gradient.time += float(eta_schedule[1])* 1 #update time for decay

            # calculate the cost of the epoch
            Neural_Network.__epoch_output(self, data, target, name = 'train')
            if test_data != None:
                Neural_Network.__epoch_output(self, *test_data, name = 'test')

            # Checking if accuracy
            if self.has_acc == True:
                if accuracy_test(self) == True:
                    break

        if validation_data != None:
            Neural_Network.__epoch_output(self, *test_data, name = 'validation')


    def classification_accuracy(self, prediction, y):
        prediction = prediction.T
        prediction = np.argmax(prediction, axis =1)
        y = np.argmax(y, axis =1)
        return len(prediction[prediction == y])/len(y)

    def sigmoid_act(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def tanh_act(self, z):
        return np.tanh(z)

    def softmax_act(self, z):
        # z shape (#nodes, #samples)
        #z -= np.max(z) 
        denom = np.sum(np.exp(z), axis = 0) #(#samples)
        denom = np.array([denom for i in range(z.shape[0])])
        return np.exp(z)/denom

    def relu_act(self, z):
        return np.where( z > 0, z, 0)

    def epoch_cost(self, f_z, target):
        cost = 0.0
        a = Neural_Network.feedforward(self, f_z)
        cost += self.cost_function(self,  self.biases[self.layers -2], target )
        return cost, a


    def cross_entropy(self, b, y):
        z = np.matmul(self.weights[self.layers -2], self.activations[self.layers -2 ]) + b
        a = self.functions[self.layers-2](self, z)
        ret = - np.sum(np.where(y==1, np.log(a), 0) )/y.shape[1]
        if self.reg[0] == 'l1':
            ret -=  float(self.reg[1]) *np.sum(np.abs(b), axis =1).mean()
        if self.reg[0] == 'l2':
            ret -=  float(self.reg[1]) * np.linalg.norm(b, axis =1).mean()
        return ret

        
    def mse(self, b, y):
        z = np.matmul(self.weights[self.layers -2], self.activations[self.layers -2 ]) + b
        a = self.functions[self.layers-2](self, z)
        res = a - y
        ret = np.dot(res[0], res[0])/len(y)
        if self.reg[0] == 'l1':
            ret -=  float(self.reg[1]) * np.sum(np.abs(b), axis = 1).mean()
        if self.reg[0] == 'l2':
            ret -=  float(self.reg[1]) * np.linalg.norm(b,axis = 1).mean()
        return ret

    #make table of information
    def __epoch_output(self, data, target, name='test'):
        data = np.copy(data)
        target = np.copy(target)
        print('Current epoch: ', self.epoch)
        cost, a = Neural_Network.epoch_cost(self, data.T, target.T)
        print('The %s cost is: %.4f' % (name, cost))
        if self.has_acc == True:
            accuracy = Neural_Network.classification_accuracy(self, a, target)
            print('The %s accuracy is : %.4f' % (name, accuracy))
        else:
            accuracy = 'Nan'
        if self.log:
            temp = pd.DataFrame({"number of layers": self.layers, "nodes per layer": self.mapping,
                                        "epoch":self.epoch, "batch size":self.gradient.mini_batch_size,
                                        "learning rate": self.gradient.gamma,"momentum parameter":self.gradient.m0,
                                        "lambda": self.lmbd, "stopping tol": self.tolerance,
                                         "cost": cost, "accuracy":accuracy, "data set":name}, index=[self.call])
            self.toi = self.toi.append(temp)
            self.call += 1
            del temp

    # check if accuracy is constant
    def accuracy_test(self):
        if self.epoch > 5:
            filter = toi['data set'] == 'test'
            accuracy = toi[filter]['accuracy']
            acc_array =  accuracy.to_numpy()
            std_acc = np.std(acc_array[-5:])
            if self.tolerance > std_acc:
                return True
        else:
            return False
