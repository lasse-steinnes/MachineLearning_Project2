# -*- coding: utf-8 -*-
import numpy as np

class Neural_Network:

    def __init__(self, training_data, number_of_nodes, eta = 0.01, lamda = 0.0):

        self.training_data = training_data
        self.training_data, self.training_target = zip(*self.training_data)

        self.nodes = number_of_nodes
        self.layers = len(number_of_nodes)
        #initialise the biases and weights with a random number
        ''' biases is a list of matrices, one matrix for each layer. the size
        is the number of nodesx1
        '''
        self.biases = [np.random.randn(i, 1) for i in self.nodes[1:]]

        '''weights is a list of matrices, one matrix for each layer.
        e.g if the layers have 10,5,2 nodes, then it creates 5x10 and 2x5 matrices
        to contain the weights
        '''
        self.weights = [np.random.randn(i, j) for j, i in zip(self.nodes[:-1], self.nodes[1:])]

        # setup up a list of activation functions
        self.functions = [Neural_Network.sigmoid_act for i in range(0, self.layers)]

    def feedforward(self, f_z):
        '''
        Feed an initial input f_z, this is feed to calculate the
        activation also called f_z, this is then feed in again
        as an input for the next layer, and so on for each layer,
        till we reach the output layer L.
        '''
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            z = np.dot(weight, f_z) + bias
            f_z = function(z)
            prob_term = np.exp(f_z)
            self.probabilities =  prob_term/np.sum(prob_term, axis = 1, keepdims = True)
        return self.f_z, self.probabilities

    def backpropagation(self,f_z,data):
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
        self.f_z, self.probabilities = feed_forward(f_z)

        # setting the first layer
        error_now = self.probabilities - self.data
        now_weights = self.weights
        # Might need these
        #self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        #self.output_bias_gradient = np.sum(error_output, axis=0)

        # looping through layers
        for i in reversed(range(1,len(f_z))): # f_z: (batch,nodes)

            error_back = np.matmul(error_now, self.weights[i].T)* self.f_z[i]*(1 - self.f_z[i]) # prevlayer*number of targets (binary 1)

        # Using errors to calculate gradients
            self.now_weights_gradient = np.matmul(self.f_z[i].T, error_now)
            self.now_bias_gradient = np.sum(error_now, axis=0)

            if self.lmbd > 0.0:
                self.now_weights_gradient += self.lmbd * self.now_weights # or 1/n taking the mean, lambda is penalty on weights

            self.weights[i] -= self.eta * self.now_weights_gradient
            self.biases[i] -= self.eta * self.now_bias_gradient
            error_now = error_back
            now_weights = self.weights[i]
        return self.now_weights_gradient,self.now_bias_gradient

# must calculate these in backpropagation: dC_dw , dC_db

    def SGD(self, cost_function, epochs =10, mini_batch_size = 10, learning_rate = 0.5, tolerance = 1, momentum =True):
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
        self.epochs = epochs

        self.num_mini_batches = self.training_x / mini_batch_size
        self.learning_rate = learning_rate
        self.tol_reached = False
        self.tolerance = tolerance
        self.store_cost = []
        self.momentum = momentum
        self.gamma = 0.9
        for epoch in range(self.epochs):
            np.random.seed(0)
            np.random.shuffle(self.training_data)
            np.random.seed(0)
            np.random.shuffle(self.training_target)
            mini_batches_data = np.array(np.array_split(self.training_data, self.num_mini_batches))
            mini_batches_target = np.array(np.array_split(self.training_target, self.num_mini_batches))
            for mini_batch_data, mini_batch_target in zip(mini_batches_data,mini_batches_target):

                a = Neural_Network.feedforward(mini_batch_data)
                Neural_Network.update_mini_batch(mini_batch_data, mini_batch_target)

                #initialise the velocity to zero
                v_dw = 0
                v_db = 0

                #calls backpropagation to find the new gradient
                dC_dw , dC_db = Neural_Network.backpropagation(mini_batch_data, mini_batch_target)

                if (self.momentum == True):
                    v_dw = v_dw * self.gamma + (1-self.gamma)* dC_dw
                    v_db = v_db * self.gamma + (1-self.gamma)* dC_dw

                    self.weights = self.weights - self.learning_rate * v_dw
                    self.biases = self.biases - self.learning_rate * v_db
                else:
                    self.weights = self.weights - self.learning_rate * dC_dw
                    self.biases = self.biases - self.learning_rate * dC_dw

            # calculate the cost of the epoch
            cost = Neural_Network.cross_entropy_cost_function(a, mini_batch_target)
            print('The cost is:', cost)
            #store the cost
            self.store_cost = self.store_costs.append(cost)
            accuracy = Neural_Network.classification_accuracy(a, mini_batch_target)
            print('accuracy is :', accuracy)

            if self.store_cost.min() < self.tolerance:
                return
    def classification_accuracy(a , target):
        for x, y in zip(a,target):
            accuracy = 0
            if x == y:
                accuracy += 1
        return accuracy

    def sigmoid_act(self, z):
        return 1.0/(1.0+np.exp(-z))

    def tanh_act(self, z):
        e_z = np.exp(z)
        e_neg_z = np.exp(-z)
        return (e_z - e_neg_z) / (e_z + e_neg_z)

    def cross_entropy_cost_function (a, y):
        return np.sum(-y * np.log(a) + (1 - y) * np.log(1 - a))
