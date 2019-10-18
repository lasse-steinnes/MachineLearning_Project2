# -*- coding: utf-8 -*-



class Neural_Network:
    
    def __init__(data, number_of_nodes):
    
        self.data = data
        self.num_nodes = number_of_nodes
        self.layers = len(number_of_nodes)
        #initialise the biases and weights with a random number
        ''' biases is a list of matrices, one matrix for each layer. the size
        is the number of nodesx1
        '''
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]  

        '''weights is a list of matrices, one matrix for each layer.
        e.g if the layers have 10,5,2 nodes, then it creates 5x10 and 2x5 matrices
        to contain the weights
        '''
        self.weights = [np.random.randn(i, j) for j, i in zip(sizes[:-1], sizes[1:])]

        # setup up a list of activation functions
        self.functions = [Neural_Network.sigmoid_act for i in self.layers]
        
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
        return f_z
        
    def backpropagation(self):
        

    def SGD(self, cost_function, training_data = np.arange(100.0), epochs =10, mini_batch_size = 10, learning_rate = 0.5, tolerance = 1):
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
        self.tol_reached = False
        self.store_cost = []
        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            mini_batches = np.array(np.array_split(self.training_data, self.num_mini_batches))
            for mini_batch in mini_batches:
                a = feedforward(mini_batch, activation_function)
                Neural_Network.update_mini_batch(mini_batch)
            
            # calculate the cost of the epoch
            cost = Neural_Network.cross_entropy_cost_function(a, t)
            #store the cost
            self.store_cost = self.store_costs.append(cost)
            #return a boolean if the tolerance is reached for early stopping
            if self.store_cost.min() < tol:
                return
            
            
    
    def update_mini_batch(self, mini_batch):
    
        #delta_grad_w = [np.zeros(w.shape) for w in self.weights]
        #delta_grad_b = [np.zeros(b.shape) for b in self.biases]
    
        for data in mini_batch:
            #calls backpropagation to find the new gradient or change
            # in gradient
            dC_dw , dC_db = Neural_Network.backpropagation(data)
            
            self.weights = self.weights - self.learning_rate * dC_dw
            self.biases = self.biases - self.learning_rate * dC_dw
            
    def momentum(self):
        
        
        
    def sigmoid_act(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def tanh_act(self, z):
        e_z = np.exp(z)
        e_neg_z = np.exp(-z)
        return (e_z - e_neg_z) / (e_z + e_neg_z)
    
    def cross_entropy_cost_function (a, y):
        return np.sum(-y * np.log(a) + (1 - y) * np.log(1 - a))
