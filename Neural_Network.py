# -*- coding: utf-8 -*-

from stocastic_gradient_descent import SGD

class Neural_Network:
    
    def __init__(data, number_of_nodes):
    
        self.data = data
        self.num_nodes = number_of_nodes
        self.layers = len(number_of_nodes)
    
    
    def feedforward(self):
        
        
    def backpropagation(self):
        
    
    def SGD(self):
        
        SGD.run_batch(self)
    
    def sigmoid_act(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def tanh_act(self, z):
        e_z = np.exp(z)
        e_neg_z = np.exp(-z)
        return (e_z - e_neg_z) / (e_z + e_neg_z)