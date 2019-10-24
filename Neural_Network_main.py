# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:36:24 2019

@author: james
"""

import Neural_Network
import numpy as np
import pandas as pd

training_data = np.array([[10,5,1,6,50],
                         [10,20,1,50,9],
                         [10,5,1,6,100],
                         [10,5,1,6,10]])

training_target = np.array([1,0,1,0,1])

print (training_data)
nodes = np.array([[5,20,10,2],
                 [5,10,5,2]])

etas = np.array([0.5, 0.8])
lmbdas = np.array([0.5, 0.8])
epochs = np.array([10])
mini_batch_sizes = np.array([1,2])

tolerance = 0.2
momentum = True

#initialise table of information
toi = pd.DataFrame(columns = ['layers', 'node', 'epoch', 'mini_batch_size', 
                   'eta', 'lmbda', 'accuracy', 'cost_function', 'activation_function'])

toi_main = pd.DataFrame(columns = ['layers', 'node', 'epoch', 'mini_batch_size', 
                   'eta', 'lmbda', 'accuracy', 'cost_function', 'activation_function'])
for node in nodes:
    for epoch in epochs:
         for mini_batch_size in mini_batch_sizes:
                for eta in etas:
                     for lmbda in lmbdas:
                             net = Neural_Network.Neural_Network(training_data, training_target,
                                   node, 'sigmoid', eta, lmbda)
                             net.SGD(epoch, mini_batch_size, tolerance, momentum =True)
                             
                             toi['layers'] = len(node)
                             toi['node'] = node
                             toi['epoch'] = epoch
                             toi['mini_batch_size'] = mini_batch_size
                             toi['eta'] = eta
                             toi['lmbda'] = lmbda
                             #toi['accuracy'] = net.accuracy
                             toi['cost_function'] = 'cross entropy'
                             toi['activation_function'] = 'sigmoid'
                             
                             toi_main = toi_main.append(toi)
                             
toi.to_csv('./Results/NeuralNetwork/ann.csv')                           



