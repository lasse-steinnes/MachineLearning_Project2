# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:36:24 2019

@author: james
"""

import Neural_Network
import numpy as np
import pandas as pd


filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

target = df[['default payment next month']].copy()
data = df[['EDUCATION', 'SEX']].copy()
print (data.head())
print(target.head())


curr_seed = 0 # the seed used for random shuffles

nodes_schedule = np.array([[2,20,10,2],
                 [2,10,5,2]])

etas = np.array([0.5, 0.8])
lmbdas = np.array([0.5, 0.8])
epochs_schedule = np.array([10])
mini_batch_sizes = np.array([1])

tolerance = 0.2
momentum = True

#initialise table of information
toi = pd.DataFrame(columns = ['layers', 'nodes', 'epochs', 'mini_batch_size', 
                   'eta', 'lmbda', 'accuracy', 'cost_function', 'activation_function'])

toi_main = pd.DataFrame(columns = ['layers', 'node', 'epoch', 'mini_batch_size', 
                   'eta', 'lmbda', 'accuracy', 'cost_function', 'activation_function'])

# 10-fold cross validation

k = 10
np.random.seed(curr_seed)
np.random.shuffle(data)
np.random.seed(curr_seed)
np.random.shuffle(target)

data_folds = np.array(np.array_split(data, k+1))
target_folds = np.array(np.array_split(target, k+1))

for i in range(k + 1):
    #train,test, validation split
    j = i + 1
    if i == 10:
        j = 1
    training_data = np.concatenate(np.delete(data_folds, i and j , 0))
    test_data  = data_folds[i]
    validation_data = data_folds[j]
    training_target = np.concatenate(np.delete(target_folds, i and j , 0))
    test_target  = target_folds[i]
    validation_target = target_folds[j]
    print ('train', training_data)
    print ('tst', test_data)
    print ('valida', validation_data)
    # store results and find the average
    
    for epochs in epochs_schedule:
        for nodes in nodes_schedule:
             for mini_batch_size in mini_batch_sizes:
                    for eta in etas:
                         for lmbda in lmbdas:
                                 net = Neural_Network.Neural_Network(training_data, training_target,
                                         test_data, test_target, validation_data, validation_target,
                                         nodes, 'sigmoid', eta, lmbda)                                                                     
                                 net.SGD(epochs, mini_batch_size, tolerance, momentum = True)
                                 
                                 toi['layers'] = len(nodes)
                                 toi['nodes'] = nodes
                                 toi['epochs'] = epochs
                                 toi['mini_batch_size'] = mini_batch_size
                                 toi['eta'] = eta
                                 toi['lmbda'] = lmbda
                                 #toi['accuracy'] = net.accuracy
                                 toi['cost_function'] = 'cross entropy'
                                 toi['activation_function'] = 'sigmoid'
                                 toi_main = toi_main.append(toi)
    
    
toi.to_csv('./Results/NeuralNetwork/ann.csv')                           



