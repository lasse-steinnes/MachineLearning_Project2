# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:36:24 2019

@author: james
"""

import Neural_Network
import numpy as np
import pandas as pd
import OneHot
from helper_functions import load_terrain, normalize,matDesign
from sklearn.preprocessing import MinMaxScaler

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

target = (df[['default payment next month']].copy()).to_numpy()
data = (df.drop(columns =['default payment next month', 'ID']).copy()).to_numpy()
#print (data.head())
#print(target.head())
'''
#############################
#Neural Network for regression
#############################
reduction = 36 # Similar to project 1
x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
x,y,z = normalize(x,y,z) # normalize training, use to normalize
order = 96
X = matDesign(x,y, order) # Design matrix, should test for several different orders
'''

curr_seed = 0 # the seed used for random shuffles

nodes_schedule = np.array([[23,20,10,2], # Might be 29 in first
                 [23,10,5,2]])
etas = np.array([0.005, 0.8]) #
lmbdas = np.array([0.0, 0.8])
epochs_schedule = np.array([10]) # Increase number
mini_batch_sizes = np.array([15]) # Increase size

tolerance = 0.2
momentum = True

#initialise table of information
toi = pd.DataFrame(columns = ['layers', 'nodes', 'epochs', 'mini_batch_size',
                   'eta', 'lmbda', 'training_accuracy', 'test_accuracy','validation_accuracy', 'cost_function', 'activation_function'])

toi_main = pd.DataFrame(columns = ['layers', 'nodes', 'epochs', 'mini_batch_size',
                   'eta', 'lmbda', 'training_accuracy', 'test_accuracy','validation_accuracy','cost_function', 'activation_function'])

# 10-fold cross validation

k = 10
np.random.seed(curr_seed)
np.random.shuffle(data)
np.random.seed(curr_seed)
np.random.shuffle(target)

print( target.shape)
onehot = OneHot.OneHot()
target = onehot.encoding(target[:,0]) # need to oneHot encode the input
print (target.shape)

data_folds = np.array(np.array_split(data, k+1))
target_folds = np.array(np.array_split(target, k+1))

# train test split; stratify
# confusion matrix for NN, F1-score
# Might learn worse from high bias --> downsampling (50 50 (60/40) default or not) or upsampling

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
    print ('test', test_data)
    print ('validation', validation_data)
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

                                 toi['layers'] = [len(nodes) for i in range(0, epochs)]
                                 toi['nodes'] = [nodes for i in range(0, epochs)]
                                 toi['epochs'] = [epochs  for i in range(0, epochs)]
                                 toi['mini_batch_size'] = [mini_batch_size  for i in range(0, epochs)]
                                 toi['eta'] = [eta  for i in range(0, epochs)]
                                 toi['lmbda'] = [lmbda  for i in range(0, epochs)]
                                 toi['training_accuracy'] = net.training_accuracy
                                 toi['test_accuracy'] = net.test_accuracy
                                 toi['validation_accuracy'] = net.validation_accuracy
                                 toi['cost_function'] = ['cross entropy'  for i in range(0, epochs)]
                                 toi['activation_function'] = ['sigmoid' for i in range(0, epochs)]
                                 toi_main = toi_main.append(toi)


toi.to_csv('./Results/NeuralNetwork/ann.csv')
