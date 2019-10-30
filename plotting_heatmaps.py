# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:49:36 2019

@author: anjat
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from plotting_functions import plotting_mse

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based) 
toi = pd.read_csv('./Results/NeuralNetwork/nn.csv') 
 
#	number of layers	nodes per layer	epoch	batch size	learning rate	momentum parameter	cost	accuracy	data set

##################  #eta heatmap #############################################
eta_filter = (toi['data set'] =='test')

f, ax = plt.subplots(figsize=(20, 6))

eta_data = pd.pivot_table(data = toi[eta_filter],
                    values = 'accuracy',
                    index = 'initial learning rate',
                    columns = 'epoch')

g = sns.heatmap(eta_data, annot=True, fmt=".2f",linewidths=0)
plt.savefig(fname ='./Results/NeuralNetwork/eta_heatmap.pdf', dpi='figure', format= 'pdf')

plt.show()
plt.close()

##################### batch size heatmap #####################################

batch_filter = (toi['data set'] == 'test')

f, ax = plt.subplots(figsize=(20, 6))

eta_data = pd.pivot_table(data = toi[batch_filter],
                    values = 'accuracy',
                    index = 'batch size',
                    columns = 'epoch')

g = sns.heatmap(eta_data, annot=True, fmt=".2f",linewidths=0)
plt.savefig(fname ='./Results/NeuralNetwork/batch_heatmap.pdf', dpi='figure', format= 'pdf')