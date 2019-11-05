from Neural_Network import Neural_Network
from helper_functions import parse_data
import pandas as pd
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import load_terrain, normalize,matDesign
import tensorflow as tf


reduction = 36
x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
x,y,z = normalize(x,y,z) #use to normalize
p_order = np.linspace(50,100,6,dtype = int)
for order in p_order:
    X = matDesign(x,y,order) # Design matrix, should test for several different orders
    #X_train, X_test, Y_train, Y_test = train_test_split(X[:,1:], z, test_size =0.1)
    #l1 = len(X[0,1:])

    model = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(X.shape[1], activation='sigmoid'),
                                    tf.keras.layers.Dense(40, activation='sigmoid'),
                                    #tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])
    model.compile(optimizer='adam',
              loss='mse')

### accuracy not important
    av_acc = 0
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X[:,1:], z, test_size =0.1)
        model.fit(X_train, y_train, epochs=5)
        _, acc =model.evaluate(X, y)
        av_acc += acc
        print(av_acc/10)
