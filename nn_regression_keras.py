"""
Compare NN with results from regression using tensorflow keras
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import load_terrain, normalize,matDesign
import tensorflow as tf


reduction = 36
x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
x,y,z = normalize(x,y,z) #use to normalize
p_order = 50

X = matDesign(x,y,p_order)
X = X[:,1:]

model = tf.keras.models.Sequential([
                                tf.keras.layers.Dense(X.shape[1], activation='sigmoid'),
                                tf.keras.layers.Dense(40, activation='sigmoid'),
                                #tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Dense(1, activation='sigmoid')
                                ])
model.compile(optimizer='adam',
              loss='mse', metrics=['mse'])

### accuracy not important

mse = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size =0.1)
    model.fit(X_train, y_train, epochs=5)
    loss_score,_ = model.evaluate(X, z)
    print(model.metrics_names)
    mse += loss_score
print(mse/10)
