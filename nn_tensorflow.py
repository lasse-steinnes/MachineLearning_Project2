import tensorflow as tf 
import pandas as pd 
from helper_functions import parse_data 
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import numpy as np


filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)
df = df.drop(columns=["ID"])
X_b, y_b = parse_data(df, "default payment next month", unbalanced= True )
X_b = X_b[: ,1:]
X, y = parse_data(df, "default payment next month", unbalanced= False )
X = X[: ,1:]#drop cont column  from model

onehot = OneHot()
y_onehot = onehot.encoding(y_b)
y = onehot.encoding(y)


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(X.shape[1], activation='sigmoid'),
                                    tf.keras.layers.Dense(40, activation='sigmoid'),
                                    #tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(2, activation='softmax')
                                    ])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

av_acc = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_b, y_onehot, test_size =0.2)
    model.fit(X_train, y_train, epochs=5)
    _, acc =model.evaluate(X, y)
    av_acc += acc
print(av_acc/10)



