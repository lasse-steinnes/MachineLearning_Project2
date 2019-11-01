from Neural_Network import Neural_Network
from helper_functions import parse_data
import pandas as pd
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import load_terrain, normalize,matDesign

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)
df = df.drop(columns=["ID"])
X, y = parse_data(df, "default payment next month", unbalanced= False )
X = X[: ,1:]#drop cont column  from model
#onehot = OneHot()
#y_onehot = onehot.encoding(y)

#############################
#Neural Network for regression
#############################
reduction = 36 # Similar to project 1
x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
x,y,z = normalize(x,y,z) # normalize training, use to normalize
p_order = np.linspace(50,100,6,dtype = int)

toi = pd.DataFrame(columns=["number of layers", "nodes per layer",
                                        "epoch", "batch size",
                                        "learning rate","initial learning rate","momentum parameter","lambda", "stopping tol",
                                         "cost", "accuracy", "data set", "pol order"])

eta = np.array([0.25,0.3,0.4,0.5])
mini_batch_size = np.array([50,100,150,200])
epochs = np.array([100])
lmbd = np.array([1e-4])
gamma = np.array([0.9])

functions = np.array(['tanh', 'sigmoid'])

for order in p_order:
    X = matDesign(x,y, order) # Design matrix, should test for several different orders
    X_train, X_test, y_train, y_test = train_test_split(X[:,1:], z, test_size =0.1)
    l1 = len(X[0,1:])

    layers = np.array([[l1,40,1],
                      [l1,40,20,1],
                      [l1,40,20,10,1],
                      [l1,10,20,40,1]])
    for n in gamma:
        for m in lmbd:
            for k in epochs:
                for j in eta:
                    for i in mini_batch_size:
                        for h in layers:
                            for g in functions:
                                nn = Neural_Network(h, g,
                                        'mse', order, regularization=('l2', 1e-2),)
                                nn.training(X_train, y_train,
                                        k, mini_batch_size=i,
                                        eta = j, eta_schedule=('decay', 0.01),
                                        momentum=True, gamma = gamma,
                                        lmbd=m, tolerance=10**-4,
                                        test_data=(X_test, y_test))

                                toi = toi.append(nn.toi)

toi.to_csv('./Results/NeuralNetwork/nn.csv')
