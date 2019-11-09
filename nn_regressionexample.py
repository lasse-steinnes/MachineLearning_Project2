from Neural_Network import Neural_Network
from helper_functions import parse_data
import pandas as pd
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import load_terrain, normalize,matDesign
from imageio import imread

#############################
#Neural Network for regression
#############################
reduction = 36 # Similar to project 1
x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
x,y,z = normalize(x,y,z) # normalize training, use to normalize
p_order = np.array([115])

toi = pd.DataFrame(columns=["number of layers", "nodes per layer",
                                        "epoch", "batch size",
                                        "learning rate","initial learning rate","momentum parameter","lambda", "stopping tol",
                                         "cost", "accuracy", "data set", "pol order"])

eta = np.array([0.4])
mini_batch_size = np.array([50])
epochs = np.array([100])
lmbd = np.array([1e-6])
gamma = np.array([0.9])
kfold = 10

functions = np.array(['tanh', 'sigmoid'])

for order in p_order:
    X = matDesign(x,y, order) # Design matrix, should test for several different orders
    l1 = len(X[0,1:])
    layers = np.array([[l1,40,1]])

    if kfold != 0:
        X_folds = np.array(np.array_split(X[:,1:], kfold))
        y_folds = np.array(np.array_split(z, kfold))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[:,1:], z, test_size =0.1)

    for n in gamma:
        for m in lmbd:
            for k in epochs:
                for j in eta:
                    for i in mini_batch_size:
                        for h in layers:
                            for g in functions:
                                if kfold != 0:
                                    temp = pd.DataFrame(columns=["number of layers", "nodes per layer",
                                            "epoch", "batch size",
                                            "learning rate","initial learning rate","momentum parameter","lambda", "stopping tol",
                                            "cost", "accuracy", "data set","pol order"])
                                    for s in range(0, kfold):
                                        t = s + 1
                                        if s == kfold-1:
                                            t = 1
                                        X_train = np.concatenate(np.delete(X_folds, s and t , 0))
                                        X_test  = X_folds[s]
                                        y_train = np.concatenate(np.delete(y_folds, s and t , 0)) # s and t # find for dataset validation
                                        y_test  = y_folds[s]
                                        X_val = X_folds[t]
                                        y_val = y_folds[t]

                                        nn = Neural_Network(h, g,
                                                'mse', order, regularization=('None', 1e-2))

                                        nn.training(X_train, y_train,
                                                k, mini_batch_size=i,
                                                eta = j, eta_schedule=('decay', 0.01),
                                                momentum=True, gamma = n,
                                                lmbd=m, tolerance=10**-4,
                                                test_data=(X_test, y_test),validation_data = (X_val,y_val))
                                        temp = temp.append(nn.toi)

                                    #print(temp)
                                # find the mean value of the cost and accuracy

                                    mean_temp = temp.groupby(["number of layers", "nodes per layer",
                                        "epoch", "batch size", "learning rate","initial learning rate",
                                        "momentum parameter","lambda", "stopping tol",
                                        "data set","accuracy","pol order"], as_index = False).mean()

                                    toi = toi.append(mean_temp)
                                    del temp

                                else:

                                    nn = Neural_Network(h, g,
                                            'mse', order, regularization=('None', 1e-2))

                                    nn.training(X_train, y_train,
                                            k, mini_batch_size=i,
                                            eta = j, eta_schedule=('decay', 0.01),
                                            momentum=True, gamma = n,
                                            lmbd=m, tolerance=10**-4,
                                            test_data=(X_test, y_test),validation_data = (X_val,y_val))

                                    toi = toi.append(nn.toi)
#################################
"""
Visualising terrain and model output
"""
#################################
z_model = nn.feedforward(X[:,1:].T)
m = int(np.sqrt(len(z_model[0])))
heights = z_model.reshape((m,m))

plt.figure()
ax1 = plt.subplot(122)
ax1.set_title('NN:Regression | MSE = 1.70 E-03 \n p = 115; $\eta$ = 0.4')
ax1.imshow(heights, cmap='viridis')
ax1.set(xlabel='X', ylabel='Y')
plt.axis('off')

m1 = int(np.sqrt(len(z)))
terrain = z.reshape((m1,m1))
ax2 = plt.subplot(121)
ax2.set_title('Processed data')
ax2.imshow(terrain,cmap = 'viridis')
ax2.set(xlabel='X', ylabel='Y')
plt.axis('off')
################################
plt.savefig(fname ='./Results/NeuralNetworkReg/terrain_final.pdf', dpi='figure', format= 'pdf')
plt.show()

toi.to_csv('./Results/NeuralNetworkReg/nn.csv')
