'''
script for running neural network for classificaiton problems specifically the 
credit card data. the network parameters are set to the optimal hyperparameters.
The K-fold cross validation method is used with test, train and validation splits.
'''

from Neural_Network import Neural_Network
from helper_functions import parse_data
import pandas as pd
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)
df = df.drop(columns=["ID"])
X, y = parse_data(df, "default payment next month", unbalanced= False)
X = X[: ,1:]#drop cont column  from model
onehot = OneHot()
y_onehot = onehot.encoding(y)

#shuffle the data before slicing into folds.
curr_seed= 0
np.random.seed(curr_seed)
np.random.shuffle(X)
np.random.seed(curr_seed)
np.random.shuffle(y_onehot)

kfold = 0

if kfold != 0:
    X_folds = np.array(np.array_split(X, kfold))
    y_folds = np.array(np.array_split(y_onehot, kfold))
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size =0.1)

toi = pd.DataFrame(columns=["number of layers", "nodes per layer",
                                        "epoch", "batch size",
                                        "learning rate","initial learning rate","momentum parameter","lambda", "stopping tol",
                                         "cost", "accuracy", "data set"])
#initialisation of hyper parameters.
eta = np.array([0.3])
mini_batch_size = np.array([50])
epochs = np.array([100])
lmbd = np.array([1e-8])
gamma = np.array([0.9])
layers = np.array([[23, 40, 2]])
functions = np.array(['tanh'])

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
                                         "cost", "accuracy", "data set"])
                                for s in range(0, kfold):
                                    #indexing s for the test fold and t for the validation fold.
                                    t = s + 1
                                    if s == kfold-1:
                                        t = 1
                                    X_train = np.concatenate(np.delete(X_folds, s and t , 0))                                    
                                    X_test  = X_folds[s]   
                                    X_validation = X_folds[t]
                                    y_train = np.concatenate(np.delete(y_folds, s and t, 0))
                                    y_test  = y_folds[s] 
                                    y_validation = y_folds[t]
                                    
                                    #initialise the network
                                    nn = Neural_Network(h, g,
                                                'classification', regularization=('l2', 1e-2))
                                    
                                    #train the network
                                    nn.training(X_train, y_train,
                                        k, mini_batch_size=i,
                                        eta = j, eta_schedule=('decay', 0.01),
                                        momentum=True, gamma = n,
                                        lmbd=m, tolerance=10**-4,
                                        test_data=(X_test, y_test),
                                        validation_data=(X_validation,y_validation))
                                    
                                    #temporarily store the results in a table of information.
                                    temp = temp.append(nn.toi)
                                    
                                # find the mean value of the cost and accuracy                                
                                mean_temp = temp.groupby(["number of layers", "nodes per layer",
                                        "epoch", "batch size", "learning rate","initial learning rate",
                                        "momentum parameter","lambda", "stopping tol",
                                        "data set"], as_index = False).mean()
                                #store the mean results
                                toi = toi.append(mean_temp)
                                #delete the temporary data
                                del temp
                            else:
                                
                                nn = Neural_Network(h, g,
                                    'classification', regularization=('l2', 1e-2))

                                nn.training(X_train, y_train,
                                    k, mini_batch_size=i,
                                    eta = j, eta_schedule=('decay', 0.01),
                                    momentum=True, gamma = n,
                                    lmbd=m, tolerance=10**-4,
                                    test_data=(X_test, y_test))
    
                                toi = toi.append(nn.toi)

toi.to_csv('./Results/NeuralNetwork/nn.csv')

plt.figure(figsize=(10,10))

plt.subplot(121)
sns.lineplot(x='epoch', y='cost', hue='data set', data = toi)
plt.xlabel("epochs", fontsize = 22)
plt.ylabel('cost', fontsize = 22)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.savefig(fname ='./Results/NeuralNetwork/epoch_cost.pdf', dpi='figure', format = 'pdf')

plt.subplot(122)
sns.lineplot(x='epoch', y='accuracy', hue='data set', data = toi)
plt.xlabel("epochs", fontsize = 22)
plt.ylabel('accuracy', fontsize = 22)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.tight_layout()
plt.savefig(fname ='./Results/NeuralNetwork/epoch_acc.pdf', dpi='figure', format = 'pdf')
plt.show()

X_eval, y_eval = parse_data(df, "default payment next month", unbalanced= False)
y_eval = onehot.encoding(y_eval)
p = nn.feedforward(X_eval[:,1:].T)
confusion = onehot.confusion(p.T, np.argmax(y_eval, axis =1))
confusion.to_latex('./Results/NeuralNetwork/confusion.tex')
#confusion.to_csv('./Results/NeuralNetwork/confusion.csv')