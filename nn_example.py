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

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size =0.1)

#k = 10
#curr_seed = 0
#np.random.seed(curr_seed)
#np.random.shuffle(X)
#np.random.seed(curr_seed)
#np.random.shuffle(y)
#X_folds = np.array(np.array_split(X, k))
#y_folds = np.array(np.array_split(y, k))

toi = pd.DataFrame(columns=["number of layers", "nodes per layer", 
                                        "epoch", "batch size",
                                        "learning rate","initial learning rate", 
                                        "momentum parameter",
                                         "cost", "accuracy", "data set"])

#for i in range(k):
#    #train, test, validation split
#    j = i + 1
#    if i == 10:
#        j = 1
#    X_train = np.concatenate(np.delete(X_folds, i and j , 0))
#    X_test  = X_folds[i]
#    X_eval = X_folds[j]
#    y_train = np.concatenate(np.delete(y_folds, i and j , 0))
#    y_test  = y_folds[i]
#    y_eval = y_folds[j]
   


#etas = np.linspace(0.01,1,10)
#etas = np.array([0.6,0.65,0.7,0.75,0.8])
eta = np.array([0.8])
mini_batch_size = np.array([40, 50, 60])

for j in eta:
    for i in mini_batch_size:
        nn = Neural_Network([23,  40,       2],
                         ['tanh',  'softmax'],
                    'classification', regularization=('l2', 1e-2))
    
        nn.training(X_train, y_train,
            30, mini_batch_size=i,
            eta = j, eta_schedule=('decay', 0.000001),
            momentum=True, gamma = 0.9,
            lmbd=0.0, tolerance=10**-4,
            test_data=(X_test, y_test))
        
        toi = toi.append(nn.toi)

toi.to_csv('./Results/NeuralNetwork/nn.csv') 

plt.figure(figsize=(10,10))

plt.subplot(121)
sns.lineplot(x='epoch', y='cost', hue='data set', data = nn.toi)
plt.xlabel("epochs", fontsize = 22)
plt.ylabel('cost', fontsize = 22)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)

plt.subplot(122)
sns.lineplot(x='epoch', y='accuracy', hue='data set', data = nn.toi)
plt.xlabel("epochs", fontsize = 22)
plt.ylabel('accuracy', fontsize = 22)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.tight_layout()
plt.show()

X_eval, y_eval = parse_data(df, "default payment next month", unbalanced= False)
y_eval = onehot.encoding(y_eval)
p = nn.feedforward(X_eval[:,1:].T)
print(onehot.confusion(p.T, np.argmax(y_eval, axis =1)))
