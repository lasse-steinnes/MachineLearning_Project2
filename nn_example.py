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
X, y = parse_data(df, "default payment next month", unbalanced= False )
X = X[: ,1:]#drop cont column  from model
onehot = OneHot()
y_onehot = onehot.encoding(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size =0.2)
X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size = 0.1)

nn = Neural_Network([23,  30,       50,      60,     100,   70,      30,    20,       10,       5,      10,     4,         2],
                         ['tanh',   'tanh', 'relu', 'relu', 'tanh', 'tanh', 'sigmoid','sigmoid', 'relu','relu', 'tanh', 'softmax'],
                    'classification')

nn.training(X_train, y_train,
            20, mini_batch_size=30,
            eta =0.001, eta_schedule='decay',
            momentum=True, gamma = 0.3,
            lmbd=0.0, tolerance=10**-4,
            test_data=(X_test, y_test))

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
plt.show()

p = nn.feedforward(X_eval.T)
print(nn.z[7])
print(p.T)
print(onehot.confusion(p.T, np.argmax(y_eval, axis =1)))
