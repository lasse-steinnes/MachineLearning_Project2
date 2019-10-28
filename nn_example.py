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

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size =0.1)


nn = Neural_Network([23,  40,       2],
                         ['tanh',  'softmax'],
                    'classification', regularization=('l2', 1e-2))

nn.training(X_train, y_train,
            200, mini_batch_size=15,
            eta =0.01, eta_schedule='decay',
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
plt.tight_layout()
plt.show()

X_eval, y_eval = parse_data(df, "default payment next month", unbalanced= False)
y_eval = onehot.encoding(y_eval)
p = nn.feedforward(X_eval[:,1:].T)
print(onehot.confusion(p.T, np.argmax(y_eval, axis =1)))
