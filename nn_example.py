from Neural_Network import Neural_Network
from helper_functions import parse_data
import pandas as pd 
from OneHot import OneHot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
##defalut values

#down sampling 
#momentum 0.9  <- James
#lambda 1e-4   <- Lukas
#epoch 100
#eta 0.3     <-Lasse
#batch size 50 -200     <- vary [50, 100, 150, 200]
#topologies [23, 40, 2], [23, 40, 20, 2], [23, 40, 20, 10, 2], [23, 10, 20, 40, 2]
#sigmoid/tanh all layers
filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)
df = df.drop(columns=["ID"])
X, y = parse_data(df, "default payment next month", unbalanced= False )
X = X[: ,1:]#drop cont column  from model
onehot = OneHot()
y_onehot = onehot.encoding(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size =0.1)


nn = Neural_Network([23,  40,       2],
                         'tanh',
                    'classification', regularization=('l2', 1e-2))

nn.training(X_train, y_train,
            10, mini_batch_size=30,
            eta =0.1, eta_schedule=('decay', 0.1),
            momentum=True, gamma = 0.3,
            lmbd=0.0, tolerance=10**-4,
            test_data=(X_test, y_test))

plt.figure(figsize=(10,10))
print(nn.toi)
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
