
"""
little script to visualize training beahavior
"""
import matplotlib.pyplot as plt
import seaborn as sns
from LogisticRegression import LogisticRegression as LR 
from helper_functions import parse_data
import pandas as pd 
import numpy as np
import sys
from sklearn.model_selection import train_test_split

keys = {"learning_rate" : 0.01, "adaptive_learning_rate" : 'decay',
        "mini_batch_size": 10, "epochs" : 20, "regularization": ('l2', 1e-5)}
keys_type = {"learning_rate" : float, "adaptive_learning_rate" : str,
        "mini_batch_size": int, "epochs" : int, "regularization": tuple}
i = 1
while i < len(sys.argv):
    if keys_type[sys.argv[i]] == tuple :
        keys[sys.argv[i]] = (sys.argv[i+1], sys.argv[i + 2])
        i += 3
        continue
    keys[sys.argv[i]] = keys_type[sys.argv[i]](sys.argv[i+1])
    i += 2


df = pd.read_excel("/home/lukas/Documents/MachineLearning_Project2/default of credit card clients.xls", header = 1)
df = df.drop(columns =["ID"])

X,y = parse_data(df, "default payment next month", unbalanced=False) 
X_t, x_t, Y_t, y_t = train_test_split(X, y, test_size = 0.2)
X_t, X_e, Y_t, y_e = train_test_split(X_t, Y_t, test_size = 0.3)
cll = LR( **keys, max_iter=10**5, logging=True)
cll.fit(X_t,Y_t, split=True, test = (X_e, y_e))
acc = cll.evaluate(x_t,y_t)
print(cll.weights)
i = 1
plt.figure(figsize=(10,10))
plt.suptitle("%s $\gamma = %.4f$, batch size = %i" % (keys["adaptive_learning_rate"], keys["learning_rate"],keys["mini_batch_size"]), fontsize = 26)
for key, val in acc.items():
    plt.subplot(2,2, i)
    plt.title( key + " = %.3f" % val, fontsize = 24)
    sns.lineplot(x='epoch', y=key, hue='data set', data = cll.logs)
    if i > 2:
        plt.xlabel("epochs", fontsize = 22)
    else: 
        plt.xlabel("")
    plt.ylabel(key, fontsize = 22)
    plt.xticks(fontsize =20)
    plt.yticks(fontsize =20)
    vmin, vmax = cll.logs[key].iloc[2:].min(), cll.logs[key].iloc[2:].max()
    shift = 0.05
    plt.ylim( (1 - np.sign(vmin)*shift) * vmin , (1 + np.sign(vmax)*shift) * vmax)
    i += 1
plt.tight_layout(pad = 1.5, rect = (0, 0, 1, 0.95))
plt.show()
