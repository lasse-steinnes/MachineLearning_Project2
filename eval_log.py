import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from helper_functions import parse_data
import LogisticRegression as Log

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

valid_size = 0.1
lamb = [('l1',1e-9), ('l1',1e-6), ('l1',1e-3),
        ('l2',1e-9), ('l2',1e-6), ('l2',1e-3),
        ('none',0)]
learning =[1 , 0.5, 0.1, 0.01]
adaptiv = ['const', 'decay']
batchsize = [10, 40, 60, 80]
epochs = [1, 10, 40, 60]
mi = 10**4
X, y = parse_data(df, "default payment next month", unbalanced= False )
"""
stats = pd.DataFrame( np.zeros((2 * 2* 4 ** 3 * 7, 8)),
                columns = ["balanced", "learning rate", "adaptive learning", "epochs", "batch size", "regularization" ,"regularization parameter", "accuracy" ])
index = 0
for balance in [False, True]:
    X, y = parse_data(df, "default payment next month", unbalanced= balance )
    N = len(y)
    print("N = "+str(N)+"; 0: %.2f; 1: %.2f" % tuple(np.bincount(y)/N))
    X_trian, X_eval, y_train, y_eval = train_test_split(X,y, test_size =valid_size)
    for l in lamb:
        for gamma in learning:
            for a in adaptiv:
                for bs in batchsize:
                    for ep in epochs: 
                        clf_own = Log.LogisticRegression(max_iter = mi, mini_batch_size=bs, epochs = ep, learning_rate=gamma, adaptive_learning_rate=a,
                                                        regualrization= l, logging = True)
                        clf_own.fit(X_trian,y_train, split = True, fraction = valid_size/(1-valid_size))
                        d =clf_own.evaluate(X_eval,y_eval, data_set ="evaluate")
                        stats.iloc[index] = [balance, gamma, a, ep, bs, l[0], l[1], d["accuracy"]]
                        index += 1
stats.to_csv("Results/LogReg/hyper_par.csv")
"""


#sklearn
clf  = LogisticRegression( C = 1, solver ='sag', max_iter = mi, class_weight='balanced')
X_trian, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
clf.fit(X_trian, y_train)
own  = 1#clf_own.evaluate(X_test,y_test)["accuracy"]
print("Own Log. Reg: %.3f vs SKL Log.Reg: %.3f" %(own, clf.score(X_test, y_test)))