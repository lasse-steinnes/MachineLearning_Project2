import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from helper_functions import parse_data
import LogisticRegression as Log

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)
X, y = parse_data(df, "default payment next month", unbalanced=False )

valid_size = 0.1
lamb = 0.001
lr = 1
mi = 10**4

N = len(y)
print("N = "+str(N)+"; 0: %.2f; 1: %.2f" % tuple(np.bincount(y)/N))
X_trian, X_eval, y_train, y_eval = train_test_split(X,y, test_size =valid_size)
clf_own = Log.LogisticRegression(max_iter = mi, mini_batch_size=60, epochs = 20, learning_rate=lr, adaptive_learning_rate='const', logging = True)
clf_own.fit(X_trian,y_train, split = True, fraction = valid_size/(1-valid_size))
clf_own.evaluate(X_eval,y_eval, data_set ="evaluate")
print(clf_own.logs)
print(clf_own.confusion_matrix(X_eval,y_eval))
#sklearn
clf  = LogisticRegression( C = lr, solver ='sag', max_iter = mi)
X_trian, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
clf.fit(X_trian, y_train)
print("Own Log. Reg: %.3f vs SKL Log.Reg: %.3f" %(clf_own.evaluate(X_test,y_test)["accuracy"], clf.score(X_test, y_test)))