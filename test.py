import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from helper_functions import parse_data
import LogisticRegression as lr

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

def test_one_hot():
    y = df["PAY_0"].to_numpy()

    l_uni = len(np.unique(y))
    l_y = len(y)

    y_hot, y_key = one_hot(y)

    y_decode = np.zeros(l_y)
    for i in range(l_y):
        y_decode[i] = y_key[np.argmax(y_hot[i])]

    assert np.all(y_decode==y)
    assert np.all(y_hot.shape == (l_y,l_uni))

def test_model():
    x = df.drop(columns =["PAY_0"]).to_numpy()
    X = model(x,1)
    assert np.all( X.shape == df.shape)
    assert np.all( X[:,0] - 1 == 0 )

def test_logistic():
    X, y = parse_data(df, "default payment next month" )
    N = len(y)
    print("N = "+str(N)+"; 0: %.2f; 1: %.2f" % tuple(np.bincount(y)/N))
    X_trian, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
    clf = lr.LogisticRegression(max_iter = 10**4, mini_batch_size=30, epochs = 100, learning_rate=1, adaptive_learning_rate='const', logging = True)
    clf.fit(X_trian,y_train)
    clf.evaluate(X_test,y_test)
    print(clf.logs)
    print(y_test[-5:],clf.predict(X_test[-5:], decoded=True))
    print(clf.confusion_matrix(X_test,y_test))

test_logistic()

clf  = LogisticRegression( )
X, y = parse_data(df, "default payment next month" )
X_trian, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
clf.fit(X_trian, y_train)
print(clf.score(X_test, y_test))