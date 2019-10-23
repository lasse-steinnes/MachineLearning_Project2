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



