import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def model(data, power):
    """
    creat the design matrix of data to given power
    currently just linear!
    """
    samples, features = data.shape
    ones = np.ones(samples)
    ret = np.zeros((features + 1, samples))
    ret[0] = ones
    ret [1:] = data.T
    return ret.T

def one_hot(y):
    """
    computes the one hot encding for the vector y
    returns y in shape (samples, #unique instances)
    """
    uni = np.unique(y)
    l_uni = len(uni)
    l_y = len(y)
    hot = np.zeros((l_y, l_uni))

    encoding = {uni[i]: i for i in range(l_uni)}
    decoding = {i:uni[i] for i in range(l_uni)}
    
    for i in range(l_y):
        index  = encoding[y[i]]
        hot[i, index] = 1
    return hot, decoding


def parse_data(df, target, power = 1, unbalanced = True):
    """
    df: pandas.dataframe
    target: column name of target in df
    power: int > 1 (currently only power=1)
    function which parses a pandas data frame to a polynomial model X of power 
    data is scaled such that [min, max] -> [0,1] in each data column
    and a one_hot encoded target y for logistic regression including a dict for translation
    """
    if unbalanced:
        y = df[target].to_numpy()
        y,c = np.unique(y,return_counts=True)
        drop = int(c[y[0]] - c[y[1]])
        df = df.sort_values(by = [target])
        df = df.drop(index = np.arange(0,drop))
        df = df.sample(frac=1, replace=True)

    y = df[target].to_numpy()
    df  = df.drop(columns=[target])
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    pre_X = df.to_numpy()   

    X = model(pre_X, power)

    return X, y
    
    


