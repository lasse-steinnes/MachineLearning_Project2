"""
functions for parsing data frame from credit card data
"""
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        _,c = np.unique(y,return_counts=True)
        drop = int(c[0]- c[1])
        df = df.sort_values(by = [target])
        df.index = np.arange(0,len(y))
        df = df.drop(labels = np.arange(0,drop), axis = 0)
        y = df[target].to_numpy()
        df = df.sample(frac=1, replace=False)

    y = df[target].to_numpy()
    df  = df.drop(columns=[target])
    scaler = StandardScaler() # MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    pre_X = df.to_numpy()   

    X = model(pre_X, power)

    return X, y
    


