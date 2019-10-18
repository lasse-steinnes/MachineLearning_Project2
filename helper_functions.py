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
        print(drop)
        df = df.sort_values(by = [target])
        df.index = np.arange(0,len(y))
        df = df.drop(labels = np.arange(0,drop), axis = 0)
        y = df[target].to_numpy()
        df = df.sample(frac=1, replace=False)

    y = df[target].to_numpy()
    df  = df.drop(columns=[target])
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    pre_X = df.to_numpy()   

    X = model(pre_X, power)

    return X, y
    
class OneHot:
    """
    class that provides one hot encoding for classification problems
    """
    def __init__(self, dictonary = None):
        if dictonary != None:
            self.one_hot_encoding_key = dictonary
            self.one_hot_decoding_key = {index: value for index, value in dictonary.items()}
            self.provided_dict =True
        else:
            self.provided_dict =False
    
    def encoding(self, y):
        """
        computes the one hot encding for the vector y
        returns y in shape (samples, #unique instances)
        """
        uni = np.unique(y)
        l_uni = len(uni)       
        l_y = len(y)
        hot = np.zeros((l_y, l_uni))
        #inferr dict only at first call otherwise it is provided from class
        if not self.provided_dict:
            self.one_hot_encoding_key = {uni[i]: i for i in range(l_uni)}
            self.one_hot_decoding_key = {i:uni[i] for i in range(l_uni)}
            self.provided_dict = True
        #actuall encoding
        for i in range(l_y):
            index  = self.one_hot_encoding_key[y[i]]
            hot[i, index] = 1
        return hot 

    def decoding(self, y):
        """
        decode one hot encoding of prediction
        """
        l_y = len(y)
        pred_class = np.zeros(l_y)
        for i in range(l_y):
            pred_class[i] = self.one_hot_decoding_key[np.argmax(y[i])] 
        return pred_class

