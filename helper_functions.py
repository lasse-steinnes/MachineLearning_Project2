"""
functions for parsing data frame from credit card data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imageio import imread

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
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    pre_X = df.to_numpy()

    X = model(pre_X, power)

    return X, y
'''
# Downsampling
def downsampling:
#    Function to downsample data. This is a way of
#    handling imbalanced classes, by reshaping data
#    to 50 - 50 of both classes.

    return downsampled
'''
# Loading the terrain data
def load_terrain(imname, sel = 4): #select every fourth
    """
    This function loads the terrain data. The data
    is then reduced by selecting every sel (eg. every 4th. element).
    It then flattens the reduced matrix and returns z(x,y) - height,
    and x,y pixel index.
    """
    terrain = imread('{:}.tif'.format(imname))
    # Show the terrain
    """
    plt.figure()
    plt.title(imname)
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    """
    #reducing terrain data
    N = len(terrain[0,::sel]) # length reduced columns
    n = len(terrain[::sel,0]) # length reduced rows
    NN = len(terrain[0,:]) # number of columns total
    nn = len(terrain[:,0]) # number of rows total
    #reducing by column
    reduced  = np.zeros((nn,N))
    for i in range(nn):
            reduced[i,:] = terrain[i,::sel]
    #reduce by rows
    reduced2 = np.zeros((n,N))
    for j in range(N):
        reduced2[:,j] = reduced[::sel,j]
    #flattening
    z = reduced2.flatten()
    # creating arrays for x and y
    x_range = np.arange(1,n+1)
    y_range = np.arange(1,N+1)
    X,Y = np.meshgrid(x_range,y_range)
    x = X.flatten();y = Y.flatten()
    return x,y,z

def normalize(x,y,z, rescale = True):
    """
    normalize x,y, z
    if rescale = True -> shift mean(z) -> 0
    """
    x = x / x.max()
    y = y / y.max()
    z =  z / z.max()
    if rescale:
        z -= np.mean(z)
    return x,y,z

def matDesign (x , y, order, indVariables = 2):
        '''This is a function to set up the design matrix
        the inputs are :dataSet, the n datapoints, x and y data in a nx2 matrix
                        order, is the order of the coefficients,
                        indVariables, the number of independant variables
        the outputs are X
        '''

        # find the number of coefficients we will end up with
        num_coeff = int((order + 1)*(order + 2)/2)

        #find the number of rows in dataSet
        n = np.shape(x)[0]
        # create an empty matrix of zeros
        design = np.zeros((n,num_coeff))

        #fast assignment
        temp = design.T
        current_col = 0

        for p in range(order + 1):
            for i in range(p + 1):
                temp[current_col] = x**(p-i) * y**(i)
                current_col += 1

        return temp.T

def downsampler(data,target):
    '''
    Downsamples the data and target
    Note: I have written the code for downsampling target
    Need to downsample data as well
    '''
    i_C1 = np.where(target > 0)[0]; i_C0 = np.where(target < 1)[0]
    n_C1 = len(i_C1)
    #most people are zeros, need to downscale this
    # So for each in index in class 1 take random from class 0
    i_C0_dsample = np.random.choice(i_C0, size=n_C1, replace=False)
    # Join together with downsampled
    targetT = target.T[0]
    dtarget = np.hstack((targetT[i_C1], targetT[i_C0_dsample]))
    # return the target in right dimensions
    dtarget = dtarget.reshape(-1,1)

    # Downsampling data
    ddata = np.vstack((data[i_C1],data[i_C0_dsample]))
    return dtarget,ddata
