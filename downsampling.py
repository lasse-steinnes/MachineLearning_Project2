######
import Neural_Network
import numpy as np
import pandas as pd
import OneHot
from helper_functions import load_terrain, normalize,matDesign
from sklearn.preprocessing import MinMaxScaler

filename = "default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

target = (df[['default payment next month']].copy()).to_numpy()
data = (df.drop(columns =['default payment next month', 'ID']).copy()).to_numpy()
print(target)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)
i_C1 = np.where(target > 0)[0]; i_C0 = np.where(target < 1)[0]
n_C1 = len(i_C1)
#most people are zeros, need to downscale this
# So for each in index in class 1 take random from class 0
i_C0_dsample = np.random.choice(i_C0, size=n_C1, replace=False)
# Join together with downsampled
targetT = target.T[0]
#print(targetT)
dtarget = np.hstack((targetT[i_C1], targetT[i_C0_dsample]))
# return the target in right dimensions
dtarget = dtarget.reshape(-1,1)
print(dtarget)
