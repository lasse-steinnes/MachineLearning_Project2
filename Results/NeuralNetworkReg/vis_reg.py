"""
script to visulaize the toi for a nn with variation in hyperparameter
usage: python vis_reg.py filename str(parameter to vary)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
#font size controles
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
sns.set_context("paper", rc={"font.size":MEDIUM_SIZE,"axes.titlesize":MEDIUM_SIZE,"axes.labelsize":MEDIUM_SIZE, 'legend':MEDIUM_SIZE})

Y = sys.argv[2]

def draw_MSE(*args, **kwargs):

#    Draw pol order (x-axis) vs MSE (y-axis)
    data = kwargs.pop('data')
    agg_func = kwargs.pop('aggregate')
    data = g.data
    if agg_func =='max':
        data = data.groupby([args[0]])[args[1]].max().reset_index()
    elif agg_func =='min':
         data = data.groupby([args[0]])[args[1]].min().reset_index()
    sns.set(style='darkgrid')
    f =sns.lineplot(x = args[0], y = args[1], data = data, **kwargs)
    pol = data[args[0]];mse = data[args[1]]
    print('pol order, lowest mse')
    print(pol[22],np.min(mse))
    return f

df = pd.read_csv(sys.argv[1])

df["activ. func."] = df["nodes per layer"].str[-1]
df = df.rename(columns ={"nodes per layer":"topology", })
df = df.replace({"activ. func.":{'d':'sigmoid', 'h':'tanh'},
                "topology": {r'([a-z])|_':''}},regex = True)

#cost function
g = sns.FacetGrid(df[ df["data set"] == 'validation'], col="batch size", row ='activ. func.', margin_titles=True)
g.map_dataframe(draw_MSE, 'pol order','cost', aggregate='min')
g.savefig(fname ='./nn_reg_final.pdf', dpi='figure', format= 'pdf')
plt.show()
