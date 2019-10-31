"""
script to viwsulaize the toi for a nn with variation in hyperparameter
usage: python vis_nn.py filename str(parameter to vary)
"""
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
def draw_heatmap(*args, **kwargs):
    """
    heatmap function for FacetGrid
    """
    data = kwargs.pop('data')
    agg_func = kwargs.pop('aggregate')
    if agg_func =='max':
        data = data.groupby([args[0], args[1]], sort=False)[args[2]].max().reset_index()
    elif agg_func =='min':
         data = data.groupby([args[0], args[1]], sort=False)[args[2]].min().reset_index()
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    f =sns.heatmap(d, annot = True,cbar = False, fmt ='.2f', annot_kws={'size':SMALL_SIZE}, **kwargs)
    return f

df = pd.read_csv(sys.argv[1])
df["activ. func."] = df["nodes per layer"].str[-1]
df = df.rename(columns ={"nodes per layer":"topology", })
df = df.replace({"activ. func.":{'d':'sigmoid', 'h':'tanh'}, 
                "topology": {r'([a-z])|_':''}},regex = True)

#accuracy plot
g = sns.FacetGrid(df[ df["data set"] == 'test'], col="topology", row ='activ. func.', margin_titles=True)
g.map_dataframe(draw_heatmap, 'batch size',Y, 'accuracy', vmin = df["accuracy"].min(), vmax = df["accuracy"].max(), aggregate = 'max')
plt.show()
#cost function
g = sns.FacetGrid(df[ df["data set"] == 'test'], col="topology", row ='activ. func.', margin_titles=True)
g.map_dataframe(draw_heatmap, 'batch size',Y, 'cost', vmin = df["cost"].min(), vmax = df["cost"].max(), aggregate='min')
plt.show()

try:
    #accuracy plot
    g = sns.FacetGrid(df[ df["data set"] == 'validation'], col="topology", row ='activ. func.', margin_titles=True)
    g.map_dataframe(draw_heatmap, 'batch size',Y, 'accuracy', vmin = df["accuracy"].min(), vmax = df["accuracy"].max(), aggregate = 'max')
    plt.show()
except:
    pass
