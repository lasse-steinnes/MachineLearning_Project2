"""
script that visualizes the .csv files in ./Results/LogReg
"""
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
#font size controles
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
sns.set_context("paper", rc={"font.size":MEDIUM_SIZE,"axes.titlesize":MEDIUM_SIZE,"axes.labelsize":MEDIUM_SIZE, 'legend':MEDIUM_SIZE})

def draw_heatmap(*args, **kwargs):
    """
    heatmap function for FacetGrid
    """
    data = kwargs.pop('data')
    data = data.groupby(by=[args[0], args[1]]).mean().reset_index()
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    f =sns.heatmap(d, annot = True,cbar = False, fmt ='.2f', annot_kws={'size':SMALL_SIZE}, **kwargs)
    return f

def heat_reg(df):
    """
    regularization dependecies
    """
    for bal in [True, False]:
        data = df[df["balanced"] == bal].drop(columns =["balanced"])
        for c in ['const', 'decay']:
            data_temp = data[data["adaptive learning"] == c].drop(columns = ["adaptive learning"])
            min_max = [ data_temp["accuracy"].min(), data_temp["accuracy"].max()]
            for X in ["learning rate", "epochs"]:
                #no regularization
                no_reg  = data_temp[data_temp["regularization"] == 'none']
                g = draw_heatmap(X, 'batch size', 'accuracy', data =no_reg, vmin = min_max[0], vmax = min_max[1])
                plt.savefig("no_regularization_bal_" + str(bal) + "_learn_" + c + "_x_"+ X +".pdf")
                plt.close('all')
                #regularization
                temp = data_temp[data_temp["regularization"] != 'none']
                g = sns.FacetGrid(temp.rename(columns={'regularization parameter':'$\lambda$'}), row = "$\lambda$", col = "regularization", margin_titles=True)
                g.map_dataframe(draw_heatmap, X, "batch size", "accuracy", vmin = min_max[0], vmax = min_max[1])
                g.savefig("regularization_bal_" + str(bal) + "_learn_" + c + "_x_"+ X +".pdf")
                plt.close('all')
def heat_training(df):
    """
    compares, training inputs epochs and batch size
    """
    df = df.drop(columns =["regularization"])
    for bal in [True, False]:
        data = df[df["balanced"] == bal].drop(columns =["balanced"])
        min_max = [ data["accuracy"].min(), data["accuracy"].max()]
        g = sns.FacetGrid(data.rename(columns={'adaptive learning': 'adapt. learn.'}), row = "adapt. learn.", col = "learning rate", margin_titles=True)
        g.map_dataframe(draw_heatmap, "epochs", "batch size", "accuracy", vmin = min_max[0], vmax = min_max[1])
        g.savefig("regularization_bal_" + str(bal) + "_epochs_batchsize.pdf")

def training(df_name):
    df = pd.read_csv(df_name)
    g = sns.FacetGrid(data = df, row = "learning rate", col ='batch size', hue='data set', margin_titles= True)
    g.map(sns.lineplot , 'epoch', 'accuracy' )
    plt.savefig(df_name[:-4] +'.pdf')
    plt.close('all')
"""
df = pd.read_csv("hyper_par.csv")
heat_reg(df)
heat_training(df)
"""
filenames = ['logs_best_False5010.csv', 'logs_best_True5010.csv']
for f in filenames:
    training(f)

