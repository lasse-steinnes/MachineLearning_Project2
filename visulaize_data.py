import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as  np 

file_name = "/home/lukas/Documents/MachinLearnin_Project2/default of credit card clients.xls"


def offdiag(x, y, xcol=True, ycol=True, COUNT = 0, **kwargs):

    if( xcol[COUNT] and ycol[COUNT]):
        sns.kdeplot(x,shade=True,**kwargs)
        sns.kdeplot(y, vertical=True, **kwargs)
    else:
        plt.scatter(x,y, **kwargs)
    COUNT +=1

def gridplot(df, xvar, yvar,filename ="DataGrid", hue="default payment next month"):
    print("Creat figure %s" % filename)
    fig =plt.figure(figsize=(10,10))
    g = sns.PairGrid(df, hue = hue, x_vars=xvar,y_vars=yvar )
    #list of encoded discrete values
    int_cat = np.array(["SEX",	"EDUCATION",	"MARRIAGE",	"PAY_0",	"PAY_2",	"PAY_3",	"PAY_4",	"PAY_5",	"PAY_6"])
    
    #find if in given name sequence some are discrete values
    xcol = np.in1d(xvar,int_cat)
    ycol = np.in1d(yvar,int_cat)
    xcol, ycol = np.meshgrid(xcol, ycol)
    xcol = xcol.flatten()
    ycol = ycol.flatten()

    COUNT = 0
    #account for blog which have no diagonal of whole grid
    try:
        g = g.map_diag(plt.hist, alpha = 0.6)
        g = g.map_offdiag(offdiag, xcol =xcol, ycol = ycol, COUNT = COUNT)
    except:
        g = g.map(offdiag, xcol =xcol, ycol = ycol, COUNT = COUNT)    

    g.add_legend()
    g.savefig(filename +".pdf")
    #free memory
    del fig, g

def visaulize( file, sheet, max_grid = 3, corr_threshold = 0.1, show_corr=False, data_fraction = 0.25):
    df = pd.read_excel(file, sheet_name=sheet,header = 1, index_col = 0, skiprows=0)
    corr = df.corr()
    
    #select only high correlation numbers for plotting:
    mask = corr[df.columns[-1]].abs() > corr_threshold 
    names = corr[df.columns[-1]][mask].index
    df = df[names]
    #reduce data
    df = df.sample(frac = data_fraction)

    if show_corr:        
        fig = plt.figure(figsize=(10,10))
        g = sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='seismic')
        plt.show()
        del fig, g

    #begin plotting
    col_name = df.columns[:-1]
    nrcols = len(col_name)
    #split large files into multiple subplots
    if nrcols <= max_grid:
        gridplot(df, xvar=col_name, yvar=col_name)
    else:
        low_ind = np.arange(0, nrcols, max_grid, dtype=np.int)
        for l1 in low_ind:
            #only comput lower block diagonal plots/ upper diagonal is transpsed
            for l2 in low_ind[low_ind <= l1]:
                u1 = l1 + max_grid
                u2 = l2 + max_grid
                if u1 >= nrcols:
                    u1 = None
                if u2 >= nrcols:
                    u2 = None                    
                gridplot(df, xvar=col_name[l1 : u1], yvar=col_name[l2 : u2], filename ="DataGrid_slice_%i_%i" % (l1,l2))
    
visaulize(file_name,'Data', max_grid=5, corr_threshold=0.05)