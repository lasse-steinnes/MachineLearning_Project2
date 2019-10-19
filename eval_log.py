import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier

from helper_functions import parse_data
import LogisticRegression as Log

def grid_search(df):
    valid_size = 0.1
    lamb = [('l1',1e-9), ('l1',1e-6), ('l1',1e-3),
            ('l2',1e-9), ('l2',1e-6), ('l2',1e-3),
            ('none',0)]
    learning =[1 , 0.5, 0.1, 0.01]
    adaptiv = ['const', 'decay']
    batchsize = [10, 40, 60, 80]
    epochs = [1, 10, 40, 60]
    mi = 10**4
    X, y = parse_data(df, "default payment next month", unbalanced= False )
    
    stats = pd.DataFrame( np.zeros((2 * 2* 4 ** 3 * 7, 8)),
                    columns = ["balanced", "learning rate", "adaptive learning", "epochs", "batch size", "regularization" ,"regularization parameter", "accuracy" ])
    index = 0
    for balance in [False, True]:
        X, y = parse_data(df, "default payment next month", unbalanced= balance )
        N = len(y)
        print("N = "+str(N)+"; 0: %.2f; 1: %.2f" % tuple(np.bincount(y)/N))
        X_trian, X_eval, y_train, y_eval = train_test_split(X,y, test_size =valid_size)
        for l in lamb:
            for gamma in learning:
                for a in adaptiv:
                    for bs in batchsize:
                        for ep in epochs: 
                            clf_own = Log.LogisticRegression(max_iter = mi, mini_batch_size=bs, epochs = ep, learning_rate=gamma, adaptive_learning_rate=a,
                                                            regualrization= l, logging = True)
                            clf_own.fit(X_trian,y_train, split = True, fraction = valid_size/(1-valid_size))
                            d =clf_own.evaluate(X_eval,y_eval, data_set ="evaluate")
                            stats.iloc[index] = [balance, gamma, a, ep, bs, l[0], l[1], d["accuracy"]]
                            index += 1
    stats.to_csv("Results/LogReg/hyper_par.csv")

def compare(df, gamma, adapt_gamma, mini_batch_size, epochs , regularization, mi = 10**4, balanced = False, name="best", k_fold = 10):
    X, y = parse_data(df, "default payment next month", unbalanced= balanced )

    clf_own = Log.LogisticRegression(max_iter = mi, mini_batch_size=mini_batch_size, epochs = epochs, learning_rate=gamma, adaptive_learning_rate=adapt_gamma,
                                                            regualrization= regularization, logging = True)
    clf_ex  = LogisticRegression(solver ='newton-cg')
    clf_sgd = SGDClassifier(loss = 'log')

    own = 0
    ex = 0
    sgd = 0
    confusion = pd.DataFrame( np.zeros((3,6)),
        columns =["0", "1", "precision", "recall", "specificity", "accuracy"],
        index = ["0", "1","occurence"])
    confusion.index.name = 'predicted class'
    confusion.columns.name = 'actual class'

    for k in range(0,k_fold):
        X_trian, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
        clf_own.fit(X_trian,y_train, split = False)
        own  += clf_own.evaluate(X_test,y_test)["accuracy"] 
        confusion += clf_own.confusion_matrix(X_test,y_test)   
    
        clf_ex.fit(X_trian, y_train)
        ex  += clf_ex.score(X_test, y_test)      
        
        clf_sgd.fit(X_trian, y_train)
        sgd += clf_sgd.score(X_test, y_test)
    
    log = clf_own.logs
    log.to_csv("Results/LogReg/logs_" +name+".csv")

    del clf_own , clf_ex, clf_sgd
    confusion /= k_fold

    f = open("Results/LogReg/confusion_" + name +".tex", 'w+')
    f.write(
        "\caption{\\textbf{Confusion Matrix}: The confusion matrix for an "
    )
    if balanced:
        f.write("balanced ")
    else:
        f.write("unablanced ")
    f.write(
        "input data set. The learning rate is %s with %.2f. The model is trained in %i epochs with a batch size of %i." 
        % (adapt_gamma, gamma, epochs, mini_batch_size)
        )
    if regularization[0] != 'none':
        f.write(
            "%s is used as regularization scheme with a strength of %e." % regularization
        )
    f.write("The overall accuracy of the own Implementation (SGD) is %.3f, of the exact \\texttt{scikit} Logisitc regression is %.3f "
            %(own/k_fold, ex/k_fold)+
            "and of the SGD \\text{scikit} implementation is %.3f.}" % (sgd/k_fold)
            )
    f.write("\n \n")
    confusion.to_latex(buf = f)
    f.write("\n \n")
    f.close()

def main():
    filename = "default of credit card clients.xls"
    df = pd.read_excel(filename, header=1)

    #comment out for hyperparameter search
    #WARNING: takes very long!
    #grid_search(df)

    res = pd. read_csv("Results/LogReg/hyper_par.csv")
    for balance in [False, True]:
        data = res[res["balanced"] == balance]
        best = data[data["accuracy"] == data["accuracy"].max()]
        gamma = best["learning rate"].mean()
        adapt_gamma = best["adaptive learning"].iloc[0]
        mini_batch_size = int(best["batch size"].mean())
        epochs = int(best["epochs"].mean())
        regularization = ( best["regularization"].iloc[0], best["regularization parameter"].mean())
        compare(df, gamma, adapt_gamma, mini_batch_size, epochs , regularization, balanced = balance, name="best_"+str(balance))

if __name__ == '__main__':
    main()








