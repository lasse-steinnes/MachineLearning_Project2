import autograd.numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from SGD import SGD
from OneHot import OneHot

class LogisticRegression (SGD, OneHot):
    """
    class that performs logistic regression with n classes to classify
    Methods:
    __init__(self, classes = 2, learning_rate = 0.01, adaptive_learning_rate = 'const', max_iter=5000, tol = 1e-7, logging = False)

    fit(X, y): performs fit on given data X shaped (sampels, features) to get targets y shaped (samples,)
               inferre parameter size beta shaped (features, classes)
               call to stochastic gradien decent, with learning rate gamma (optional decay; momentum, own function depending on #of calls to sgd)

    evaluate(X,y) tests the models performenc on predicting the given y shaped (tests,) when givven the input data X shaped (tests, features)

    predict(X) predict the outcomes of given data X shaped (predictions, features), returns predictions of the model shaped (predictions, classes)


    """
    def __init__(self, classes = 2, class_dict = None, learning_rate = 0.01, adaptive_learning_rate = 'const',

                 epochs = 10, mini_batch_size=10, max_iter=5000, tol = 1e-7, regularization = ('l2', 0.01),
                 momentum = True, m0 = 1e-2, logging = False):

        """
        classes: #predicted classes
        class_dict: provide a dictonary which encodes the on hot mapping {value: index}
                    if not provided it is inferred when calling fit (might give prblems when not all classes are in fit target y)
        adaptive_learning_rate: decides how to treat learning rate
                                if set to 'const' get cont. learning rate
                                if set to 'decay' use adaptive learning rate
                                provide own function which takes inital learning_rate and time_step t as argument
        max_iter, tol sets maximal #iterations and minimal change of weights beta in sgd
        regularization:  tuple (norm, lambda) supported norms are l1, l2
        logging: if True keep log of all updates
        """
        self.classes = classes
        self.reg = regularization

        OneHot.__init__(self, dictonary=class_dict)      
        SGD.__init__(self, LogisticRegression.__cross_entropy, epochs =epochs, mini_batch_size = mini_batch_size,

                     learning_rate = learning_rate, adaptive_learning_rate = adaptive_learning_rate, tolerance = tol,
                     momentum = momentum, m0 = m0)

        
        self.__fit_count = 0
        self.log = logging
        if logging:
            self. epochs = 1
            self.log_epochs = epochs
            self.logs = pd.DataFrame(columns=["Fit nr","data set", "epoch", "batch size","learning rate", "mse", "r2", "accuracy", "cross entropy"])
            self.__log_calls = 0

    def fit(self, X, y, split = False , fraction = 0.2, test = None):
        #split True splits training data again and keeps best parameter
        #test is optinonal with (X_test, y_test)
        #make sure no change on input data
        X = np.copy(X)
        y = np.copy(y)
        self.__fit_count += 1

        if split and (test == None):
            X, X_test, y, y_test = train_test_split(X, y, test_size = fraction)
            test = (X_test, y_test)
                
        #convert to one hot encoding 
        y_one_hot = OneHot.encoding(self, y)

        #setting up weights
        shape = X.shape
        self.weights = 10**(-6)*np.random.randn(shape[1]* self.classes).reshape((shape[1], self.classes))  # initialization
        best_weights = np.copy(self.weights)
        best_acc = 0
        
        #sgd
        if self.log or split:
            #training with logging or splitting
            #set up necessarities for epoch
            num_mini_batches = shape[0] // self.mini_batch_size
            self.gamma = self.learning_rate

            for self.current_epoch in range(0, self.log_epochs +1):
                #logging
                sc = LogisticRegression.__log_training(self, X, y, test)
                #training one epoch at a time
                SGD.run_epoch(self, X, y_one_hot, num_mini_batches)

                if split and sc > best_acc:
                    best_weights = np.copy(self.weights)
                    best_acc = sc
        
        else:
            #training without logging
            SGD.run_SGD(self, X, y_one_hot)

        #use best par
        if split:
            self.weights = best_weights


    def predict(self, X, decoded = False):
        z = X@self.weights
        #softmax function
        nom = np.vstack((np.sum( np.exp(z), axis= 1)) * self.classes)
        p = np.exp(z) / nom
        if decoded:
            return OneHot.decoding(self, p)
        return p

    def evaluate(self, X, y):
        prediction = LogisticRegression.predict(self, X)
        
        pred_class = OneHot.decoding(self, prediction)

        scores = {'mse' : LogisticRegression.__MSE(self,pred_class, y),
                  'r2': LogisticRegression.__R2(self, X, y),
                  'accuracy': LogisticRegression.__accuracy(self,pred_class, y),
                  'cross entropy' : LogisticRegression.__cross_entropy(self, self.weights, X, OneHot.encoding(self, y))}

        return scores

    def confusion_matrix(self, X, y):
        """
        returns the confusion matrix, i.e. number of true positives and flase negatives for all classes
        """
        prediction = LogisticRegression.predict(self, X)
        return OneHot.confusion(self, prediction, y)
    
    #Cross entropy function
    def __cross_entropy(self,W, X, y):
        z = X@W
        #softmax function
        nom = np.vstack((np.sum( np.exp(z), axis= 1)) * self.classes)
        prediction = np.exp(z) / nom
        ret = - np.sum(np.where(y ==1, np.log(prediction), 0))/len(y)
        if self.reg[0] == 'l1':
            ret -=  float(self.reg[1]) * np.sum(np.abs(W))
        if self.reg[0] == 'l2':
            ret -=  float(self.reg[1]) * np.sum(np.linalg.norm(W, axis = 1))
        return ret

    #MSE; R2; accuracy
    def __MSE(self, prediction, y):
        res = prediction -y
        return np.dot(res,res)/len(res)

    def __R2(self, X, y):
        #from likelihood
        W = np.zeros(self.weights.shape)
        W[0] = self.weights[0]
        z = X@W
        nom = np.vstack((np.sum( np.exp(z), axis= 1)) * self.classes)
        pred = np.exp(z)/nom
        L0 = np.sum(np.log(pred))
        LB = np.sum(np.log(LogisticRegression.predict(self, X)))
        return (L0-LB)/L0
        """
        res_den = prediction -y
        res_nom = y - np.mean(y)
        return 1 - np.dot(res_den,res_den) / np.dot(res_nom, res_nom)
        """

    def __accuracy(self, prediction, y):
        mask = prediction == y
        return len(prediction[mask])/len(prediction)
    
    def __log_entry(self, scores, learning_rate, batchsize, data_set):
        #log information
        temp = pd.DataFrame(dict({"Fit nr": self.__fit_count,
                            "data set": data_set,
                            "epoch":self.current_epoch,
                            "batch size": batchsize,
                            "learning rate":learning_rate ,
                            },**scores), index=[self.__log_calls])
        self.logs = self.logs.append(temp)
        self.__log_calls += 1
        del temp

    def __log_training(self, X, y, test = None):
        score = LogisticRegression.evaluate(self, X, y)
        LogisticRegression.__log_entry(self, score, self.gamma,self.mini_batch_size , "train")
        if test != None:
            score = LogisticRegression.evaluate(self, *test)
            LogisticRegression.__log_entry(self, score, self.gamma,self.mini_batch_size ,"test")
        sc =score["accuracy"]
        print("Epoch %i " %self.current_epoch, "accuracy: %.2f" %  sc)
        return sc
