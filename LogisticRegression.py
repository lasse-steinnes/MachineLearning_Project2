import numpy as np
import pandas as pd

class LogisticRegression :
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
    def __init__(self, classes = 2, class_dict = None, learning_rate = 0.01, adaptive_learning_rate = 'const', max_iter=5000, tol = 1e-7, logging = False):
        """
        classes: #predicted classes
        class_dict: provide a dictonary which encodes the on hot mapping {value: index}
                    if not provided it is inferred when calling fit (might give prblems when not all classes are in fit target y)
        adaptive_learning_rate: decides how to treat learning rate
                                if set to 'const' get cont. learning rate
                                if set to 'decay', 'momentum' use respectivly schemes for adaptive learning rate
                                provide own function which takes inital learning_rate and time_step t as argument
        max_iter, tol sets maximal #iterations and minimal change of weights beta in sgd
        logging: if True keep log of all updates
        """
        self.classes = classes
        if class_dict != None:
            self.one_hot_encoding = class_dict
            self.one_hot_decoding = {index: value for index, value in class_dict.item()}
            self.__provided_dict =True
        else:
            self.__provided_dict =False

        self.max_iter = max_iter
        self.tol = tol
        self.gamma = learning_rate        
        try:
            self.learning_rate_adaption = {'const': False, 'decay': LogisticRegression.__decay, 'momentum': LogisticRegression.__momentum}
        except:
            self.learning_rate_adaption = adaptive_learning_rate
        
        self.__fit_count = 0
        self.log = logging
        if logging:
            self.logs = pd.DataFrame(columns=["Fit nr", "data set", "mse", "r2", "accuracy"])

    def fit(self, X, y, batch_size = 10):
        self.__fit_count += 1
        #convert to one hot encoding 
        y_one_hot = LogisticRegression.__one_hot(self, y)
        self.beta = np.zeros((X.shape()[1], self.classes)) # initialization?
        #sgd

        #evaluate
        score = LogisticRegression.evaluate(self, X, y, data_set="train")
        return score

    def predict(self, X):
        z = X@self.beta
        #softmax function
        nom = 1 + np.sum( np.exp(z))
        return np.exp(z) / nom

    def evaluate(self, X, y, data_set= "test"):
        prediction = LogisticRegression.predict(self, X)
        l_y = len(y)
        pred_class = np.zeros(l_y)
        #decode one hot encoding of prediction
        for i in range(l_y):
            pred_class[i] = self.one_hot_decoding[np.argmax(prediction[i])]   

        scores = {'mse' : LogisticRegression.__MSE(self,pred_class, y),
                  'r2': LogisticRegression.__R2(self,pred_class, y),
                  'accuracy': LogisticRegression.__accuracy(self,pred_class, y)}
        
        if self.log:
            #log information
            temp = pd.DataFrame(scores.update({"Fit nr": self.__fit_count, "data set": data_set}))
            self.logs = self.logs.append(temp)
            del temp

        return scores

    #functions for adaptive learning rate
    def __decay(self,gamma0, t):
        pass
    def __momentum(self,gamma0, t):
        pass
    
    #Cross entropy function
    def __cross_entropy(self, prediction, y):
        return - np.sum(y @ np.log(prediction.T))

    def __one_hot(self, y):
        """
        computes the one hot encding for the vector y
        returns y in shape (samples, #unique instances)
        """
        uni = np.unique(y)
        l_uni = len(uni)
        if l_uni != self.classes:
            print("Not all classes in training data!")
        
        l_y = len(y)
        hot = np.zeros((l_y, l_uni))
        #inferr dict only at first call otherwise it is provided from class

        if (self.__fit_count == 1) and not self.__provided_dict:
            self.one_hot_encoding = {uni[i]: i for i in range(l_uni)}
            self.one_hot_decoding = {i:uni[i] for i in range(l_uni)}
            
        for i in range(l_y):
            index  = self.one_hot_encoding[y[i]]
            hot[i, index] = 1
        return hot 

    #MSE; R2 with multiclass
    def __MSE(self, prediction, y):
        res = prediction -y
        res = res.sum()
        return res.T@res/len(res)

    def __R2(self, prediction, y):
        res_den = prediction -y
        res_den = res_den.sum()
        res_nom = y - np.mean(y)
        res_nom = res_nom.sum()
        return 1 - res_den.T@res_den / res_nom.T@res_nom

    def __accuracy(self, prediction, y):
        mask = prediction == y
        return len(prediction[mask])/len(prediction)