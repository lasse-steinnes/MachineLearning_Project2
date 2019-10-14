import numpy as np

class LogisticRegression :
    """
    class that performs logistic regression with n classes to classify
    Methods:
    __init__(self, classes = 2, learning_rate = 0.01, adaptive_learning_rate = 'const', max_iter=5000, tol = 1e-7, logging = False)

    fit(X, y): performs fit on given data X shaped (sampels, features) to get targets y shaped (samples,classes) (y is hot_one encoded)
               inferre parameter size beta shaped (features, classes) 
               call to stochastic gradien decent, with learning rate gamma (optional decay; momentum, own function depending on #of calls to sgd)

    evaluate(X,y) tests the models performenc on predicting the given y shaped (tests, classes) when givven the input data X shaped (tests, features)

    predict(X) predict the outcomes of given data X shaped (predictions, features), returns predictions of the model shaped (predictions, classes)


    """
    def __init__(self, classes = 2, learning_rate = 0.01, adaptive_learning_rate = 'const', max_iter=5000, tol = 1e-7, logging = False):
        """
        classes: #of predicted classes
        adaptive_learning_rate: decides how to treat learning rate
                                if set to 'const' get cont. learning rate
                                if set to 'decay', 'momentum' use respectivly schemes for adaptive learning rate
                                provide own function which takes inital learning_rate and time_step t as argument
        max_iter, tol sets maximal #iterations and minimal change of weights beta in sgd
        logging: if True keep log of all updates
        """
        self.classes = classes
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = learning_rate        
        try:
            self.learning_rate_adaption = {'const': False, 'decay': LogisticRegression.__decay, 'momentum': LogisticRegression.__momentum}
        except:
            self.learning_rate_adaption = adaptive_learning_rate
        self.__fit_count = 0

    def fit(self, X, y):
        self.__fit_count += 1
        self.beta = np.zeros((X.shape()[1], self.classes)) # initialization?
        #sgd
    
    def predict(self, X):
        z = X@self.beta
        #softmax function
        nom = 1 + np.sum( np.exp(z))
        return np.exp(z) / nom

    def evaluate(self, X, y):
        prediction = LogisticRegression.predict(self, X)
        #find predicted class
        prediction = np.where(prediction == prediction.max(axis=1),1,0)
        #how to deal with dim 1?
        return LogisticRegression.__MSE(prediction, y), LogisticRegression.__R2(prediction, y)

    #functions for adaptive learning rate
    def __decay(self,gamma0, t):
        pass
    def __momentum(self,gamma0, t):
        pass
    
    #Cross entropy function
    def __cross_entropy(self, prediction, y):
        N = np.shape(y)[0]
        return - np.sum(y @ np.log(prediction.T))


    #MSE; R2 with multiclass
    def __MSE(self, prediction, y):
        res = prediction -y
        res = res.sum(axis=1)
        return res.T@res/len(res)
    def __R2(self, prediction, y):
        res_den = prediction -y
        res_den = res_den.sum(axis=1)
        res_nom = y - np.mean(y)
        res_nom = res_nom.sum(axis=1)
        return 1 - res_den.T@res_den / res_nom.T@res_nom