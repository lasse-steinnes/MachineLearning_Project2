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
                                if set to 'decay' use adaptive learning rate
                                provide own function which takes inital learning_rate and time_step t as argument
        max_iter, tol sets maximal #iterations and minimal change of weights beta in sgd
        logging: if True keep log of all updates
        """
        self.classes = classes
        if class_dict != None:
            self.one_hot_encoding_key = class_dict
            self.one_hot_decoding_key = {index: value for index, value in class_dict.item()}
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
            self.__log_calls = 0

    def fit(self, X, y, batch_size = 10):
        self.__fit_count += 1
        #convert to one hot encoding 
        y_one_hot = LogisticRegression.__one_hot_encoding(self, y)
        self.beta = np.zeros((X.shape[1], self.classes)) # initialization?
        self.beta = 10**(-6)*np.random.randn(X.shape[1]* self.classes).reshape((X.shape[1], self.classes))  # initialization?
        #sgd

        #evaluate
        score = LogisticRegression.evaluate(self, X, y, data_set="train")
        return score

    def predict(self, X, decoded = False):
        z = X@self.beta
        #softmax function
        nom = np.sum( np.exp(z))
        p = np.exp(z) / nom
        if decoded:
            return LogisticRegression.__one_hot_decoding(self, p)
        return p

    def evaluate(self, X, y, data_set= "test"):
        prediction = LogisticRegression.predict(self, X)
        
        pred_class = LogisticRegression.__one_hot_decoding(self, prediction)

        scores = {'mse' : LogisticRegression.__MSE(self,pred_class, y),
                  'r2': LogisticRegression.__R2(self,pred_class, y),
                  'accuracy': LogisticRegression.__accuracy(self,pred_class, y)}
        if self.log:
            #log information
            temp = pd.DataFrame(dict({"Fit nr": self.__fit_count, "data set": data_set},**scores), index=[self.__log_calls])
            self.logs = self.logs.append(temp)
            self.__log_calls += 1
            del temp

        return scores

    def confusion_matrix(self, X, y):
        """
        returns the confusion matrix, i.e. number of true positives and flase negatives for all classes
        """
        prediction = LogisticRegression.predict(self, X)
        prediction = LogisticRegression.__one_hot_decoding(self, prediction)
        
        list_of_classes = np.array(list(self.one_hot_encoding_key))
        list_of_classes = list_of_classes[:,np.newaxis]

        matrix = np.zeros( (self.classes, self.classes))
        for i, value in enumerate(self.one_hot_encoding_key):
            mask = prediction == value
            true_class = y[mask]
            matrix[i] = np.sum(true_class == list_of_classes, axis=1)

        tp = np.diag(matrix)
        fp = np.sum(matrix,axis=1) - tp
        fn = np.sum(matrix, axis=0) -tp
        tn = np.sum(matrix) - tp -fp -fn
        P = tp /(tp + fp)
        R = tp /(tp + fn)
        S = tn /(tn + fp)
        A = (tp +tn) / (tp + tn + fp +fn)
        
        metrics = [P,R,S,A]

        input_df = np.zeros( (self.classes, self.classes +4))
        input_df[:,:self.classes] = matrix
        for i, val in enumerate(metrics):
            input_df[:,-4 + i] = val

        confusion = pd.DataFrame(input_df,
                                 columns = np.append([ str(self.one_hot_decoding_key[i]) for i in range(self.classes)], ["precision", "recall", "specificity", "accuracy"]),
                                 index = [self.one_hot_decoding_key[i] for i in range(self.classes)])
        confusion.index.name = 'predicted class'
        confusion.columns.name = 'actual class'
        return confusion



    #functions for adaptive learning rate
    def __decay(self, gamma0, t):
        return gamma0 / ( gamma0*t +1)
    
    #Cross entropy function
    def __cross_entropy(self, prediction, y):
        return - np.sum(y @ np.log(prediction.T))/len(y)

    def __one_hot_encoding(self, y):
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
            self.one_hot_encoding_key = {uni[i]: i for i in range(l_uni)}
            self.one_hot_decoding_key = {i:uni[i] for i in range(l_uni)}
            
        for i in range(l_y):
            index  = self.one_hot_encoding_key[y[i]]
            hot[i, index] = 1
        return hot 

    def __one_hot_decoding(self, y):
        """
        decode one hot encoding of prediction
        """
        l_y = len(y)
        pred_class = np.zeros(l_y)
       
        for i in range(l_y):
            pred_class[i] = self.one_hot_decoding_key[np.argmax(y[i])] 
        return pred_class

    #MSE; R2; accuracy
    def __MSE(self, prediction, y):
        res = prediction -y
        return np.dot(res,res)/len(res)

    def __R2(self, prediction, y):
        res_den = prediction -y
        res_nom = y - np.mean(y)
        return 1 - np.dot(res_den,res_den) / np.dot(res_nom, res_nom)

    def __accuracy(self, prediction, y):
        mask = prediction == y
        return len(prediction[mask])/len(prediction)