import autograd.numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from SGD import SGD
from helper_functions import OneHot

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
                 epochs = 10, mini_batch_size=10, max_iter=5000, tol = 1e-7, logging = False):
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
        OneHot.__init__(self, dictonary=class_dict)      
        SGD.__init__(self, LogisticRegression.__cross_entropy, epochs =epochs, mini_batch_size = mini_batch_size,
                     learning_rate = learning_rate, adaptive_learning_rate = adaptive_learning_rate, tolerance = tol, max_iter = max_iter)
        self.__fit_count = 0
        self.log = logging
        if logging:
            self. epochs = 1
            self.log_epochs = epochs
            self.logs = pd.DataFrame(columns=["Fit nr","epoch", "data set", "mse", "r2", "accuracy"])
            self.__log_calls = 0

    def fit(self, X, y, split = False , fraction = 0.2):
        self.__fit_count += 1
        if split:
            X, X_test, y, y_test = train_test_split(X, y, test_size = fraction)
        else:
            X_test, y_test = X, y
        #convert to one hot encoding 
        y_one_hot = OneHot.encoding(self, y)
        self.weights = 10**(-3)*np.random.rand(X.shape[1]* self.classes).reshape((X.shape[1], self.classes))  # initialization
        best_weights = np.copy(self.weights)
        best_acc = 0
        #sgd
        if self.log:
            for self.current_epoch in range(0, self.log_epochs):
                SGD.run_SGD(self, X, y_one_hot)
                score = LogisticRegression.evaluate(self, X, y, data_set="train")
                if split:
                    score = LogisticRegression.evaluate(self, X_test, y_test, data_set="test")
                sc =score["accuracy"]
                if sc > best_acc:
                    best_weights = np.copy(self.weights)
                print("Epoch %i " %self.current_epoch, "accuracy: %.2f" %  sc)
        else:
            SGD.run_SGD(self, X, y_one_hot)

        #evaluate
        if split:
            final_eval = "test"
            self.weights = best_weights
        else:
            final_eval ="train"
        score = LogisticRegression.evaluate(self, X_test, y_test, data_set=final_eval)
        return score

    def predict(self, X, decoded = False):
        z = X@self.weights
        #softmax function
        nom = np.sum( np.exp(z))
        p = np.exp(z) / nom
        if decoded:
            return OneHot.decoding(self, p)
        return p

    def evaluate(self, X, y, data_set= "test"):
        prediction = LogisticRegression.predict(self, X)
        
        pred_class = OneHot.decoding(self, prediction)

        scores = {'mse' : LogisticRegression.__MSE(self,pred_class, y),
                  'r2': LogisticRegression.__R2(self,pred_class, y),
                  'accuracy': LogisticRegression.__accuracy(self,pred_class, y)}
        if self.log:
            #log information
            temp = pd.DataFrame(dict({"Fit nr": self.__fit_count,"epoch":self.current_epoch + 1 , "data set": data_set},**scores), index=[self.__log_calls])
            self.logs = self.logs.append(temp)
            self.__log_calls += 1
            del temp

        return scores

    def confusion_matrix(self, X, y):
        """
        returns the confusion matrix, i.e. number of true positives and flase negatives for all classes
        """
        prediction = LogisticRegression.predict(self, X)
        prediction = OneHot.decoding(self, prediction)
        
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

        norm = np.sum(matrix)
        input_df = np.zeros( (self.classes +1, self.classes +4))
        input_df[:-1,:self.classes] = matrix
        input_df[-1, :self.classes] = np.sum(matrix,axis = 1)/norm

        for i, val in enumerate(metrics):
            input_df[:-1,-4 + i] = val

        confusion = pd.DataFrame(input_df,
                                 columns = np.append([ str(self.one_hot_decoding_key[i]) for i in range(self.classes)], ["precision", "recall", "specificity", "accuracy"]),
                                 index = np.append([self.one_hot_decoding_key[i] for i in range(self.classes)],'occurence'))
        confusion.index.name = 'predicted class'
        confusion.columns.name = 'actual class'
        return confusion

    
    #Cross entropy function
    def __cross_entropy(self,W, X, y):
        z = X@W
        #softmax function
        nom = np.sum( np.exp(z))
        prediction = np.exp(z) / nom
        return - np.sum(np.where(y ==1,np.log(prediction), 0))/len(y)

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