import numpy as np
import pandas as pd
class OneHot:
    """
    class that provides one hot encoding for classification problems
    """
    def __init__(self, dictonary = None):
        if dictonary != None:
            self.one_hot_encoding_key = dictonary
            self.one_hot_decoding_key = {index: value for index, value in dictonary.items()}
            self.provided_dict =True
        else:
            self.provided_dict =False
    
    def encoding(self, y):
        """
        computes the one hot encding for the vector y
        returns y in shape (samples, #unique instances)
        """
        uni = np.unique(y)
        l_uni = len(uni)       
        l_y = len(y)
        hot = np.zeros((l_y, l_uni))
        #inferr dict only at first call otherwise it is provided from class
        if not self.provided_dict:
            self.one_hot_encoding_key = {uni[i]: i for i in range(l_uni)}
            self.one_hot_decoding_key = {i:uni[i] for i in range(l_uni)}
            self.provided_dict = True
        #actuall encoding
        for i in range(l_y):
            index  = self.one_hot_encoding_key[y[i]]
            hot[i, index] = 1
        return hot 

    def decoding(self, y):
        """
        decode one hot encoding of prediction
        """
        l_y = len(y)
        pred_class = np.zeros(l_y)
        for i in range(l_y):
            pred_class[i] = self.one_hot_decoding_key[np.argmax(y[i])] 
        return pred_class

    def confusion(self, prediction, target):
        """
        returns the confusion matrix, i.e. number of true positives and flase negatives for all classes
        """
        prediction = OneHot.decoding(self, prediction)
        
        list_of_classes = np.array(list(self.one_hot_encoding_key))
        list_of_classes = list_of_classes[:,np.newaxis]

        classes = len(self.one_hot_decoding_key)
        matrix = np.zeros( (classes, classes))

        for i, value in enumerate(self.one_hot_encoding_key):
            mask = prediction == value
            true_class = target[mask]
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
        input_df = np.zeros( (classes +1, classes +4))
        input_df[:-1,:classes] = matrix
        input_df[-1, :classes] = np.sum(matrix,axis = 0)/norm

        for i, val in enumerate(metrics):
            input_df[:-1,-4 + i] = val

        confusion = pd.DataFrame(input_df,
                                 columns = np.append([ str(self.one_hot_decoding_key[i]) for i in range(classes)], ["precision", "recall", "specificity", "accuracy"]),
                                 index = np.append([self.one_hot_decoding_key[i] for i in range(classes)],'occurence'))
        confusion.index.name = 'predicted class'
        confusion.columns.name = 'actual class'
        return confusion