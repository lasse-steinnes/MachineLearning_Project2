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