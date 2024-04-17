from sklearn.preprocessing import LabelBinarizer
import numpy as np

class SparceLabelBinarizer():

    def __init__(self, variables=None):
        if (variables is not None):
            self.setVariables(variables)

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.inv = {v:i for i, v in enumerate(self.classes_)}

    def transform(self, y):
        arr = np.zeros((len(y), len(self.classes_)))
        for i, v in enumerate(y):
            arr[i, self.inv[v]] = 1
        return arr

    def inverse_transform(self, Y):
        return np.array([self.classes_[np.argmax(y)] for y in Y])
        

    def getVariables(self):
        return self.classes_
    
    def setVariables(self, variables):
        self.fit(variables)
        
