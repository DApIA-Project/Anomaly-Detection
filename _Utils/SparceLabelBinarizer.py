from sklearn.preprocessing import LabelBinarizer
import numpy as np

class SparceLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
        

    def getVariables(self):
        return self.classes_
    
    def setVariables(self, variables):
        self.fit(variables)
        
