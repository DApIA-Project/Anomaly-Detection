
from numpy_typing import np, ax, ax

class SparceLabelBinarizer():

    def __init__(self, variables:np.ndarray=None)->None:
        if (variables is not None):
            self.set_variables(variables)

    def fit(self, y:np.ndarray)->None:
        self.classes_ = np.unique(y)
        self.inv = {v:i for i, v in enumerate(self.classes_)}

    def transform(self, y:np.ndarray)->np.ndarray:
        arr = np.zeros((len(y), len(self.classes_)))
        for i, v in enumerate(y):
            l = self.inv.get(v, np.nan)
            if (l is not np.nan):
                arr[i, l] = 1
        return arr

    def inverse_transform(self, Y:np.ndarray)->np.ndarray:
        res = np.zeros(len(Y), dtype=object)
        for i, y in enumerate(Y):
            mi = np.argmax(y)
            if (y[mi] == 0):
                res[i] = np.nan
            else:
                res[i] = self.classes_[mi]
        return res


    def get_variables(self)->np.ndarray:
        return self.classes_

    def set_variables(self, variables:np.ndarray)->None:
        self.fit(variables)

