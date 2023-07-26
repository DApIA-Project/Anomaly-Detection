from typing import Any
import numpy as np

class MinMaxScaler3D:

    def __init__(self, min=0, max=1):
        self.mins = []
        self.maxs = []
        self.min = min
        self.max = max
        self.__isFitted__ = False

    def fit(self, X):
        self.__isFitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for b in range(len(X)):
            for t in range(len(X[b])):
                if (size == -1):
                    size = len(X[b][t])
                if (size != len(X[b][t])):
                    raise Exception("The 3rd dimension is not always the same size")


        self.mins = np.full(size, np.inf)
        self.maxs = np.full(size, -np.inf)

        for b in range(len(X)):
            # for f in range(len(X[b][0])):
            _min = np.min(X[b], axis=0)
            _max = np.max(X[b], axis=0)
            self.mins = np.min([self.mins, _min], axis=0)
            self.maxs = np.max([self.maxs, _max], axis=0)
        return self

    def transform(self, X):
        
        X = X.copy()

        for b in range(len(X)):
            for f in range(len(X[b][0])):
                if (self.maxs[f] == self.mins[f]):
                    X[b][:, f] = self.min
                else:
                    X[b][:, f] = (X[b][:, f] - self.mins[f]) / (self.maxs[f] - self.mins[f])
                    X[b][:, f] = X[b][:, f] * (self.max - self.min) + self.min



        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):

        X = X.copy()

        # check if is numpy array
        for b in range(len(X)):
            for f in range(len(X[b][0])):
                X[b][:, f] = (X[b][:, f] - self.min) / (self.max - self.min)
                X[b][:, f] = X[b][:, f] * (self.maxs[f] - self.mins[f]) + self.mins[f]
    
        return X
    
    def isFitted(self):
        return self.__isFitted__


    def getVariables(self):
        return np.array([self.mins, self.maxs, self.min, self.max])
    
    def setVariables(self, variables):
        self.mins = variables[0]
        self.maxs = variables[1]
        self.min = variables[2]
        self.max = variables[3]
        self.__isFitted__ = True
        return self

