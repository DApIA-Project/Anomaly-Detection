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
        
        self.mins = np.nanmin(X, axis=(0,1))
        self.maxs = np.nanmax(X, axis=(0,1))

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



class StandardScaler3D:

    def __init__(self):
        self.means = []
        self.stds = []
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


        self.means = np.full(size, np.inf)
        self.stds = np.full(size, -np.inf)
        
        self.means = np.nanmean(X, axis=(0,1))
        self.stds = np.nanstd(X, axis=(0,1))

        return self
    
    def transform(self, X):
            
            X = X.copy()
    
            for b in range(len(X)):
                for f in range(len(X[b][0])):
                    if (self.stds[f] == 0):
                        X[b][:, f] = 0
                    else:
                        X[b][:, f] = (X[b][:, f] - self.means[f]) / self.stds[f]
    
            return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        
            X = X.copy()
        
            # check if is numpy array
            for b in range(len(X)):
                for f in range(len(X[b][0])):
                    X[b][:, f] = X[b][:, f] * self.stds[f] + self.means[f]
        
            return X
    
    def isFitted(self):
        return self.__isFitted__
    
    def getVariables(self):
        return np.array([self.means, self.stds])
    
    def setVariables(self, variables):
        self.means = variables[0]
        self.stds = variables[1]
        self.__isFitted__ = True
        return self
    
def fillNaN3D(x:"list[np.array]", values):
    """
    Fill NaN values in a 3D array with the given values
    """
    # remplace nan by min values
    for i in range(len(x)):
        for f in range(len(values)):
            x[i][:, f] = np.nan_to_num(x[i][:, f], nan=values[f])

    return x