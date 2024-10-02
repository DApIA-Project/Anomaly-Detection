from numpy_typing import np, ax, ax
from typing_extensions import Self

IRREGULAR_DIMENSION = "The last dimension is not always the same size"

class MinMaxScaler3D:

    def __init__(self, min:float=0, max:float=1) -> None:
        self.mins = []
        self.maxs = []
        self.min = min
        self.max = max
        self.__is_fitted__ = False

    def fit(self, X:np.float64_3d) -> "Self":
        self.__is_fitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for b in range(len(X)):
            for t in range(len(X[b])):
                if (size == -1):
                    size = len(X[b][t])
                if (size != len(X[b][t])):
                    raise ValueError(IRREGULAR_DIMENSION)


        self.mins = np.full(size, np.inf)
        self.maxs = np.full(size, -np.inf)

        self.mins = np.nanmin(X, axis=(0,1))
        self.maxs = np.nanmax(X, axis=(0,1))

        return self

    def transform(self, X:np.float64_3d) -> np.float64_3d:

        X = X.copy()

        for b in range(len(X)):
            for f in range(len(X[b][0])):
                if (self.maxs[f] == self.mins[f]):
                    X[b][:, f] = self.min
                else:
                    X[b][:, f] = (X[b][:, f] - self.mins[f]) / (self.maxs[f] - self.mins[f])
                    X[b][:, f] = X[b][:, f] * (self.max - self.min) + self.min



        return X

    def fit_transform(self, X:np.float64_3d) -> np.float64_3d:
        return self.fit(X).transform(X)

    def inverse_transform(self, X:np.float64_3d) -> np.float64_3d:

        X = X.copy()

        # check if is numpy array
        for b in range(len(X)):
            for f in range(len(X[b][0])):
                X[b][:, f] = (X[b][:, f] - self.min) / (self.max - self.min)
                X[b][:, f] = X[b][:, f] * (self.maxs[f] - self.mins[f]) + self.mins[f]

        return X

    def is_fitted(self) -> bool:
        return self.__is_fitted__


    def get_variables(self) -> "tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature], float, float]":
        return [self.mins, self.maxs, self.min, self.max]

    def set_variables(self, variables:"tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature], float, float]")\
            -> "Self":

        self.mins = variables[0]
        self.maxs = variables[1]
        self.min = variables[2]
        self.max = variables[3]
        self.__is_fitted__ = True
        return self



class MinMaxScaler2D:

    def __init__(self, min:float=0, max:float=1) -> None:
        self.mins = []
        self.maxs = []
        self.min = min
        self.max = max
        self.max_min = self.max - self.min
        self.__is_fitted__ = False

    def fit(self, X:np.float64_2d) -> "Self":
        self.__is_fitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for t in range(len(X)):
            if (size == -1):
                size = len(X[t])
            if (size != len(X[t])):
                raise ValueError(IRREGULAR_DIMENSION)


        self.mins = np.full(size, np.inf)
        self.maxs = np.full(size, -np.inf)

        self.mins = np.nanmin(X, axis=0)
        self.maxs = np.nanmax(X, axis=0)

        for f in range(len(X[0])):
            if (self.maxs[f] == self.mins[f]):
                self.maxs[f] = self.mins[f] + 0.0001
        self.maxs_mins = self.maxs - self.mins

        return self

    def transform(self, X:np.float64_2d) -> np.float64_2d:
        return ((X - self.mins) / self.maxs_mins) * self.max_min + self.min

    def fit_transform(self, X:np.float64_2d) -> np.float64_2d:
        return self.fit(X).transform(X)

    def inverse_transform(self, X:np.float64_2d) -> np.float64_2d:
        return ((X - self.min) / self.max_min) * self.maxs_mins + self.mins

    def is_fitted(self) -> bool:
        return self.__is_fitted__

    def get_variables(self) -> "tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature], float, float]":
        return [self.mins, self.maxs, self.min, self.max]

    def set_variables(self, variables:"tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature], float, float]")\
            -> "Self":
        self.mins = variables[0]
        self.maxs = variables[1]
        self.min = variables[2]
        self.max = variables[3]
        self.maxs_mins = self.maxs - self.mins
        self.max_min = self.max - self.min
        self.__is_fitted__ = True
        return self



class StandardScaler3D:

    def __init__(self) -> None:
        self.means = []
        self.stds = []
        self.__is_fitted__ = False

    def fit(self, X:np.float64_3d) -> "Self":
        self.__is_fitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for b in range(len(X)):
            for t in range(len(X[b])):
                if (size == -1):
                    size = len(X[b][t])
                if (size != len(X[b][t])):
                    raise ValueError(IRREGULAR_DIMENSION)


        self.means = np.full(size, np.inf)
        self.stds = np.full(size, -np.inf)

        self.means = np.nanmean(X, axis=(0,1))
        self.stds = np.nanstd(X, axis=(0,1))

        return self

    def transform(self, X:np.float64_3d) -> np.float64_3d:

            X = X.copy()

            for b in range(len(X)):
                for f in range(len(X[b][0])):
                    if (self.stds[f] == 0):
                        X[b][:, f] = 0
                    else:
                        X[b][:, f] = (X[b][:, f] - self.means[f]) / self.stds[f]

            return X

    def fit_transform(self, X:np.float64_3d) -> np.float64_3d:
        return self.fit(X).transform(X)

    def inverse_transform(self, X:np.float64_3d) -> np.float64_3d:

            X = X.copy()

            # check if is numpy array
            for b in range(len(X)):
                for f in range(len(X[b][0])):
                    X[b][:, f] = X[b][:, f] * self.stds[f] + self.means[f]

            return X

    def is_fitted(self) -> bool:
        return self.__is_fitted__

    def get_variables(self) -> "tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]":
        return np.array([self.means, self.stds])

    def set_variables(self, variables:"tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]") -> "Self":
        self.means = variables[0]
        self.stds = variables[1]
        self.__is_fitted__ = True
        return self

class StandardScaler2D:

    def __init__(self) -> None:
        self.means = []
        self.stds = []
        self.__is_fitted__ = False

    def fit(self, X:np.float64_2d) -> "Self":
        self.__is_fitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for t in range(len(X)):
            if (size == -1):
                size = len(X[t])
            if (size != len(X[t])):
                raise ValueError(IRREGULAR_DIMENSION)


        self.means = np.full(size, np.inf)
        self.stds = np.full(size, -np.inf)

        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)

        return self

    def transform(self, X:np.float64_2d) -> np.float64_2d:

            X = X.copy()

            for f in range(len(X[0])):
                if (self.stds[f] == 0):
                    X[:, f] = 0
                else:
                    X[:, f] = (X[:, f] - self.means[f]) / self.stds[f]

            return X

    def fit_transform(self, X:np.float64_2d) -> np.float64_2d:
        return self.fit(X).transform(X)

    def inverse_transform(self, X:np.float64_2d) -> np.float64_2d:

            X = X.copy()

            # check if is numpy array
            for f in range(len(X[0])):
                X[:, f] = X[:, f] * self.stds[f] + self.means[f]

            return X

    def is_fitted(self) -> bool:
        return self.__is_fitted__

    def get_variables(self) -> "tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]":
        return np.array([self.means, self.stds])

    def set_variables(self, variables:"tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]") -> "Self":
        self.means = variables[0]
        self.stds = variables[1]
        self.__is_fitted__ = True
        return self

def fill_nan_3d(x:"np.float64_3d", values:"np.float64_1d[ax.feature]") -> "np.float64_3d":
    """
    Fill NaN values in a 3D array with the given values
    """
    # remplace nan by min values
    for i in range(len(x)):
        for f in range(len(values)):
            x[i][:, f] = np.nan_to_num(x[i][:, f], nan=values[f])
    return x

def fill_nan_2d(x:"np.float64_2d", values:"np.float64_1d[ax.feature]") -> "np.float64_2d":
    """
    Fill NaN values in a 2D array with the given values
    """
    # remplace nan by min values
    for f in range(len(values)):
        x[:, f] = np.nan_to_num(x[:, f], nan=values[f])
    return x

import math

def __sigmoid__(x:float) -> float:
    return 1.0 / (1.0 + math.exp(float(-x)))

def __sigmoid_inverse__(x:float) -> float:
    if (np.isnan(x)):
        return x

    if (x <= 0):
        return -10e9
    elif (x >= 1):
        return 10e9

    return math.log(x / (1.0 - x))

np_sig_vec = np.vectorize(__sigmoid__)
np_sig_inv_vec = np.vectorize(__sigmoid_inverse__)

def sigmoid(x:np.ndarray) -> np.ndarray:
    return np_sig_vec(x)
def sigmoid_inverse(x:np.ndarray) -> np.ndarray:
    return np_sig_inv_vec(x)

class SigmoidScaler2D():
    """
    same as standard scaler but is sigmoided
    to avoid outliers
    """

    def __init__(self) -> None:
        self.means = []
        self.stds = []
        self.__is_fitted__ = False

    def fit(self, X:np.float64_2d) -> "Self":
        self.__is_fitted__ = True

        # check if 3dr channel is allways the same size
        size = -1
        for t in range(len(X)):
            if (size == -1):
                size = len(X[t])
            if (size != len(X[t])):
                raise ValueError(IRREGULAR_DIMENSION)


        self.means = np.full(size, np.inf)
        self.stds = np.full(size, -np.inf)

        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)

        return self

    def transform(self, X:np.float64_2d) -> np.float64_2d:

            X = X.copy()

            for f in range(len(X[0])):
                if (self.stds[f] == 0):
                    X[:, f] = 0
                else:
                    X[:, f] = (X[:, f] - self.means[f]) / self.stds[f]
                    X[:, f] = sigmoid(X[:, f])

            return X

    def fit_transform(self, X:np.float64_2d) -> np.float64_2d:
        return self.fit(X).transform(X)

    def inverse_transform(self, X:np.float64_2d) -> np.float64_2d:
            X=np.array(X)

            X = X.copy()

            # check if is numpy array
            for f in range(len(X[0])):
                X[:, f] = sigmoid_inverse(X[:, f])
                X[:, f] = X[:, f] * self.stds[f] + self.means[f]

            return X

    def is_fitted(self) -> bool:
        return self.__is_fitted__

    def get_variables(self) -> "tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]":
        return np.array([self.means, self.stds])

    def set_variables(self, variables:"tuple[np.float64_1d[ax.feature], np.float64_1d[ax.feature]]") -> "Self":
        self.means = variables[0]
        self.stds = variables[1]
        self.__is_fitted__ = True
        return self

