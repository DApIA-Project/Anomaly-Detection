import numpy as np

class MinMaxScaler3D:

    def __init__(self, min=0, max=1):
        self.mins = []
        self.maxs = []
        self.min = min
        self.max = max

    def fit(self, X):

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
            for t in range(len(X[b])):
                for f in range(len(X[b][t])):
                    if (X[b][t][f] < self.mins[f]):
                        self.mins[f] = X[b][t][f]
                    if (X[b][t][f] > self.maxs[f]):
                        self.maxs[f] = X[b][t][f]

        return self

    def transform(self, X):
        
        X = X.copy()

        for b in range(len(X)):
            for t in range(len(X[b])):
                for f in range(len(X[b][t])):
                    if (self.maxs[f] == self.mins[f]):
                        X[b][t][f] = self.min
                    else:
                        X[b][t][f] = (X[b][t][f] - self.mins[f]) / (self.maxs[f] - self.mins[f])
                        X[b][t][f] = X[b][t][f] * (self.max - self.min) + self.min

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):

        X = X.copy()

        for b in range(len(X)):
            for t in range(len(X[b])):
                for f in range(len(X[b][t])):
                    if (self.maxs[f] == self.mins[f]):
                        X[b][t][f] = self.mins[f]
                    else:
                        X[b][t][f] = (X[b][t][f] - self.min) / (self.max - self.min)
                        X[b][t][f] = X[b][t][f] * (self.maxs[f] - self.mins[f]) + self.mins[f]
    
        return X