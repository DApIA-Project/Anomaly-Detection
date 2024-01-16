
import math


class SliceCluster:

    def __init__(self, slices = 10, levels = 1) -> None:
        self.slices = slices
        self.levels = levels

        self.radius = []

    def fit(self, X, y=None):
        ds = []
        for i in range(len(X)):
            d = math.sqrt(X[i][0] ** 2 + X[i][1] ** 2)
            ds.append(d)

        ds = sorted(ds)
        self.radius = []
        for i in range(self.levels):
            self.radius.append(ds[int(len(ds) * (i + 1) / self.levels)-1])

        return self
    
    def predict(self, X):
        res = []
        for i in range(len(X)):
            a = math.atan2(X[i][1], -X[i][0])
            if (a >= math.pi):
                a -= 2 * math.pi
            slice_id = int((a + math.pi) / (2 * math.pi) * self.slices)
            d = math.sqrt(X[i][0] ** 2 + X[i][1] ** 2)
            radius_id = 0
            for j in range(len(self.radius)):
                if (d < self.radius[j]):
                    radius_id = j
                    break

            if (d < 0.000002):
                id = 0
            else:
                id = slice_id * self.levels + radius_id + 1

            res.append(id)
        return res

    def getVariables(self):
        return self.slices, self.levels, self.radius
    
    def setVariables(self, v):
        self.slices, self.levels, self.radius = v
