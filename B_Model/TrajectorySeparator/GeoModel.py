
from B_Model.AbstractModel import Model as AbstactModel

import _Utils.FeatureGetter as FG
import _Utils.geographic_maths as GEO
import D_DataLoader.Utils as U

from _Utils.numpy import np, ax

def lerp(a:float, b:float, t:float) -> float:
    return a + (b-a)*t



def forecast(history:np.float64_2d[ax.time, ax.feature], t:int) -> "tuple[float, float]":
    if (len(history) == 0):
        return None
    if (len(history) == 1):
        return FG.lat(history[-1]), FG.lon(history[-1])

    a = -2
    b = -1
    while (-a < len(history) and
           FG.lat(history[a]) == FG.lat(history[b]) and
           FG.lon(history[a]) == FG.lon(history[b])):

        a -= 1

    lat_a = FG.lat(history[a])
    lon_a = FG.lon(history[a])
    t_a = FG.timestamp(history[a])

    lat_b = FG.lat(history[b])
    lon_b = FG.lon(history[b])
    t_b = FG.timestamp(history[b])

    bearing = GEO.bearing(lat_a, lon_a, lat_b, lon_b)
    distance = GEO.distance(lat_a, lon_a, lat_b, lon_b)/(t_b-t_a)

    lat, lon = GEO.predict(lat_b, lon_b, bearing, distance*(t-t_b))

    return lat, lon



class Model(AbstactModel):

    name = "GEO"

    def __init__(self, CTX:dict) -> None:
        # load context
        self.CTX = CTX

    def predict(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], t:np.int64_1d[ax.sample])\
            -> np.float64_2d[ax.sample, ax.feature]:

        preds = np.zeros((len(x), 2))
        for i in range(len(x)):

            preds[i] = forecast(x[i], t[i])
        return preds


    def compute_loss(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], t:np.int64_1d[ax.sample],
                     y:np.float64_2d[ax.sample, ax.feature])\
            -> "tuple[np.float64_1d[ax.sample], np.float64_2d[ax.sample, ax.feature]]":

        preds = self.predict(x, t)
        loss = np.zeros(len(x))
        for i in range(len(x)):
            loss[i] = GEO.distance(preds[i][0], preds[i][1], y[i][0], y[i][1])

        return loss, preds



    def training_step(self, *args) -> None:
        return None


    def visualize(self, save_path:str="./_Artifacts/") -> None:
        """No visualization possible for this model"""


    def get_variables(self) -> None:
        """No variables in this model"""


    def set_variables(self, variables:object) -> None:
        """No variables in this model"""
