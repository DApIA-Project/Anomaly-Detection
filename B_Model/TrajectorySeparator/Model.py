
from B_Model.AbstractModel import Model as AbstactModel

import _Utils.FeatureGetter as FG
import _Utils.geographic_maths as GEO
import D_DataLoader.Utils as U

import numpy as np

def lerp(a, b, t):
    return a + (b-a)*t



def forecast(history:np.ndarray, t:int):
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

    name = "ALG"

    def __init__(self, CTX:dict):
        # load context
        self.CTX = CTX

    def predict(self, x:np.ndarray, t:np.ndarray) -> np.ndarray:

        preds = np.zeros((len(x), 2))
        for i in range(len(x)):

            preds[i] = forecast(x[i], t[i])
        return preds


    def compute_loss(self, x:np.ndarray, y:np.ndarray):
        """
        Make a prediction and compute the lossSequelize
        that will be used for training
        """
        """
        Make prediction for x
        """
        preds = self.predict(x)
        loss = np.zeros(len(x))
        for i in range(len(x)):
            loss[i] = GEO.distance(preds[i][0], preds[i][1], y[i][0], y[i][1])

        return loss, preds




    def training_step(self, x, y):
        """
        Fit the model, add new data !
        """
        return None



    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """
        pass


    def getVariables(self):
        """
        Return the variables of the model
        """
        return 0


    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        pass
