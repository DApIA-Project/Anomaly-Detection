
from B_Model.AbstractModel import Model as AbstactModel

import _Utils.FeatureGetter as FG
import _Utils.geographic_maths as GEO
from _Utils import Limits
import D_DataLoader.Utils as U

import numpy as np


def eval(y, y_):
    associated = {}
    errors = 0
    for t in range(len(y)):
        if y[t] not in associated:
            associated[y[t]] = y_[t]
        else:
            if associated[y[t]] != y_[t]:
                errors += 1
                associated[y[t]] = y_[t]

    return errors/len(y)

def forecast(history, t):
    if (len(history) == 0):
        return None
    if (len(history) == 1):
        return FG.lat(history[-1]), FG.lon(history[-1])
    # return FG.lat(history[-1]), FG.lon(history[-1])
    
    
    lat_a = FG.lat(history[-2])
    lon_a = FG.lon(history[-2])
    t_a = FG.timestamp(history[-2])
    lat_b = FG.lat(history[-1])
    lon_b = FG.lon(history[-1])
    t_b = FG.timestamp(history[-1])

    bearing = GEO.bearing(lat_a, lon_a, lat_b, lon_b)
    distance = GEO.distance(lat_a, lon_a, lat_b, lon_b)/(t_b-t_a)
    return GEO.predict(lat_b, lon_b, bearing, distance*(t-t_b))



class Model(AbstactModel):

    name = "ALG"

    def __init__(self, CTX:dict):
        # load context
        self.CTX = CTX
        
    def predict(self, x):
        parents = {}
        sub_flights = []
        y_ = np.zeros(len(x), dtype=int)

        unique_t = []
        x_order = {}
        x_per_t = {}

        for i in range(len(x)):
            t = FG.timestamp(x[i])
            if t not in x_per_t:
                x_per_t[t] = []
                x_order[t] = []
                unique_t.append(t)
            x_per_t[t].append(x[i])
            x_order[t].append(i)

        i = 0
        for t in unique_t:
            msgs = x_per_t[t]
            order = x_order[t]
            dist_matrix = np.zeros((len(msgs), len(sub_flights)))



            for s in range(len(sub_flights)):
                lat, lon = forecast(sub_flights[s], t)
                for m in range(len(msgs)):
                    dist_matrix[m][s] = GEO.distance(FG.lat(msgs[m]), FG.lon(msgs[m]), lat, lon) 
            
            # Assignation
            # each time associate the message with the minimum distance with the subflight
            associated = np.zeros(len(msgs))
            run = (len(sub_flights) > 0)
            while (run):
                min_i = np.argmin(dist_matrix)
                min_m, min_s = min_i // len(sub_flights), min_i % len(sub_flights)
                    
                if dist_matrix[min_m][min_s] < 300:
                    associated[min_m] = 1
                    sub_flights[min_s].append(msgs[min_m])   
                    y_[order[min_m]] = min_s           
                    dist_matrix[min_m, :] = Limits.INT_MAX
                    dist_matrix[:, min_s] = Limits.INT_MAX
                else:
                    break
                

            # manage remaining messages
            for m in range(len(msgs)):
                if associated[m] == 0:
                    sub_flights.append([msgs[m]])
                    y_[order[m]] = len(sub_flights)-1
            i += 1
        
        return y_
            
        


    def compute_loss(self, x, y):
        """
        Make a prediction and compute the lossSequelize
        that will be used for training
        """
        """
        Make prediction for x 
        """
        y_ = self.predict(x)
        print(len(y), len(y_))
        loss = eval(y, y_)
        return loss, y_

    

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


    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        pass
