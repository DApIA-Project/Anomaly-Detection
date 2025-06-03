
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(GlobalArchitectureV2):

    name = "LSTM"

    def __init__(self, CTX:dict):
        super().__init__(CTX, ads_b_module)



def ads_b_module(CTX, x, x_takeoff, airport, x_map):
    
    cat = [x]
    if (x_takeoff is not None):
        cat.append(x_takeoff)
        
    if (airport is not None):
        airport = RepeatVector(CTX["INPUT_LEN"])(airport)
        cat.append(airport)
        
    if (x_map is not None):
        x_map = RepeatVector(CTX["INPUT_LEN"])(x_map)
        cat.append(x_map)
        
    if (len(cat) > 1):
        x = Concatenate()(cat)
        
    
    x = TimeDistributed(Dense(CTX["UNITS"], activation="linear"))(x)
    x_skip = x
    for _ in range(CTX["LAYERS"]):
        x = LSTM(CTX["UNITS"], return_sequences=True)(x)
        if (CTX["RESIDUAL"] > 0):
            x_skip = x_skip * CTX["RESIDUAL"]
            x = Add()([x, x_skip])
            
    x = LSTM(CTX["UNITS"], return_sequences=False)(x)
    
    return x
    