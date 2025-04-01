
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(TensorflowModel):

    name = "MLP"

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
        
    x = Flatten()(x)
		
    for i in range(CTX["LAYERS"]):
        x = Dropout(CTX["DROPOUT"])(x)
        x = Dense(CTX["UNITS"], activation='relu')(x)

    return x
    
    