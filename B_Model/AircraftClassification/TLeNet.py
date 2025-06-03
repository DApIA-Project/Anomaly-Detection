
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(GlobalArchitectureV2):

    name = "TLeNet"

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
        
    conv_1 = Conv1D(filters=5,kernel_size=5,activation='relu', padding='same')(x)
    conv_1 = MaxPool1D(pool_size=2)(conv_1)
    
    conv_2 = Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
    conv_2 = MaxPool1D(pool_size=4)(conv_2)
    
    # they did not mention the number of hidden units in the fully-connected layer
    # so we took the lenet they referenced 
    
    flatten_layer = Flatten()(conv_2)
    fully_connected_layer = Dense(500,activation='relu')(flatten_layer)
    
    
    return fully_connected_layer
    