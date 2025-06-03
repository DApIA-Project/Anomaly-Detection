
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(GlobalArchitectureV2):

    name = "MCDCNN"

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
        
    x = TimeDistributed(Dense(64, activation="linear"))(x)
        
    
    n_vars = x.shape[-1]
    
    conv2_layers = []
    for i in range(n_vars):
        
        xf = x[:,:,i:i+1]
        conv1_layer = Conv1D(filters=CTX["UNITS"],kernel_size=5,activation='relu',padding=CTX["MODEL_PADDING"])(xf)
        conv1_layer = MaxPooling1D(pool_size=2)(conv1_layer)

        conv2_layer = Conv1D(filters=CTX["UNITS"],kernel_size=5,activation='relu',padding=CTX["MODEL_PADDING"])(conv1_layer)
        conv2_layer = MaxPooling1D(pool_size=2)(conv2_layer)
        conv2_layer = Flatten()(conv2_layer)

        conv2_layers.append(conv2_layer)

    if n_vars == 1:
        # to work with univariate time series
        concat_layer = conv2_layers[0]
    else:
        concat_layer = Concatenate(axis=-1)(conv2_layers)

    fully_connected = Dense(units=732,activation='relu')(concat_layer)
    
    return fully_connected
    