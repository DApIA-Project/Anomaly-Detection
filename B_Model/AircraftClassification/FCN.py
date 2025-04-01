
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(TensorflowModel):

    name = "FCN"

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
        
    
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    gap_layer = GlobalAveragePooling1D()(conv3)

    return gap_layer
    
    