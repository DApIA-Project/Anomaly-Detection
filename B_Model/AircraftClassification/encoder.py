
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(GlobalArchitectureV2):

    name = "encoder"

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
        
    
    # conv block -1
    conv1 = Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(x)
    # InstanceNormalization()
    conv1 = BatchNormalization(axis=[0, -1])(conv1)
    conv1 = PReLU(shared_axes=[1])(conv1)
    conv1 = Dropout(rate=0.2)(conv1)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
    conv2 = BatchNormalization(axis=[0, -1])(conv2)
    conv2 = PReLU(shared_axes=[1])(conv2)
    conv2 = Dropout(rate=0.2)(conv2)
    conv2 = MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
    conv3 = BatchNormalization(axis=[0, -1])(conv3)
    conv3 = PReLU(shared_axes=[1])(conv3)
    conv3 = Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = Lambda(lambda x: x[:,:,:256])(conv3)
    attention_softmax = Lambda(lambda x: x[:,:,256:])(conv3)
    # attention mechanism
    attention_softmax = Softmax()(attention_softmax)
    multiply_layer = Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = Dense(units=256,activation='sigmoid')(multiply_layer)
    dense_layer = BatchNormalization(axis=[0, -1])(dense_layer)
    # output layer
    flatten_layer = Flatten()(dense_layer)
    return flatten_layer
    
    