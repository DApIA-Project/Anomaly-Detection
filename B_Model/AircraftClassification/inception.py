
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(TensorflowModel):

    name = "inception"

    def __init__(self, CTX:dict):
        super().__init__(CTX, ads_b_module)
    

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def _inception_module(CTX, input_tensor, stride=1, activation='linear'):

        if CTX["BOTTLENECK_SIZE"] > 0 and int(input_tensor.shape[-1]) > CTX["BOTTLENECK_SIZE"]:
            input_inception = Conv1D(filters=CTX["BOTTLENECK_SIZE"], kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [CTX["KERNEL_SIZE"] // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=CTX["UNITS"], kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=CTX["UNITS"], kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x
    
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
        
    input_res = x

    for d in range(CTX["LAYERS"]):

        x = _inception_module(CTX, x)

        if CTX["RESIDUAL"] > 0 and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)


    return gap_layer
    
    