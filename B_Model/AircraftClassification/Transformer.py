
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AircraftClassification.Utils import *



class Model(GlobalArchitectureV2):

    name = "Transformer"

    def __init__(self, CTX:dict):
        super().__init__(CTX, ads_b_module)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def ads_b_module(CTX, x, x_takeoff, airport, x_map):
    
    cat = [x]
    if (x_takeoff is not None):
        cat.append(x_takeoff)
        
    if (airport is not None):
        airport = RepeatVector(CTX["INPUT_LEN"])(airport)
        cat.append(airport)
        
    if (x_map is not None):
        x_map_ = RepeatVector(CTX["INPUT_LEN"])(x_map)
        cat.append(x_map_)
        
    if (len(cat) > 1):
        x = Concatenate()(cat)
        
    
    for _ in range(CTX["LAYERS"]):
        x = transformer_encoder(x, CTX["HEAD_SIZE"], CTX["NUM_HEADS"], CTX["FF_DIM"], CTX["DROPOUT"])

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    
    if (x_map is not None):
        x = Concatenate()([x, x_map])
        
    for dim in range(CTX["LAYERS"]):
        x = Dense(CTX["FF_DIM"], activation="relu")(x)
        x = Dropout(CTX["DROPOUT"])(x)
    x = Dense(CTX["LABELS_OUT"], activation="linear")(x)
    x = Activation(CTX["ACTIVATION"])(x)
    
    return x
    

