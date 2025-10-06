
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from reservoirpy.nodes import Reservoir


from numpy_typing import np, ax
from _Utils.os_wrapper import os

def __predict_reservoir__(CTX:dict, reservoir, x):
    y_ = np.zeros((len(x), CTX["INPUT_LEN"], reservoir.output_dim))

    for s in range(len(x)):
        # reset the reservoir state
        reservoir.reset()
        # give the time series to the reservoir
        for t in range(len(x[s])):
            y_[s, t] = reservoir(x[s][t].reshape(1, -1))

    return y_


class Model(AbstactModel):

    name = "Reservoir"

    def __init__(self, CTX:dict):

        self.CTX = CTX

        self.model = Module(CTX)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])

        self.nb_train = 0



    @tf.function
    def predict(self, x, training=False):
        return self.model.predict(x, training=training)
    


    @tf.function
    def compute_loss(self, x, y, training=False):
        y_ = self.predict(x, training=training)
        loss = self.loss(y_, y)
        return loss, y_
    
    

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.predict(x, training=True)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_
    


    def visualize(self, filename="./_Artifacts/"):
        filename = os.path.join(filename, self.name+".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)

    def nb_parameters(self):
        return np.sum([np.prod(list(v._shape)) for v in self.model.trainable_variables])
    
    def get_variables(self):
        return self.model.get_variables()
    
    
    
    def set_variables(self, variables):
        self.model.set_variables(variables)




class Module(tf.Module):
    
    def __init__(self, CTX:dict):

        self.CTX = CTX

        self.reservoir = Reservoir(self.CTX["R_UNITS"],
            lr=self.CTX["LR"],
            sr=self.CTX["SR"],
            input_connectivity=self.CTX["INPUT_CONNECTIVITY"],
            rc_connectivity=self.CTX["RC_CONNECTIVITY"],
        )

        self.map_model = None
        
        if (CTX["ADD_MAP_CONTEXT"]): 
            map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
            x_map = Input(shape=map_input_shape, name='map')
            y_map = map_module(CTX, x_map)
            self.map_model = tf.keras.Model(x_map, y_map)
            
        readout_input_shape = (self.CTX["INPUT_LEN"], self.CTX["R_UNITS"])
        map_input_shape = (self.CTX["INPUT_LEN"], self.CTX["UNITS"])
        
        x_readout = Input(shape=readout_input_shape, name='readout')
        x_map = Input(shape=map_input_shape, name='map')
        y_readout = output_module(CTX, x_readout, x_map)
        self.readout_model = tf.keras.Model([x_readout, x_map], y_readout)


    @tf.function
    def predict(self, x_, training=False):
        x = x_.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): 
            takeoff = x_.pop(0)
            x = np.concatenate([x, takeoff], axis=2)
            
        if (self.CTX["ADD_MAP_CONTEXT"]): 
            map = x_.pop(0)
            map = self.map_model(map, training=training)
            x = np.concatenate([x, map], axis=2)
            
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): 
            airport = x_.pop(0)
            airport = np.repeat(airport[:, None], self.CTX["INPUT_LEN"], axis=1)
            x = np.concatenate([x, airport], axis=2)

        z =__predict_reservoir__(self.CTX, self.reservoir, x)
        
        y = self.readout_model([z, map], training=training)

        return y
    
    def get_variables(self):
        return self.variables, self.reservoir



    def set_variables(self, variables):
        model_vars, reservoir = variables
        for i in range(len(model_vars)):
            self.variables[i].assign(model_vars[i])

        self.reservoir = reservoir
        
        
    

def map_module(CTX, x):
    for _ in range(CTX["MAP_LAYERS"]):
        x = Conv2DModule(16, 3, padding=CTX["MODEL_PADDING"], 
                         batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
    x = MaxPooling2D()(x)
    for _ in range(CTX["MAP_LAYERS"]):
        x = Conv2DModule(32, 3, padding=CTX["MODEL_PADDING"], 
                         batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = DenseModule(CTX["UNITS"], CTX["DROPOUT"])(x)
    x = RepeatVector(CTX["INPUT_LEN"])(x)
    return x

    
    
def output_module(CTX, x, x_map):
    x = Concatenate()([x, x_map])
    x = TimeDistributed(Dense(CTX["UNITS"], activation="linear"))(x)
    x = Flatten()(x)
    x = Dense(CTX["LABELS_OUT"], activation="linear", name="prediction")(x)
    x = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])(x)
    return x


