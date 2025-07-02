
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AbstractModel import Model as AbstactModel
from _Utils.os_wrapper import os
import numpy as np

def map_module(CTX, x):
    for _ in range(CTX["MAP_LAYERS"]):
        x = Conv2DModule(16, 3, padding=CTX["MODEL_PADDING"], 
                         batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
    x = MaxPooling2D()(x)
    for _ in range(CTX["MAP_LAYERS"]):
        x = Conv2DModule(32, 3, padding=CTX["MODEL_PADDING"], 
                         batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
        
    x = MaxPooling2D()(x)
    for _ in range(CTX["MAP_LAYERS"]-1):
        x = Conv2DModule(32, 3, padding=CTX["MODEL_PADDING"], 
                         batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
    
    x = Conv2DModule(16, 3, padding=CTX["MODEL_PADDING"], 
                        batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(x)
    
    
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    return x


    
def output_module(CTX, x_adsb, ctxs:list=None):
    x = [x_adsb]
    if (ctxs is not None):
        x.extend(ctxs)
    
    if (len(x) == 1):
        x = x[0]
    else:
        x = Concatenate()(x)
    x = Dense(CTX["LABELS_OUT"], activation="linear", name="prediction")(x)
    x = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])(x)
    return x




class GlobalArchitectureV2(AbstactModel):
    
    name = ""

    def __init__(self, CTX:dict, ads_b_module:callable):


        self.CTX = CTX

        # prepare input shapes
        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
        if (CTX["ADD_AIRPORT_CONTEXT"]): airport_input_shape = (self.CTX["AIRPORT_CONTEXT_IN"],)

        # generate layers
        inputs = []
        
        x = None
        x_takeoff = None
        x_map = None
        x_airport = None
        
        x = Input(shape=x_input_shape, name='input', batch_size=CTX["BATCH_SIZE"])
        inputs.append(x)
        
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)
            x_takeoff = takeoff
            
        if (CTX["ADD_MAP_CONTEXT"]):
            map = Input(shape=map_input_shape, name='map')
            inputs.append(map)
            
        if (CTX["ADD_AIRPORT_CONTEXT"]):
            airport = Input(shape=airport_input_shape, name='airport')
            inputs.append(airport)
            x_airport = airport
            
            
        ctx = []
        if (CTX["ADD_MAP_CONTEXT"]):
            x_map = map_module(CTX, map)
            ctx.append(x_map)
            
        x = ads_b_module(CTX, x, x_takeoff, x_airport, x_map)
        y = output_module(CTX, x, ctx)


        # generate model
        self.model = tf.keras.Model(inputs, y)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])
        # self.opt = tf.keras.optimizers.AdamW(learning_rate=self.CTX["LEARNING_RATE"])


        self.nb_train = 0


    def predict(self, x, training=False):
        return self.model(x, training=training)


    def compute_loss(self, x, y, taining=False):
        y_ = self.model(x, training=taining)
        return self.loss(y_, y), y_


    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.model(x, training=True)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_



    def visualize(self, filename="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(filename, self.name+".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)

    def nb_parameters(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])



    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.model.variables

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables)):
            self.model.variables[i].assign(variables[i])






class GlobalArchitectureV1(AbstactModel):
    
    name = ""

    def __init__(self, CTX:dict, ads_b_module:callable):


        self.CTX = CTX

        # prepare input shapes
        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
        if (CTX["ADD_AIRPORT_CONTEXT"]): airport_input_shape = (self.CTX["AIRPORT_CONTEXT_IN"],)

        # generate layers
        inputs = []
        
        x = None
        x_takeoff = None
        x_map = None
        x_airport = None
        
        x = Input(shape=x_input_shape, name='input', batch_size=CTX["BATCH_SIZE"])
        inputs.append(x)
        
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)
            x_takeoff = takeoff
            
        if (CTX["ADD_MAP_CONTEXT"]):
            map = Input(shape=map_input_shape, name='map')
            inputs.append(map)
            
        if (CTX["ADD_AIRPORT_CONTEXT"]):
            airport = Input(shape=airport_input_shape, name='airport')
            inputs.append(airport)
            x_airport = airport
            
            
            
        if (CTX["ADD_MAP_CONTEXT"]):
            x_map = map_module(CTX, map)
            
        x = ads_b_module(CTX, x, None, None, None)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            x_takeoff = ads_b_module(CTX, x_takeoff, None, None, None)
            
        x_cat = []
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_cat.append(x_takeoff)
        if (CTX["ADD_AIRPORT_CONTEXT"]): x_cat.append(x_airport)
        if (CTX["ADD_MAP_CONTEXT"]): x_cat.append(x_map)
            
        y = output_module(CTX, x, x_cat)


        # generate model
        self.model = tf.keras.Model(inputs, y)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])
        # self.opt = tf.keras.optimizers.AdamW(learning_rate=self.CTX["LEARNING_RATE"])


        self.nb_train = 0


    def predict(self, x, training=False):
        return self.model(x, training=training)


    def compute_loss(self, x, y, taining=False):
        y_ = self.model(x, training=taining)
        return self.loss(y_, y), y_


    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.model(x, training=True)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_



    def visualize(self, filename="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(filename, self.name+".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)

    def nb_parameters(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])

    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.model.variables

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables)):
            self.model.variables[i].assign(variables[i])

