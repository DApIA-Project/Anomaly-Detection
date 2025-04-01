
from keras.layers import *
from B_Model.Utils.TF_Modules import *
from B_Model.AbstractModel import Model as AbstactModel
from _Utils.os_wrapper import os

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
    return x


    
def output_module(CTX, x_adsb, x_map=None):
    x = [x_adsb]
    if (x_map is not None):
        x.append(x_map)
    
        
    x = Concatenate()(x)
    x = Dense(CTX["LABELS_OUT"], activation="linear", name="prediction")(x)
    x = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])(x)
    return x




class TensorflowModel(AbstactModel):
    
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
        
        x = Input(shape=x_input_shape, name='input')
        inputs.append(x)
        
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = Input(shape=takeoff_input_shape, name='takeoff', batch_size=CTX["BATCH_SIZE"])
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
            
        x = ads_b_module(CTX, x, x_takeoff, x_airport, x_map)
        y = output_module(CTX, x, x_map)


        # generate model
        self.model = tf.keras.Model(inputs, y)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])

        self.nb_train = 0


    def predict(self, x):
        """
        Make prediction for x
        """
        return self.model(x)

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.model(x)
        loss = self.loss(y_, y)
        return loss, y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.model(x)
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



    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.model.trainable_variables

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables)):
            self.model.trainable_variables[i].assign(variables[i])


