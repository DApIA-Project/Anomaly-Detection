
# import tensorflow as tf
# from keras.layers import *

# from B_Model.AbstractModel import Model as AbstactModel
# from B_Model.Utils.TF_Modules import *


# from numpy_typing import np, ax, ax

# from _Utils.os_wrapper import os

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# class Model(AbstactModel):

#     name = "Transformer"

#     def __init__(self, CTX:dict):
#         """
#         Generate model architecture
#         Define loss function
#         Define optimizer
#         """


#         # load context
#         self.CTX = CTX
#         self.dropout = CTX["DROPOUT"]
#         self.outs = CTX["LABELS_OUT"]

#         # save the number of training steps
#         self.nb_train = 0


#         # build model's architecture
#         feature_in = self.CTX["FEATURES_IN"] * (2 if self.CTX["ADD_TAKE_OFF_CONTEXT"] else 1)
#         input_shape = (self.CTX["INPUT_LEN"], feature_in)
#         x = tf.keras.Input(shape=input_shape)
#         z = x
#         for _ in range(CTX["LAYERS"]):
#             z = transformer_encoder(z, self.CTX["HEAD_SIZE"], self.CTX["NUM_HEADS"],  self.CTX["FF_DIM"], self.CTX["DROPOUT"])

#         z = GlobalAveragePooling1D(data_format="channels_first")(z)
#         for dim in range(CTX["LAYERS"]):
#             z = Dense(self.CTX["FF_DIM"], activation="relu")(z)
#             z = Dropout(self.CTX["DROPOUT"])(z)
#         y = Dense(self.CTX["LABELS_OUT"], activation="softmax")(z)
#         self.model =  tf.keras.Model(inputs=[x], outputs=[y])


#         # define loss function
#         # self.loss = tf.keras.losses.CategoricalCrossentropy()
#         self.loss = tf.keras.losses.MeanSquaredError()

#         # define optimizer
#         self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])


#     def predict(self, x):
#         """
#         Make prediction for x
#         """
#         return self.model(x)

#     def compute_loss(self, x, y):
#         """
#         Make a prediction and compute the loss
#         that will be used for training
#         """
#         y_ = self.model(x)
#         return self.loss(y_, y), y_

#     def training_step(self, x, y):
#         """
#         Do one forward pass and gradient descent
#         for the given batch
#         """
#         with tf.GradientTape(watch_accessed_variables=True) as tape:
#             loss, output = self.compute_loss(x, y)

#             gradients = tape.gradient(loss, self.model.trainable_variables)
#             self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

#         self.nb_train += 1
#         return loss, output



#     def visualize(self, save_path="./_Artifacts/"):
#         """
#         Generate a visualization of the model's architecture
#         """

#         # Only plot if we train on CPU
#         # Assuming that if you train on GPU (GPU cluster) it mean that
#         # you don't need to check your model's architecture
#         device = tf.test.gpu_device_name()
#         if "GPU" not in device:

#             filename = os.path.join(save_path, self.name+".png")
#             tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)


#     def get_variables(self):
#         """
#         Return the variables of the model
#         """
#         return self.model.trainable_variables

#     def set_variables(self, variables):
#         """
#         Set the variables of the model
#         """
#         for i in range(len(variables)):
#             self.model.trainable_variables[i].assign(variables[i])




import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *


from _Utils.os_wrapper import os

class Model(AbstactModel):

    name = "Transformer"

    def __init__(self, CTX:dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX



        # prepare input shapes
        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)

        # generate layers
        x = tf.keras.Input(shape=x_input_shape, name='input')
        inputs = [x]
        outputs = []

        adsb_module_inputs = [x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)

            self.takeoff_module = TakeOffModule(self.CTX)
            takeoff_ctx = self.takeoff_module(takeoff)
            adsb_module_inputs.append(takeoff_ctx)
            self.TAKEOFF = len(outputs)


        if (CTX["ADD_MAP_CONTEXT"]):
            map = tf.keras.Input(shape=map_input_shape, name='map')
            inputs.append(map)

            self.map_module = MapModule(self.CTX)
            map_ctx = self.map_module(map)
            adsb_module_inputs.append(map_ctx)
            self.MAP = len(outputs)

        if (CTX["ADD_AIRPORT_CONTEXT"]):
            airport = tf.keras.Input(shape=(self.CTX["AIRPORT_CONTEXT_IN"],), name='airport')
            inputs.append(airport)

            self.airport_module = AirportModule(self.CTX)
            airport_ctx = self.airport_module(airport)
            adsb_module_inputs.append(airport_ctx)
            self.AIRPORT = len(outputs)


        self.ads_b_module = ADS_B_Module(self.CTX)
        proba = self.ads_b_module(adsb_module_inputs)
        outputs.insert(0, proba)


        # generate model
        self.model = tf.keras.Model(inputs, outputs)

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







class TakeOffModule(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]


        x = tf.keras.Input(shape=(self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), name='input')
        z = x
        for _ in range(CTX["LAYERS"]):
            z = transformer_encoder(z, self.CTX["HEAD_SIZE"], self.CTX["NUM_HEADS"],  self.CTX["FF_DIM"], self.CTX["DROPOUT"])

        z = GlobalAveragePooling1D(data_format="channels_first")(z)
        for dim in range(CTX["LAYERS"]):
            z = Dense(self.CTX["FF_DIM"], activation="relu")(z)
            z = Dropout(self.CTX["DROPOUT"])(z)

        self.transformer = tf.keras.Model(inputs=[x], outputs=[z])



    def __call__(self, x):
        return self.transformer(x)

class MapModule(tf.Module):


    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = 1
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]


        convNN = []
        for _ in range(self.layers):
            convNN.append(Conv2DModule(16, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling2D())
        for _ in range(self.layers):
            convNN.append(Conv2DModule(32, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling2D())
        for _ in range(self.layers):
            convNN.append(Conv2DModule(64, 3, padding=self.CTX["MODEL_PADDING"]))

        # convNN.append(GlobalMaxPooling2D())

        convNN.append(Conv2D(32, (2, 2), (2, 2)))
        convNN.append(BatchNormalization())
        convNN.append(Flatten())
        convNN.append(DenseModule(256, dropout=self.dropout))

        self.convNN = convNN


    def __call__(self, x):
        for layer in self.convNN:
            x = layer(x)
        return x


class AirportModule(tf.Module):
    """ very simple only a (dense module 64) * nb layers"""

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]

        denseNN = []
        for _ in range(self.layers):
            denseNN.append(DenseModule(64, dropout=self.dropout))
        self.denseNN = denseNN

    def __call__(self, x):
        for layer in self.denseNN:
            x = layer(x)
        return x


class ADS_B_Module(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]

        # identity layer
        x = tf.keras.Input(shape=(self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), name='input')
        z = x
        for _ in range(CTX["LAYERS"]):
            z = transformer_encoder(z, self.CTX["HEAD_SIZE"], self.CTX["NUM_HEADS"],  self.CTX["FF_DIM"], self.CTX["DROPOUT"])

        z = GlobalAveragePooling1D(data_format="channels_first")(z)
        for dim in range(CTX["LAYERS"]):
            z = Dense(self.CTX["FF_DIM"], activation="relu")(z)
            z = Dropout(self.CTX["DROPOUT"])(z)
        self.transformer = tf.keras.Model(inputs=[x], outputs=[z])


        self.cat = Concatenate()

        denseNN = []
        denseNN.append(Dense(self.outs, activation="linear", name="prediction"))

        self.denseNN = denseNN
        self.probability = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])

    def __call__(self, x):

        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x.pop(0)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            airport = x.pop(0)

        # preprocess
        x = adsb
        x = self.transformer(x)

        # concat takeoff and map
        cat = [x]
        if (self.CTX["ADD_MAP_CONTEXT"]):
            cat.append(map)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            cat.append(takeoff)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            cat.append(airport)

        x = self.cat(cat)

        # get prediction
        for layer in self.denseNN:
            x = layer(x)
        x = self.probability(x)
        return x


# global accuracy mean :  92.0 ( 575 / 625 )
# global accuracy count :  92.2 ( 576 / 625 )
# global accuracy max :  87.2 ( 545 / 625 )


