
import tensorflow as tf
from keras.api.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

from numpy_typing import np, ax

from _Utils.os_wrapper import os
from random import randint

class Model(AbstactModel):

    name = "CNN"

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

        self.TAKEOFF = None
        self.MAP = None
        self.PROBA = None

        adsb_module_inputs = [x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)

            self.takeoff_module = TakeOffModule(self.CTX)
            takeoff_ctx, takeoff_skip = self.takeoff_module(takeoff)
            adsb_module_inputs.append(takeoff_ctx)
            outputs.append(takeoff_skip)
            self.TAKEOFF = len(outputs)


        if (CTX["ADD_MAP_CONTEXT"]):
            map = tf.keras.Input(shape=map_input_shape, name='map')
            inputs.append(map)

            self.map_module = MapModule(self.CTX)
            map_ctx, mapskip = self.map_module(map)
            adsb_module_inputs.append(map_ctx)
            outputs.append(mapskip)
            self.MAP = len(outputs)


        self.ads_b_module = ADS_B_Module(self.CTX)
        proba = self.ads_b_module(adsb_module_inputs)
        outputs.insert(0, proba)
        self.PROBA = 0





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
        if (not(self.CTX["ADD_TAKE_OFF_CONTEXT"]) and not(self.CTX["ADD_MAP_CONTEXT"])):
            return self.model(x)

        return self.model(x)[self.PROBA]

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.model(x)

        if (not(self.CTX["ADD_TAKE_OFF_CONTEXT"]) and not(self.CTX["ADD_MAP_CONTEXT"])):
            loss = self.loss(y_, y)
            return loss, y_

        loss = self.loss(y_[self.PROBA], y)
        return loss, y_[self.PROBA]

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """

        skip_w = self.CTX["SKIP_CONNECTION"]


        with tf.GradientTape(watch_accessed_variables=True) as tape:


            y_ = self.model(x)
            loss = self.loss(y_[self.PROBA], y)


            if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
                takeoff_loss = self.loss(y_[self.TAKEOFF], y)
                loss += takeoff_loss * skip_w

            if (self.CTX["ADD_MAP_CONTEXT"]):
                map_loss = self.loss(y_[self.MAP], y)
                loss += map_loss * skip_w

            if (not(self.CTX["ADD_TAKE_OFF_CONTEXT"]) and not(self.CTX["ADD_MAP_CONTEXT"])):
                loss = self.loss(y_, y)



            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        if (not(self.CTX["ADD_TAKE_OFF_CONTEXT"]) and not(self.CTX["ADD_MAP_CONTEXT"])):
            return loss, y_
        return loss, y_[self.PROBA]



    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(save_path, self.name+".png")
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

    CTX_SIZE = 128

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]


        convNN = []
        for _ in range(self.layers):
            convNN.append(Conv1DModule(128, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling1D())
        for _ in range(self.layers):
            convNN.append(Conv1DModule(128, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(Flatten())
        convNN.append(DenseModule(self.CTX_SIZE, dropout=self.dropout))

        self.convNN = convNN
        self.gradiant_skip = Dense(self.outs, activation=CTX["ACTIVATION"], name="skip")


    def __call__(self, x):
        for layer in self.convNN:
            x = layer(x)
        return x, self.gradiant_skip(x)

class MapModule(tf.Module):

    CTX_SIZE = 128

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
            convNN.append(Conv2DModule(32, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(Flatten())
        convNN.append(DenseModule(self.CTX_SIZE, dropout=self.dropout))

        self.convNN = convNN
        self.gradiant_skip = Dense(self.outs, activation=CTX["ACTIVATION"], name="map_skip")


    def __call__(self, x):
        for layer in self.convNN:
            x = layer(x)
        return x, self.gradiant_skip(x)



class ADS_B_Module(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]

        preNN = []
        for _ in range(self.layers):
            preNN.append(Conv1DModule(256, 3, padding=self.CTX["MODEL_PADDING"]))
        preNN.append(MaxPooling1D())
        convNN = []
        for _ in range(self.layers):
            convNN.append(Conv1DModule(256, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(Conv1DModule(self.outs, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(GlobalMaxPooling1D())
        # convNN.append(Dense(self.outs, activation="linear", name="prediction"))

        valid_padding_to_remove = CTX["LAYERS"] * (CTX["MODEL_PADDING"] == "valid")

        self.preNN = preNN
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.takeoff = RepeatVector(self.CTX["INPUT_LEN"] // 2 - valid_padding_to_remove)
        if (CTX["ADD_MAP_CONTEXT"]):
            self.map = RepeatVector(self.CTX["INPUT_LEN"] // 2 - valid_padding_to_remove)
        self.cat = Concatenate()
        self.convNN = convNN
        self.probability = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])

    def __call__(self, x):

        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x.pop(0)

        # preprocess
        x = adsb
        for layer in self.preNN:
            x = layer(x)

        # concat takeoff and ctx
        to_cat = [x]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = self.takeoff(takeoff)
            to_cat.append(takeoff)

        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = self.map(map)
            to_cat.append(map)

        if (len(to_cat)>1):
            x = self.cat(to_cat)

        # get prediction
        for layer in self.convNN:
            x = layer(x)
        x = self.probability(x)
        return x


# global accuracy mean :  92.0 ( 575 / 625 )
# global accuracy count :  92.2 ( 576 / 625 )
# global accuracy max :  87.2 ( 545 / 625 )