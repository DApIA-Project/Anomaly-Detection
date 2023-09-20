
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

import numpy as np

import os
from random import randint

class Model(AbstactModel):
    """
    Convolutional neural network model for 
    aircraft classification based on 
    recordings of ADS-B data fragment.

    Parameters:
    ------------

    CTX: dict
        The hyperparameters context


    Attributes:
    ------------

    name: str (MENDATORY)
        The name of the model for mlflow logs
    
    Methods:
    ---------

    predict(x): (MENDATORY)
        return the prediction of the model

    compute_loss(x, y): (MENDATORY)
        return the loss and the prediction associated to x, y and y_

    training_step(x, y): (MENDATORY)
        do one training step.
        return the loss and the prediction of the model for this batch

    visualize(save_path):
        Generate a visualization of the model's architecture
    """

    name = "CNN"

    def __init__(self, CTX:dict):
        """ 
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX

        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.takeoff_module = TakeOffModule(self.CTX)
        self.ads_b_module = ADS_B_Module(self.CTX)

        self.loss = tf.keras.losses.MeanSquaredError()
        self.nb_train = 0

        
    def predict(self, x):
        """
        Make prediction for x 
        """
        inputs = [x[0]]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff_ctx, _ = self.takeoff_module(x[1])
            inputs.append(takeoff_ctx)

        return self.ads_b_module(inputs)[1]

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        inputs = [x[0]]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff_ctx, _ = self.takeoff_module(x[1])
            inputs.append(takeoff_ctx)
        y_ = self.ads_b_module(inputs)

        loss = self.loss(y_[0], y)

        return loss, y_[1]

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        # y *= 0.
        train_adsb = True
        adsb = x[0]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x[1]
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x[2]
        
        adsb_inputs = [adsb]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            if (randint(0, 32-1) == 0):
                train_adsb = False
                loss, (takeoffctx, _) = self.takeoff_module.training_step(takeoff, y)

            else:
                takeoffctx, _ = self.takeoff_module(takeoff)

            adsb_inputs.append(takeoffctx)

        if (train_adsb):
            loss, outputs = self.ads_b_module.training_step(adsb_inputs, y)
        else:
            outputs = self.ads_b_module(adsb_inputs)

        self.nb_train += 1
        loss = self.loss(outputs[0], y)
        return loss, outputs[1]



    def visualize(self, save_path="./_Artefact/"):
        """
        Generate a visualization of the model's architecture
        """
        adsb = tf.keras.Input(shape=(self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), name='ads-b')
        inputs = [adsb]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            take_off = tf.keras.Input(shape=(self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), name='take_off')
            inputs.append(take_off)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            # inputs.append(tf.keras.Input(shape=(MapModule.CTX_SIZE, ), name='map'))
            pass

        adsb_inputs = [adsb]
        outputs = []
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff_ctx, y_takeoff = self.takeoff_module(take_off)
            adsb_inputs.append(takeoff_ctx)
            outputs.append(y_takeoff)

        y_ = self.ads_b_module(adsb_inputs)
        outputs.append(y_)
    
        model = tf.keras.Model(inputs, outputs)
            
        filename = os.path.join(save_path, self.name+".png")
        tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)



    def getVariables(self):
        """
        Return the variables of the model
        """
        return [self.takeoff_module.model.trainable_variables, self.ads_b_module.model.trainable_variables]

    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables[0])):
            self.takeoff_module.model.trainable_variables[i].assign(variables[0][i])
        for i in range(len(variables[1])):
            self.ads_b_module.model.trainable_variables[i].assign(variables[1][i])







class TakeOffModule(tf.Module):

    CTX_SIZE = 128

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

        input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=input_shape, name='takeoff')
        z = x
        for _ in range(self.layers):
            z = Conv1DModule(64, 3, padding="same")(z)
        z = MaxPooling1D()(z)
        for _ in range(self.layers):
            z = Conv1DModule(128, 3, padding="same")(z)

        z = Flatten()(z)
        out_ctx = DenseModule(self.CTX_SIZE, dropout=self.dropout)(z)
        gradiant_skip = Dense(self.outs, activation="sigmoid", name="skip")(out_ctx)

        self.model = tf.keras.Model(x, [out_ctx, gradiant_skip])

        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])

    def __call__(self, x):
        return self.model(x)
    
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            variables = self.model.trainable_variables
            tape.watch(variables)
            
            ctx, y_ = self.model(x)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, variables)
            self.opt.apply_gradients(zip(gradients, variables))

        return loss, (ctx, y_)
    

class ADS_B_Module(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

        input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        adsb_x = tf.keras.Input(shape=input_shape, name='ads-b')

        adsb = adsb_x

        for _ in range(self.layers):
            adsb = Conv1DModule(128, 3, padding="same")(adsb)
        adsb = MaxPooling1D()(adsb)

        
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): 
            takeoff_input_shape = (TakeOffModule.CTX_SIZE, )
            takeoff_x = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            takeoff = takeoff_x
            takeoff = RepeatVector(self.CTX["INPUT_LEN"] // 2)(takeoff)

            adsb = Concatenate()([adsb, takeoff])
        
        if (CTX["ADD_MAP_CONTEXT"]):
            pass

        for _ in range(self.layers):
            adsb = Conv1DModule(256, 3, padding="same")(adsb)
        adsb = Conv1DModule(self.outs, 3, padding="same")(adsb)


        adsb = Flatten()(adsb)
        adsb_raw = Dense(self.outs, activation="linear")(adsb)
        # apply sigmoid 
        adsb = Activation("sigmoid")(adsb_raw)

        adsb_y = adsb
        adsb_y_raw = adsb_raw


        inputs = [adsb_x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): inputs.append(takeoff_x)
        # if (CTX["ADD_MAP_CONTEXT"]): inputs.append(map)

        self.model = tf.keras.Model(inputs, [adsb_y, adsb_y_raw])

        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])

    def __call__(self, x):
        return self.model(x)
    

    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            variables = self.model.trainable_variables
            tape.watch(variables)
            
            y_ = self.model(x) # return sigmoid output for training
            loss = self.loss(y_[0], y)

            gradients = tape.gradient(loss, variables)
            self.opt.apply_gradients(zip(gradients, variables))

        return loss, y_

