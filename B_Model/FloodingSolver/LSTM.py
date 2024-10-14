
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

from numpy_typing import np, ax, ax

from _Utils.os_wrapper import os

import _Utils.Color as C
from   _Utils.Color import prntC

class Model(AbstactModel):

    name = "LSTM"

    def __init__(self, CTX:dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """


        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]
        self.outs = CTX["FEATURES_OUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=x_input_shape, name='input')
        # inputs = [x]

        z = x


        n = self.CTX["LAYERS"]
        # on 8 features keep 2:8
        # z_gps = z[:, :, :2]
        # z_other = z[:, :, 2:]

        # z = Conv1DModule(128, 1)(z_gps)
        # for _ in range(n-1):
        #     # dilatation = int(self.CTX["DILATION_RATE"] ** i)
        #     res = z * self.CTX["RESUDUAL"]
        #     z = LSTM(128, return_sequences=True, dropout=self.dropout)(z)

        #     z = Add()([z, res])
        # z_gps= LSTM(128, return_sequences=False)(z)

        z = Conv1DModule(128, 1)(z)
        for _ in range(n-1):
            # dilatation = int(self.CTX["DILATION_RATE"] ** i)
            res = z * self.CTX["RESUDUAL"]
            z = LSTM(128, return_sequences=True, dropout=self.dropout)(z)

            z = Add()([z, res])
        z = LSTM(128, return_sequences=False)(z)

        # z = Concatenate()([z_gps, z_other])




        # z = DenseModule(256, dropout=self.dropout)(z)
        z = Dense(self.outs, activation="sigmoid")(z)
        y = z
        # y = (z + 1) / 2


        self.model = tf.keras.Model(x, y)


        # define loss function
        self.loss = tf.keras.losses.MeanSquaredError()

        # define optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])


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
        return self.loss(y_, y), y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss, output = self.compute_loss(x, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, output



    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        params = 0
        for i in range(len(self.model.trainable_variables)):
            params += np.prod(self.model.trainable_variables[i].shape)

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
