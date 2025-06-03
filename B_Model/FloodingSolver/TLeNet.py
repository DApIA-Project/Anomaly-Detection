
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from B_Model.Utils.MulAttention import MulAttention

from numpy_typing import np, ax

from _Utils.os_wrapper import os

import _Utils.Color as C
from   _Utils.Color import prntC


class Model(AbstactModel):

    name = "TLeNet"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=x_input_shape, name='input')
        
        conv_1 = Conv1D(filters=5,kernel_size=5,activation='relu', padding='same')(x)
        conv_1 = MaxPool1D(pool_size=2)(conv_1)
        
        conv_2 = Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
        conv_2 = MaxPool1D(pool_size=4)(conv_2)
        
        # they did not mention the number of hidden units in the fully-connected layer
        # so we took the lenet they referenced 
        
        flatten_layer = Flatten()(conv_2)
        fully_connected_layer = Dense(500,activation='relu')(flatten_layer)
    
        # output
        z = Dense(CTX["FEATURES_OUT"], activation="linear")(fully_connected_layer)
        
        if (CTX["ACTIVATION"] != "linear"):
            z = Activation(CTX["ACTIVATION"])(z)
        y = z

        self.model = tf.keras.Model(x, y)


        # define loss function
        self.loss = tf.keras.losses.MeanSquaredError()

        # define optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])


    def predict(self, x, training=False):
        return self.model(x, training=training)


    def compute_loss(self, x, y, taining=False):
        y_ = self.model(x, training=taining)
        return self.loss(y_, y), y_
    

    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss, output = self.compute_loss(x, y, taining=True)

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
