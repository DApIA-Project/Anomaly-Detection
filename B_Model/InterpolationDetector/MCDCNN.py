
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

from numpy_typing import np, ax

from _Utils.os_wrapper import os

import _Utils.Color as C
from   _Utils.Color import prntC

class Model(AbstactModel):

    name = "MCDCNN"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = Input(shape=x_input_shape, name='input')
        
        z = x

        z = TimeDistributed(Dense(64, activation="linear"))(z)
    
        n_vars = z.shape[-1]
        
        conv2_layers = []
        for i in range(n_vars):
            
            xf = z[:,:,i:i+1]
            conv1_layer = Conv1D(filters=CTX["UNITS"],kernel_size=5,activation='relu',padding=CTX["MODEL_PADDING"])(xf)
            conv1_layer = MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = Conv1D(filters=CTX["UNITS"],kernel_size=5,activation='relu',padding=CTX["MODEL_PADDING"])(conv1_layer)
            conv2_layer = MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = Concatenate(axis=-1)(conv2_layers)

        fully_connected = Dense(units=732,activation='relu')(concat_layer)
    
        # output
        z = Dense(1, activation="sigmoid")(fully_connected)
        y = z

        self.model = tf.keras.Model(x, y)


        # define loss function
        self.loss = tf.keras.losses.MeanSquaredError()

        # define optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])
        # self.opt = tf.keras.optimizers.SGD(learning_rate=CTX["LEARNING_RATE"])


    def predict(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        y_ = self.model(x)
        return self.loss(y_, y), y_

    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss, output = self.compute_loss(x, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, output



    def visualize(self, save_path="./_Artifacts/"):

        params = 0
        for i in range(len(self.model.trainable_variables)):
            params += np.prod(self.model.trainable_variables[i].shape)

        filename = os.path.join(save_path, self.name+".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)

    def nb_parameters(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])


    def get_variables(self):
        return self.model.variables

    def set_variables(self, variables):
        for i in range(len(variables)):
            self.model.variables[i].assign(variables[i])
