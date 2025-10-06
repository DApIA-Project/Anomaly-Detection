
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from B_Model.Utils.MulAttention import MulAttention

from numpy_typing import np, ax

from _Utils.os_wrapper import os

import _Utils.Color as C
from   _Utils.Color import prntC

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def _inception_module(CTX, input_tensor, stride=1, activation='linear'):

    if CTX["BOTTLENECK_SIZE"] > 0 and int(input_tensor.shape[-1]) > CTX["BOTTLENECK_SIZE"]:
        input_inception = Conv1D(filters=CTX["BOTTLENECK_SIZE"], kernel_size=1,
                                                padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [CTX["KERNEL_SIZE"] // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(Conv1D(filters=CTX["UNITS"], kernel_size=kernel_size_s[i],
                                                strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(filters=CTX["UNITS"], kernel_size=1,
                                    padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

class Model(AbstactModel):

    name = "inception"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=x_input_shape, name='input')
        z = x
        z = TimeDistributed(Dense(64, activation="linear"))(z)
        
        input_res = z
        
        for d in range(CTX["LAYERS"]):

            z = _inception_module(CTX, z)

            if CTX["RESIDUAL"] > 0 and d % 3 == 2:
                z = _shortcut_layer(input_res, z)
                input_res = z

        gap_layer = GlobalAveragePooling1D()(z)
        # output
        z = Dense(CTX["FEATURES_OUT"], activation="linear")(gap_layer)
        
        if (CTX["ACTIVATION"] != "linear"):
            z = Activation(CTX["ACTIVATION"])(z)
        y = z

        self.model = tf.keras.Model(x, y)


        # define loss function
        self.loss = tf.keras.losses.MeanSquaredError()

        # define optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])


<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
    def predict(self, x, training=False):
        return self.model(x, training=training)


<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
    def compute_loss(self, x, y, taining=False):
        y_ = self.model(x, training=taining)
        return self.loss(y_, y), y_
    

<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
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

    def nb_parameters(self):
<<<<<<< HEAD
        return np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])
=======
        return np.sum([np.prod(list(v._shape)) for v in self.model.trainable_variables])
>>>>>>> master


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
