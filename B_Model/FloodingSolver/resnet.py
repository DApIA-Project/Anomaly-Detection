
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

    name = "resnet"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=x_input_shape, name='input')
        
        n_feature_maps = CTX["UNITS"]

        conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(x)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_1 = add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_3 = add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # FINAL
        gap_layer = GlobalAveragePooling1D()(output_block_3)
    
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


    @tf.function
    def predict(self, x, training=False):
        return self.model(x, training=training)


    @tf.function
    def compute_loss(self, x, y, taining=False):
        y_ = self.model(x, training=taining)
        return self.loss(y_, y), y_
    

    @tf.function
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
        return np.sum([np.prod(list(v._shape)) for v in self.model.trainable_variables])


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
