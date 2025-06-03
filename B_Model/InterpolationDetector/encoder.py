
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

from numpy_typing import np, ax

from _Utils.os_wrapper import os

import _Utils.Color as C
from   _Utils.Color import prntC

class Model(AbstactModel):

    name = "encoder"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = Input(shape=x_input_shape, name='input', batch_size=CTX["BATCH_SIZE"])
        
        # conv block -1
        conv1 = Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(x)
        # InstanceNormalization()
        conv1 = BatchNormalization(axis=[0, -1])(conv1)
        conv1 = PReLU(shared_axes=[1])(conv1)
        conv1 = Dropout(rate=0.2)(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
        conv2 = BatchNormalization(axis=[0, -1])(conv2)
        conv2 = PReLU(shared_axes=[1])(conv2)
        conv2 = Dropout(rate=0.2)(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
        conv3 = BatchNormalization(axis=[0, -1])(conv3)
        conv3 = PReLU(shared_axes=[1])(conv3)
        conv3 = Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = Lambda(lambda x: x[:,:,:256])(conv3)
        attention_softmax = Lambda(lambda x: x[:,:,256:])(conv3)
        # attention mechanism
        attention_softmax = Softmax()(attention_softmax)
        multiply_layer = Multiply()([attention_softmax,attention_data])
        # last layer
        dense_layer = Dense(units=256,activation='sigmoid')(multiply_layer)
        dense_layer = BatchNormalization(axis=[0, -1])(dense_layer)
        # output
        flatten_layer = Flatten()(dense_layer)

        z = Dense(1, activation="sigmoid")(flatten_layer)
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



    def get_variables(self):
        return self.model.trainable_variables

    def set_variables(self, variables):
        for i in range(len(variables)):
            self.model.trainable_variables[i].assign(variables[i])
