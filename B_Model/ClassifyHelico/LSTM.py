
import math
import tensorflow as tf
from keras.layers import *
import numpy as np


class Model(tf.Module):
    """
    Class for managing the data processing
    
    Must always contain :
    
    Methods
    -------
    predict(x):
        return the prediction of the model
    compute_loss(x, y):
        return the loss and the prediction of the model
    training_step(x, y):
        return the loss and the prediction of the model
        and apply the gradient descent
    """

    name = "LSTM"

    def __init__(self, CTX:dict, name="Model"):

        self.seq_len = CTX["HISTORY"]
        self.channels = CTX["FEATURES_IN"]
        self.outs = CTX["FEATURES_OUT"]
        self.nb_lstm = CTX["NB_LSTM"]
        self.nb_units = CTX["NB_UNITS"]
        self.dropout = CTX["DROPOUT"]
        self.nb_dense = CTX["NB_DENSE"]
        self.nb = CTX["NB_NEURONS"]
        self.nb_train = 0
    
        self.layers = []

        for i in range(self.nb_lstm-1):
            self.layers.append(LSTM(self.nb_units, return_sequences=True, dropout=self.dropout))
        self.layers.append(LSTM(self.nb_units, return_sequences=False, dropout=self.dropout))
        # softmax
        for i in range(self.nb_dense):
            self.layers.append(Dense(self.nb))
            self.layers.append(Dropout(self.dropout))
            self.layers.append(LeakyReLU())

        self.layers.append(Dense(self.outs, activation="softmax"))


        self.mse = tf.keras.losses.CategoricalCrossentropy()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])
        

    def predict(self, x):
        return self.__call__(x)

    def __raw_call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    @tf.function(experimental_relax_shapes=True)
    def __call__(self, x):
        return self.__raw_call__(x)
    

    def compute_loss(self, x, y):
        y_ = self.__call__(x)

        return self.mse(y_, y), y_

    def training_step(self, x, y):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss, output = self.compute_loss(x, y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.trainable_variables))

        self.nb_train += 1
        return loss, output


    def visualize(self, filename="./Output_artefact/model.png"):
        input_shape = (self.CTX["NB_BATCH"], self.CTX["BATCH_SIZE"], len(self.CTX["USED_FEATURES"]))
        
        # get a keras.plot_model image
        input = tf.keras.Input(shape=input_shape, dtype='int32', name='input')
        output = self.__raw_call__(input)
        
        model = tf.keras.Model(inputs=[input], outputs=[output])
        
        tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)
