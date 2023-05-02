
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

    name = "CONV"

    def __init__(self, CTX:dict, name="Model"):

        self.CTX = CTX
        self.seq_len = CTX["HISTORY"]
        self.channels = CTX["FEATURES_IN"]
        self.outs = CTX["FEATURES_OUT"]
        self.dropout = CTX["DROPOUT"]
        self.nb_dense = CTX["NB_DENSE"]
        self.nb = CTX["NB_NEURONS"]
        self.nb_train = 0
    
        self.layers = []

        self.layers.append(Conv1D(16, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append("save")

        self.layers.append(Conv1D(32, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append(Conv1D(32, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append("concat save")
        self.layers.append(Conv1D(32, 1))

        self.layers.append(MaxPooling1D())


        self.layers.append("save")
        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append("concat save")
        self.layers.append(Conv1D(64, 1))

        self.layers.append(MaxPooling1D())



        self.layers.append("save")
        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())   
        self.layers.append(LeakyReLU())

        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append("concat save")
        self.layers.append(Conv1D(64, 1))

        self.layers.append(MaxPooling1D())




        self.layers.append("save")
        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())   
        self.layers.append(LeakyReLU())

        self.layers.append(Conv1D(64, 3, padding="same"))
        self.layers.append(BatchNormalization())        
        self.layers.append(LeakyReLU())

        self.layers.append("concat save")
        self.layers.append(Conv1D(64, 1))

        self.layers.append(Flatten())



        
        # softmax
        self.layers.append(Dense(self.nb))
        self.layers.append(Dropout(self.dropout))
        self.layers.append("save")
        self.layers.append(LeakyReLU())


        self.layers.append(Dense(self.nb))
        self.layers.append(Dropout(self.dropout))
        self.layers.append(LeakyReLU())

        self.layers.append(Dense(self.nb))
        self.layers.append(Dropout(self.dropout))
        self.layers.append(LeakyReLU())

        self.layers.append("concat save")

        self.layers.append(Dense(self.outs, activation="softmax"))


        self.mse = tf.keras.losses.CategoricalCrossentropy()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])



        input_shape = (self.CTX["HISTORY"], len(self.CTX["USED_FEATURES"]))
        inp = tf.keras.Input(shape=input_shape, name='input')
        output = self.__raw_call__(inp)
        self.model = tf.keras.Model(inputs=[inp], outputs=[output])
        
        

    def predict(self, x):
        return self.__call__(x)


    def __raw_call__(self, x):
        saves = {}
        for l in self.layers:
            if (type(l) == str):
                if (l[0:4] == "save"):
                    saves[l] = x

                if (l[0:6] == "concat"):
                    name = l[7:].split(" ")

                    if (len(name) == 1):
                        x = tf.concat([x, saves[name[0]]], axis=-1)
                    else:
                        a = [saves[n] for n in name]
                        x = tf.concat(a, axis=-1)
            else:
                x = l(x)
        return x
    

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, x):
        return self.model(x)

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
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)
