
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

import os


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
        units = CTX["UNITS"]

        # save the number of training steps
        self.nb_train = 0
    

        # build model's architecture
        input_shape = (self.CTX["TIMESTEPS"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=input_shape, name='input')
        z = x

        # stem layer
        z = Conv1D(self.CTX["UNITS"], 9, strides=4, padding="same")(z)

        n = self.CTX["LAYERS"]
        residual_freq = self.CTX["RESIDUAL"]
        r = 0

        save = z

        for i in range(n):
            seq = (i < n-1)
            z = LSTM(self.CTX["UNITS"], dropout=self.dropout, return_sequences=seq)(z)

            r += 1
            if (r == residual_freq):
                z = Add()([z, save])
                save = z
                r = 0

        z = Dense(self.outs, activation=self.CTX["ACTIVATION"])(z)

        y = z
        self.model = tf.keras.Model(inputs=[x], outputs=[y])


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



    def visualize(self, save_path="./_Artefact/"):
        """
        Generate a visualization of the model's architecture
        """
        
        # Only plot if we train on CPU 
        # Assuming that if you train on GPU (GPU cluster) it mean that 
        # you don't need to check your model's architecture
        device = tf.test.gpu_device_name()
        if "GPU" not in device:
            
            filename = os.path.join(save_path, self.name+".png")
            tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)



    def getVariables(self):
        """
        Return the variables of the model
        """
        return self.model.trainable_variables

    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables)):
            self.model.trainable_variables[i].assign(variables[i])
