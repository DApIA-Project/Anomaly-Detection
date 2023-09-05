
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
import numpy as np  

import os


class Model(AbstactModel):
    """
    Convolutional neural network model for 
    aircraft trajectories classification based on 
    trajectories. With the addition of an image context.
    And the possibility to add a take-off context.

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

    name = "CNN_img"

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

        n = CTX["LAYERS"]
    

        # build model's architecture
        feature_in = self.CTX["FEATURES_IN"] * (2 if self.CTX["ADD_TAKE_OFF_CONTEXT"] else 1)
        input_shape = (self.CTX["INPUT_LEN"], feature_in)
        x = tf.keras.Input(shape=input_shape, name='input')
        z = x


        # # split x in two parts of 11 features
        # zADSB = Lambda(lambda x: x[:, :, :11])(z)
        # if self.CTX["ADD_TAKE_OFF_CONTEXT"]:
        #     zContext = Lambda(lambda x: x[:, :, 11:])(z)

        # stem layer
        # z = zADSB
        Conv1DModule(64, 9, strides=4, padding="same")(z)

        for f in [64, 128, 128]:
            for _ in range(n):
                z = Conv1DModule(f, padding="same")(z)
            z = MaxPooling1D()(z)

        zADSB = Flatten()(z)


        # if self.CTX["ADD_TAKE_OFF_CONTEXT"]:

        #     z1 = zContext

        #     z = Conv1D(32, 1, padding="same")(z1)
        #     z = BatchNormalization()(z)
        #     z = Activation("relu")(z)

        #     Conv1DModule(64, 17, strides=4, padding="same")(z)

        #     for f in [64, 128, 128]:
        #         for _ in range(n):
        #             z = Conv1DModule(f, padding="same")(z)
        #         z = MaxPooling1D()(z)

        #     zContext = Flatten()(z)


        input_shape_img = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
        x_img = tf.keras.Input(shape=input_shape_img, name='input_img')
        z_img = x_img
        
        n=1
        for _ in range(n):
            z_img = Conv2DModule(64, 3, padding="same")(z_img)
        z_img = AveragePooling2D()(z_img)

        for _ in range(n):
            z_img = Conv2DModule(64, 3, padding="same")(z_img)
        z_img = AveragePooling2D()(z_img)

        for _ in range(n):
            z_img = Conv2DModule(16, 3, padding="same")(z_img)


        z_img = Flatten()(z_img)

        # if self.CTX["ADD_TAKE_OFF_CONTEXT"]:
        #     z = Concatenate()([zADSB, zContext, z_img])
        # else:
        z = Concatenate()([zADSB, z_img])

        z = DenseModule(256, dropout=self.dropout)(z)

        z = Dense(self.outs, activation="softmax")(z)

        y = z


        self.model = tf.keras.Model(inputs=[x, x_img], outputs=[y])


        # define loss function
        # self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.loss = tf.keras.losses.MeanSquaredError()

        # define optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])

        
    def predict(self, x, x_img):
        """
        Make prediction for x 
        """
        return self.model([x, x_img])

    def compute_loss(self, x, x_img, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.model([x, x_img])
        return self.loss(y_, y), y_

    def training_step(self, x, x_img, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss, output = self.compute_loss(x, x_img, y)

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
