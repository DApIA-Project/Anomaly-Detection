
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

from _Utils.numpy import np, ax

from _Utils.os_wrapper import os


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
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)

        x = tf.keras.Input(shape=x_input_shape, name='input')
        inputs = [x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)

        if (CTX["ADD_MAP_CONTEXT"]):
            map = tf.keras.Input(shape=map_input_shape, name='map')
            inputs.append(map)

        # concat takeoff and x
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            z = Concatenate(axis=2)([x, takeoff])
        else:
            z = x

        # stem layer

        n = self.CTX["LAYERS"]
        for i in range(n):
            z = resLSTM(256, 2, self.dropout, i < n - 1)(z)

        # z = Attention(heads=1)(z)
        # z = GlobalAveragePooling1D()(z)
        z = Flatten()(z)

        if (CTX["ADD_MAP_CONTEXT"]):
            y_map = map

            n=1
            for _ in range(n):
                y_map = Conv2DModule(64, 3, padding="same")(y_map)
            y_map = MaxPooling2D()(y_map)

            for _ in range(n):
                y_map = Conv2DModule(64, 3, padding="same")(y_map)
            y_map = MaxPooling2D()(y_map)

            for _ in range(n):
                y_map = Conv2DModule(16, 3, padding="same")(y_map)
            y_map = GlobalAveragePooling2D()(y_map)
            y_map = Flatten()(y_map)


        to_concat = [z]
        if (CTX["ADD_MAP_CONTEXT"]): to_concat.append(y_map)

        z = Concatenate()(to_concat)
        z = DenseModule(256, dropout=self.dropout)(z)
        z = Dense(self.outs, activation="softmax")(z)
        y = z


        self.model = tf.keras.Model(inputs, y)


        # define loss function
        # self.loss = tf.keras.losses.CategoricalCrossentropy()
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
