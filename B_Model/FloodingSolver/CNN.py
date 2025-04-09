
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *


from _Utils.os_wrapper import os


class Model(AbstactModel):

    name = "CNN"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX
        self.outs = CTX["FEATURES_OUT"]

        # save the number of training steps
        self.nb_train = 0

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=x_input_shape, name='input')

        z = x
        n = self.CTX["LAYERS"]
        for i in range(n):
            z = Conv1DModule(CTX["UNITS"], 3, padding=CTX["MODEL_PADDING"],
                    batch_norm=not(CTX["USE_DYT"]), dyt=CTX["USE_DYT"])(z)

        z = Flatten()(z)
        z = Dense(self.outs, activation="linear")(z)
        z = Activation(CTX["ACTIVATION"])(z)
        y = z


        self.model = tf.keras.Model(x, y)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])



    def predict(self, x):
        return self.model(x)
    


    def compute_loss(self, x, y):
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
