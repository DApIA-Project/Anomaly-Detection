
from _Utils.os_wrapper import os
import tensorflow as tf
import keras.layers  as layers

from numpy_typing import np, ax, ax

from B_Model.AbstractModel import Model as AbstactModel

import matplotlib.pyplot as plt


class Model(AbstactModel):
    name = "AE"

    def __init__(self, CTX:dict) -> None:
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX

        self.ae:"dict[int, AE]" = {}
        for l in self.CTX["USED_LABELS"]:
            self.ae[l] = AE(CTX)

        self.PREDICTED_FEATURES_INDEXS = np.array([self.CTX["FEATURE_MAP"][f] for f in self.CTX["PRED_FEATURES"]],
                                                    dtype=np.int32)

    def predict(self, x:np.float64_3d[ax.sample, ax.time, ax.feature]) -> np.float64_2d[ax.sample, ax.label]:
        """
        Make prediction for x
        """
        x = x[0] # only take the first input
        y = x[:, :, self.PREDICTED_FEATURES_INDEXS]
        res = {l: self.ae[l].compute_loss(x, y) for l in self.CTX["USED_LABELS"]}

        loss = np.zeros((len(x), len(self.CTX["USED_LABELS"])))
        for i, l in enumerate(self.CTX["USED_LABELS"]):
            loss[:, i] = res[l][0]

        probabilities = 1 - loss
        return probabilities




    def compute_loss(self, x, y):
        x = x[0] # only take the first input
        y = x[:, :, self.PREDICTED_FEATURES_INDEXS]
        res = {l: self.ae[l].compute_loss(x, y) for l in self.CTX["USED_LABELS"]}

        loss = np.zeros((len(x), len(self.CTX["USED_LABELS"])))
        for i, l in enumerate(self.CTX["USED_LABELS"]):
            loss[:, i] = res[l][0]

        mean_loss = np.mean(np.mean(loss, axis=1))
        probabilities = 1 - loss

        return mean_loss, probabilities

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        x = x[0] # only take the first input
        labels_i = np.argmax(y, axis=1)
        labels = [self.CTX["USED_LABELS"][i] for i in labels_i]

        batches = {l: [] for l in self.CTX["USED_LABELS"]}
        for i in range(len(labels)):
            l = labels[i]
            batches[l].append(x[i])

        res = {}
        for l in self.CTX["USED_LABELS"]:
            if len(batches[l]) > 0:
                b = np.array(batches[l])
                res[l] = self.ae[l].training_step(b, b[:, :, self.PREDICTED_FEATURES_INDEXS])

        loss = np.zeros((len(x), len(self.CTX["USED_LABELS"])))
        for i, l in enumerate(self.CTX["USED_LABELS"]):
            if l in res:
                loss[:, i] = res[l][0]
            else:
                loss[:, i] = 0
        # on all axis
        mean_loss = np.mean(np.mean(loss, axis=1))
        probabilities = 1 - loss
        return mean_loss, probabilities


    def visualize(self, filename="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(filename, self.name+".png")
        self.ae[self.CTX["USED_LABELS"][0]].visualize(filename)

    def get_variables(self):
        """
        Return the variables of the model
        """
        return {l: self.ae[l].get_variables() for l in self.CTX["USED_LABELS"]}

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        for l in self.CTX["USED_LABELS"]:
            self.ae[l].set_variables(variables[l])


class AE:

    def __init__(self, CTX:dict) -> None:
        self.CTX = CTX

        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])

        #Conv1D autoencoder

        x = tf.keras.Input(shape=x_input_shape, name='input')

        #Encoder
        z = x

        z = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(z)
        z = layers.MaxPooling1D(pool_size=2, padding='same')(z)
        z = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(z)
        z = layers.MaxPooling1D(pool_size=2, padding='same')(z)

        #Decoder
        z = layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same')(z)
        z = layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(z)

        z = layers.Conv1D(filters=self.CTX["FEATURES_OUT"], kernel_size=3, activation='sigmoid', padding='same')(z)

        y = z

        self.model = tf.keras.Model(inputs=x, outputs=y)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.nb_train = 0

    def predict(self, x):
        """
        Make prediction for x
        """
        return self.model(x)

    def compute_loss(self, x, y):
        y_ = self.model(x)
        loss = self.loss(y_, y)
        return loss, y_


    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """

        fig, ax = plt.subplot(1, 2, figsize=(10, 5))

        lat = x[self.CTX["FEATURE_MAP"]["latitude"]]
        lon = x[self.CTX["FEATURE_MAP"]["longitude"]]
        


        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.model(x)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_

    def visualize(self, filename):
        """
        Generate a visualization of the model's architecture
        """
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
        for i, v in enumerate(variables):
            self.model.trainable_variables[i].assign(v)
