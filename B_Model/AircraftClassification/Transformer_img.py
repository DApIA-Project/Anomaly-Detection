
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

import os

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


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

    name = "Transformer_img"

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
    

        # build model's architecture
        input_shape = (self.CTX["TIMESTEPS"], self.CTX["FEATURES_IN"])
        x = tf.keras.Input(shape=input_shape)
        z = x
        for _ in range(CTX["LAYERS"]):
            z = transformer_encoder(z, self.CTX["HEAD_SIZE"], self.CTX["NUM_HEADS"],  self.CTX["FF_DIM"], self.CTX["DROPOUT"])

        z = GlobalAveragePooling1D(data_format="channels_first")(z)




        #################################""
        input_shape_img = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
        x_img = tf.keras.Input(shape=input_shape_img, name='input_img')
        z_img = x_img

        # stem layer
        z_img = Conv2D(32, 8, strides=(2, 2))(z_img)

        for _ in range(CTX["LAYERS"]):
            z_img = Conv2DModule(32, 3, padding="same")(z_img)
        z_img = MaxPooling2D()(z_img)

        for _ in range(CTX["LAYERS"]):
            z_img = Conv2DModule(64, 3, padding="same")(z_img)
        z_img = MaxPooling2D()(z_img)

        for _ in range(CTX["LAYERS"]):
            z_img = Conv2DModule(128, 3, padding="same")(z_img)
        z_img = Conv2DModule(32, 3, padding="same")(z_img)
        z_img = Flatten()(z_img)

        
        z = Concatenate()([z, z_img])
        z = DenseModule(1024, dropout=self.dropout)(z)
        
        for dim in range(CTX["LAYERS"]):
            z = Dense(self.CTX["FF_DIM"], activation="relu")(z)
            z = Dropout(self.CTX["DROPOUT"])(z)
        y = Dense(self.CTX["FEATURES_OUT"], activation="softmax")(z)


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
