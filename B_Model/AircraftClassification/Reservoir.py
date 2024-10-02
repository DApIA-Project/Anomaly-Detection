
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from reservoirpy.nodes import Reservoir


from numpy_typing import np, ax, ax
from _Utils.os_wrapper import os

def __predict_reservoir__(CTX:dict, reservoir, x):
    y_ = np.zeros((len(x), CTX["INPUT_LEN"], reservoir.output_dim))

    for s in range(len(x)):
        # reset the reservoir state
        reservoir.reset()
        # give the time series to the reservoir
        for t in range(len(x[s])):
            y_[s, t] = reservoir(x[s][t].reshape(1, -1))

    return y_


class Model(AbstactModel):

    name = "Reservoir"

    def __init__(self, CTX:dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX

        self.reservoir = Reservoir(1000,
            lr=0.5, sr=0.9
        )
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.take_off_reservoir = Reservoir(1000,
                lr=0.5, sr=0.9
            )



        # prepare input shapes
        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["RESERVOIR_UNITS"])
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["RESERVOIR_UNITS"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)

        # generate layers
        x = tf.keras.Input(shape=x_input_shape, name='input')
        inputs = [x]
        outputs = []

        adsb_module_inputs = [x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)

            self.takeoff_module = TakeOffModule(self.CTX)
            takeoff_ctx = self.takeoff_module(takeoff)
            adsb_module_inputs.append(takeoff_ctx)
            self.TAKEOFF = len(outputs)


        if (CTX["ADD_MAP_CONTEXT"]):
            map = tf.keras.Input(shape=map_input_shape, name='map')
            inputs.append(map)

            self.map_module = MapModule(self.CTX)
            map_ctx = self.map_module(map)
            adsb_module_inputs.append(map_ctx)
            self.MAP = len(outputs)

        if (CTX["ADD_AIRPORT_CONTEXT"]):
            airport = tf.keras.Input(shape=(self.CTX["AIRPORT_CONTEXT_IN"],), name='airport')
            inputs.append(airport)

            self.airport_module = AirportModule(self.CTX)
            airport_ctx = self.airport_module(airport)
            adsb_module_inputs.append(airport_ctx)
            self.AIRPORT = len(outputs)


        self.ads_b_module = ADS_B_Module(self.CTX)
        proba = self.ads_b_module(adsb_module_inputs)
        outputs.insert(0, proba)


        # generate model
        self.model = tf.keras.Model(inputs, outputs)

        # define loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])

        self.nb_train = 0



    def predict(self, x):
        """
        Make prediction for x
        """
        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]): map = x.pop(0)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): airport = x.pop(0)

        adsb =__predict_reservoir__(self.CTX, self.reservoir, adsb)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff=__predict_reservoir__(self.CTX, self.take_off_reservoir, takeoff)

        x = [adsb]

        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): x.append(takeoff)
        if (self.CTX["ADD_MAP_CONTEXT"]): x.append(map)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): x.append(airport)

        return self.model(x)

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.predict(x)
        loss = self.loss(y_, y)
        return loss, y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            y_ = self.predict(x)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_



    def visualize(self, filename="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(filename, self.name+".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)



    def get_variables(self):
        """
        Return the variables of the model
        """
        reservoirs = [self.reservoir]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): reservoirs.append(self.take_off_reservoir)

        return self.model.trainable_variables, reservoirs

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        model_vars, reservoirs = variables
        for i in range(len(variables)):
            self.model.trainable_variables[i].assign(model_vars[i])

        self.reservoir = reservoirs[0]
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.take_off_reservoir = reservoirs[1]







class TakeOffModule(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]


        self.flatten = Flatten()
        self.dense = DenseModule(self.CTX["RESERVOIR_UNITS"], dropout=self.dropout)


    def __call__(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x

class MapModule(tf.Module):


    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = 1
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]


        convNN = []
        for _ in range(self.layers):
            convNN.append(Conv2DModule(16, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling2D())
        for _ in range(self.layers):
            convNN.append(Conv2DModule(32, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling2D())
        for _ in range(self.layers):
            convNN.append(Conv2DModule(64, 3, padding=self.CTX["MODEL_PADDING"]))

        # convNN.append(GlobalMaxPooling2D())

        convNN.append(Conv2D(32, (2, 2), (2, 2)))
        convNN.append(BatchNormalization())
        convNN.append(Flatten())
        convNN.append(DenseModule(256, dropout=self.dropout))

        self.convNN = convNN


    def __call__(self, x):
        for layer in self.convNN:
            x = layer(x)
        return x


class AirportModule(tf.Module):
    """ very simple only a (dense module 64) * nb layers"""

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]

        denseNN = []
        for _ in range(self.layers):
            denseNN.append(DenseModule(64, dropout=self.dropout))
        self.denseNN = denseNN

    def __call__(self, x):
        for layer in self.denseNN:
            x = layer(x)
        return x


class ADS_B_Module(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["LABELS_OUT"]

        self.flatten = Flatten()
        self.dense = DenseModule(self.CTX["RESERVOIR_UNITS"], dropout=self.dropout)


        self.cat = Concatenate()

        denseNN = []
        denseNN.append(Dense(self.outs, activation="linear", name="prediction"))

        self.denseNN = denseNN
        self.probability = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])

    def __call__(self, x):

        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x.pop(0)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            airport = x.pop(0)




        # preprocess
        x = adsb
        x = self.flatten(x)
        x = self.dense(x)

        # concat takeoff and map
        cat = [x]
        if (self.CTX["ADD_MAP_CONTEXT"]):
            cat.append(map)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            cat.append(takeoff)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            cat.append(airport)

        x = self.cat(cat)

        # get prediction
        for layer in self.denseNN:
            x = layer(x)
        x = self.probability(x)
        return x



# # Nathan Trouvain, Nicolas P. Rougier, Xavier Hinaut. Create Efficient and Complex Reservoir Computing Architectures with ReservoirPy. SAB 2022 - FROM ANIMALS TO ANIMATS 16: The 16th International Conference on the Simulation of Adaptive Behavior, Sep 2022, Cergy-Pontoise / Hybrid, France. ⟨hal-03761440⟩

# @article{reservoir,
#     title={Create Efficient and Complex Reservoir Computing Architectures with ReservoirPy},
#     author={Nathan Trouvain, Nicolas P. Rougier, Xavier Hinaut},
#     journal={SAB 2022 - FROM ANIMALS TO ANIMATS 16: The 16th International Conference on the Simulation of Adaptive Behavior},
#     month={Sep},
#     year={2022},
#     address={Cergy-Pontoise / Hybrid, France}
# }