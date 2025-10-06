from catboost import CatBoostRegressor, train

from B_Model.AbstractModel import Model as AbstactModel

from _Utils.os_wrapper import os
from numpy_typing import np, ax


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



class Model(AbstactModel):

    name = "CatBoost"

    def __init__(self, CTX:dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        # load context
        self.CTX = CTX

        # save the number of training steps
        self.nb_train = 0

        self.model = []

        for i in range(CTX["FEATURES_OUT"]):
            self.model.append(CatBoostRegressor(iterations=1,
                                       learning_rate=CTX["LEARNING_RATE"],
                                       depth=CTX["LAYERS"],
                                       verbose=False))
        self.loss = mse

<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
    def predict(self, x):
        """
        Make prediction for x
        """
        x = x.reshape((x.shape[0], -1))
        y_ = np.zeros((x.shape[0], self.CTX["FEATURES_OUT"]))
        for i in range(self.CTX["FEATURES_OUT"]):
            y_[:, i] = self.model[i].predict(x)
        return y_

<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        """
        y_ = self.predict(x)
        return self.loss(y, y_), y_

<<<<<<< HEAD
=======
    @tf.function
>>>>>>> master
    def training_step(self, x, y):
        """
        Train the model for one step
        """
        x_ = x.reshape((x.shape[0], -1))
        for i in range(self.CTX["FEATURES_OUT"]):
            init_model = None
            if self.nb_train > 0:
                init_model = self.model[i]

            self.model[i].fit(x_, y[:, i], init_model=init_model)
        self.nb_train += 1


        y_ = self.predict(x)
        loss = self.loss(y, y_)

        return loss, y_

    def visualize(self, save_path="./_Artifacts/"):
        """
        Visualize the model
        """
        pass

    def get_variables(self):
        """
        Get the model variables
        """
        # save trained model
        for i in range(self.CTX["FEATURES_OUT"]):
            self.model[i].save_model("tmp"+str(i)+".json", format="json")
        variables = []
        for i in range(self.CTX["FEATURES_OUT"]):
            variables.append(open("tmp"+str(i)+".json", "r").read())
            os.remove("tmp"+str(i)+".json")
        print(variables)
        return variables


    def set_variables(self, variables):
        """
        Set the model variables
        """
        for i in range(self.CTX["FEATURES_OUT"]):
            with open("tmp"+str(i)+".json", "w") as f:
                f.write(variables[i])
            self.model[i] = CatBoostRegressor().load_model("tmp"+str(i)+".json", format="json")
            os.remove("tmp"+str(i)+".json")
        return variables
