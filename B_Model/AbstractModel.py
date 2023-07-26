

import numpy as np


class Model():
    """
    Abstract class representing a model

    /!\ Your model must always inherit from this class
    and implement the methods marked "MENDATORY" below

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
        (Optional : some models are too complex to be visualized)
    """

    name = "AbstractModel (TO OVERRIDE)"

    def __init__(self, CTX:dict):
        """ 
        Generate model architecture
        Define loss function
        Define optimizer
        """
        pass


    def predict(self, x:np.ndarray):
        """
        Make prediction for x 
        """
        raise NotImplementedError

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        raise NotImplementedError
        y_ = self.predict(x)
        return 0.0, y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        raise NotImplementedError
        loss, out = self.compute_loss(x, y)
        # compute and apply gradient
        return loss, out


    def visualize(self, save_path="./_Artefact/"):
        """
        Generate a visualization of the model's architecture
        """
        pass



    def getVariables(self)->np.ndarray:
        """
        Return the variables of the model
        """
        raise NotImplementedError
    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        raise NotImplementedError
