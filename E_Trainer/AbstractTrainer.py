


import pandas as pd
from numpy_typing import np, ax
import matplotlib.pyplot as plt

from B_Model.AbstractModel import Model

import _Utils.Color as C
from _Utils.Color import prntC



class Trainer:
    """"
    Template Trainer class (use Inherit).
    Manage the whole training of a model.

    Attributes :
    ------------

    CTX : dict
        The hyperparameters context

    model : type[Model]

    Methods :
    ---------

    run():
        Run the whole training pipeline
        and return metrics about the model's performance

    train(): Abstract
        Manage the training loop

    eval(): Abstract
        Evaluate the model and return metrics

    """


    ###################################################
    # Initialization
    ###################################################

    def __init__(self, CTX:dict, model:"type[Model]"):

        pass

    def viz_model(self, filename:str):
        try:
            self.model.visualize(filename)
        except Exception as e:

            prntC(C.WARNING, "Visualization of the model failed")
            prntC(C.RED, e)
            prntC()

    ###################################################
    # Traier main loop
    ###################################################

    def run(self):
        """
        Run the whole training pipeline
        and return metrics about the model's performance

        Returns:
        --------

        metrics : dict
            The metrics dictionary representing model's performance
        """
        stats_t = None
        if (self.CTX["EPOCHS"] > 0):
            stats_t = self.train()
        else:
            self.load()

        stats = self.eval()
        if (stats_t is None):
            return stats
        
        for key, value in stats_t.items():
            if (key not in stats):
                stats[key] = value
        return stats


    ###################################################
    # Save and load model
    ###################################################

    def save(self):
        """
        Save the model's weights in the _Artifacts folder
        """
        raise NotImplementedError

    def load(self):
        """
        Load the model's weights from the _Artifacts folder
        """
        raise NotImplementedError



    def train(self) -> dict:
        """
        Manage the training loop.
        Testing is also done here.
        At the end, you can save your best model.
        """
        raise NotImplementedError

    ###################################################
    # Evaluation
    ###################################################

    def eval(self) -> dict:
        """
        Evaluate the model and return metrics

        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        raise NotImplementedError

        # example of return metrics:
        return {
            "Accuracy": 0.5,
            "False-Positive": 0.5,
            "False-Negative": 0.5
        }


















