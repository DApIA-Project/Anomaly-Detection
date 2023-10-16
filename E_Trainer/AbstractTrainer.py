


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from B_Model.AbstractModel import Model



class Trainer:
    """"
    /!\ Abstract Trainer class
    Manage the whole training of a model.

    Parameters :
    ------------

    CTX : dict
        The hyperparameters context
    
    model : type[Model]
        The model class of the model we want to train

    Methods :
    ---------

    run():
        Run the whole training pipeline
        and give metrics about the model's performance

    train(): Abstract
        Manage the training loop

    eval(): Abstract
        Evaluate the model and return metrics
    
    """

    def __init__(self, CTX:dict, model:"type[Model]"):
        pass

    def run(self):
        """
        Run the whole training pipeline
        and give metrics about the model's performance

        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        if (self.CTX["EPOCHS"] > 0):
            self.train()
        else:
            self.load()

        # return {} # leave early for testing
        return self.eval()


    def train(self):
        """
        Manage the training loop
        """
        raise NotImplementedError

        # train you'r model as you want here

    def load(self):
        """
        Load the model from a file
        """
        raise NotImplementedError

    def eval(self):
        """
        Evaluate the model and return metrics

        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        raise NotImplementedError

        # evaluate you're model as you want here

        # example of return metrics:
        return {
            "Accuracy": 0.5, 
            "False-Positive": 0.5, 
            "False-Negative": 0.5
        }


    
    






    





            
            
