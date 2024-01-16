


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from B_Model.AbstractModel import Model



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

    def __init__(self, CTX:dict, model:"type[Model]"):
        pass

    def run(self):
        """
        Run the whole training pipeline
        and return metrics about the model's performance

        Returns:
        --------

        metrics : dict
            The metrics dictionary representing model's performance
        """
        if (self.CTX["EPOCHS"] > 0):
            self.train()
        else:
            self.load()

        # return {} # leave early for testing
        return self.eval()


    def train(self):
        """
        Manage the training loop.        
        Testing is also done here.
        At the end, you can save your best model.
        """
        raise NotImplementedError

        # train you'r model as you want here

    def load(self):
        """
        Implement the loading of trained model.
        Used when EPOCHS = 0 to directly test the model.
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


    
    






    





            
            
