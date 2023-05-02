


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from B_Model.AbstractModel import Model



class Trainer:

    def __init__(self, CTX:dict, model:"type[Model]"):
        pass

    def run(self):
        history = self.train()
        perf = self.eval()
        return history, perf


    def train(self):
        raise NotImplementedError
        history = [[], []]

        # training as you want

        return history

    def eval(self):
        raise NotImplementedError

        # evaluation as you want

        return {
            "Accuracy": 0.5, 
            "False-Positive": 0.5, 
            "False-Negative": 0.5
        }


    
    






    





            
            
