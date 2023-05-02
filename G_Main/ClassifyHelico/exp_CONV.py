

# used method
from F_Runner.SimpleFit import *
from F_Runner.GeneticOpt import *

# used model LSM
from B_Model.ClassifyHelico.CONV import Model
import C_Constants.ClassifyHelico.CONV as CTX
from E_Trainer.ClassifyHelico import Trainer

import math

# get parent folder of the current file
import os

project_name = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
experiment_name = Model.name


def __main__():
    trainable_hyperparameters = {
            "LEARNING_RATE" : Mutation(function=mult, rate=math.sqrt(10)),
            "EPOCHS" : Mutation(function=mult, rate=2, min=1, check=lambda ctx : ctx["EPOCHS"] * ctx["NB_BATCH"] <= 102400),
            "BATCH_SIZE" : Mutation(function=mult, rate=2, min=1, max=256),
            "NB_BATCH" : Mutation(function=mult, rate=2, min=1, check=lambda ctx : ctx["EPOCHS"] * ctx["NB_BATCH"] <= 102400),
            "HISTORY" : Mutation(function=add, rate=1, min=1, max=24),
            "DROPOUT" : Mutation(function=add, rate=0.1, min=0, max=1),
        }

    # genetic_fit(
    #     Model, 
    #     Trainer, 
    #     CTX, 
    #     trainable_hyperparameters, 
    #     evaluated_on="Accuracy", # used metrics to select the best model (returned by eval() of the trainer)
    #     project_name=project_name)


    simple_fit(
        Model, 
        Trainer, 
        CTX, 
        project_name=project_name)

