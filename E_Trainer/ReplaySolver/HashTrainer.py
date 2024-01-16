import _Utils.mlflow as mlflow
from _Utils.save import write, load
import _Utils.Color as C
from _Utils.Color import prntC


from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.ReplaySolver.DataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


import os
import time
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages




class Trainer(AbstractTrainer):
    """"
    Manage model's training environment to solve trajectories replay.

    Attributes :
    ------------

    CTX : dict
        The hyperparameters context

    dl : DataLoader
        The data loader corresponding to the problem
        we want to solve

    model : Model
        The model instance we want to train   

    Methods :
    ---------

    run(): Inherited from AbstractTrainer

    train():
        Manage the training loop

    eval():
        Evaluate the model and return metrics
    """

    def __init__(self, CTX:dict, Model:"type[_Model_]"):
        super().__init__(CTX, Model)
        self.CTX = CTX

        self.model:_Model_ = Model(CTX)
        
        try:
            self.model.visualize()
        except:
            print("WARNING : visualization of the model failed")

        # create the data loader
        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")
        
        # If "_Artifactss/" folder doesn't exist, create it.
        if not os.path.exists("./_Artifacts"):
            os.makedirs("./_Artifacts")


    def train(self):
        """
        Train the model.
        """
        CTX = self.CTX

        # if _Artifacts/modelsW folder exists and is not empty, clear it
        if os.path.exists("./_Artifacts/modelsW"):
            if (len(os.listdir("./_Artifacts/modelsW")) > 0):
                os.system("rm ./_Artifacts/modelsW/*")
        else:
            os.makedirs("./_Artifacts/modelsW")

        for ep in range(1, CTX["EPOCHS"] + 1):
            ##############################
            #         Training           #
            ##############################
            start = time.time()
            x_inputs, y_batches = self.dl.genEpochTrain(CTX["NB_BATCH"], CTX["BATCH_SIZE"])

            
            for batch in range(len(x_inputs)):
                loss, output = self.model.training_step(x_inputs[batch], y_batches[batch])

            ##############################
            #          Testing           #
            ##############################
            x_inputs, test_y = self.dl.genEpochTest()
            acc, res = self.model.compute_loss(x_inputs, test_y)
            for i in range(len(res)):
                print("pred : ", res[i], " true : ", test_y[i])

            print("Epoch : ", ep, " acc : ", acc * 100.0, " time : ", time.time() - start, flush=True)

        if (CTX["EPOCHS"]):
            write("./_Artifacts/"+self.model.name+".w", self.model.getVariables())


    def load(self):
        """
        Load the model's weights from the _Artifacts folder
        """
        self.model.setVariables(load("./_Artifacts/"+self.model.name+".w"))


    def eval(self):
        """
        Evaluate the model and return metrics

        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        x_inputs, test_y, alterations = self.dl.genEval()
        acc, res = self.model.compute_loss(x_inputs, test_y)
        for i in range(len(res)):
            print(test_y[i], " with ", alterations[i], " pred : ", res[i])

        print("Eval",  "acc : ", acc * 100.0, flush=True)
        
        return {}

