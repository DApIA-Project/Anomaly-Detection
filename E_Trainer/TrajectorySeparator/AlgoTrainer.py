
from _Utils.save import write, load
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.ProgressBar import ProgressBar


from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer
from D_DataLoader.TrajectorySeparator.DataLoader import DataLoader


import os
import time
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages


PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]


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
        self.dl = DataLoader(CTX)


    def makes_artifacts(self):
        self.ARTIFACTS = "./_Artifacts/"+PBM_NAME
        if not os.path.exists("./_Artifacts"):
            os.makedirs("./_Artifacts")
        if not os.path.exists("./_Artifacts/"+PBM_NAME):
            os.makedirs("./_Artifacts/"+PBM_NAME)

    def train(self):
        """
        Train the model.
        """
        # no training needed
        pass


    def load(self):
        """
        Load the model's weights from the _Artifacts folder
        """
        pass


    def eval(self):
        """
        Evaluate the model and return metrics

        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        CTX = self.CTX
        FOLDER = "./A_Dataset/TrajectorySeparator/"
        folders = [f for f in os.listdir(FOLDER) if f != "base_files" and "." not in f]
        mean_acc = 0

        for folder in folders:
            prntC(C.INFO, "Evaluating folder : " + folder)
            x, y, df = self.dl.genEval(os.path.join(FOLDER, folder))
            loss, pred = self.model.compute_loss(x, y)
            acc = (1-loss)*100
            prntC(C.INFO, "Accuracy : ", round(acc, 2))
            mean_acc += acc

            # re-generate splited dataframes from predictions
            sub_df = {}
            for t in range(len(pred)):
                if (pred[t] not in sub_df):
                    sub_df[pred[t]] = pd.DataFrame(columns=df.columns)
                sub_df[pred[t]].loc[len(sub_df[pred[t]])] = df.loc[t]

            # save csv files
            if not os.path.exists(os.path.join(self.ARTIFACTS, folder)):
                os.makedirs(os.path.join(self.ARTIFACTS, folder))
            else: os.system("rm "+os.path.join(self.ARTIFACTS, folder)+"/*")

            for key in sub_df.keys():
                sub_df[key]["icao24"] = str(key)
                sub_df[key].to_csv(os.path.join(self.ARTIFACTS, folder, str(key) + ".csv"), index=False)
            prntC()

        return {"accuracy": mean_acc/len(folders)}

