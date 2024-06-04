import _Utils.Color as C
from _Utils.Color import prntC

from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.ReplaySolver.DataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer

import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# parent folder
PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]


###################################################
# Trainer class
###################################################

class Trainer(AbstractTrainer):

    ###################################################
    # Initialization
    ###################################################

    def __init__(self, CTX:dict, Model:"type[_Model_]"):
        super().__init__(CTX, Model)
        self.CTX = CTX
        self.model:_Model_ = Model(CTX)

        self.makes_artifacts()
        self.viz_model("./test.png")

        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")


    def makes_artifacts(self):
        self.ARTIFACTS = "./_Artifacts/"+PBM_NAME+"/"+self.model.name

        if not os.path.exists("./_Artifacts"):
            os.makedirs("./_Artifacts")
        if not os.path.exists("./_Artifacts/"+PBM_NAME):
            os.makedirs("./_Artifacts/"+PBM_NAME)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)


    ###################################################
    # Save and load model
    ###################################################

    def save(self):
        write(self.ARTIFACTS+"/w", self.model.getVariables())

    def load(self):
        self.model.set_variables(load(self.ARTIFACTS+"/w"))



    ###################################################
    # Training
    ###################################################

    def train(self):

        CTX = self.CTX

        for ep in range(1, CTX["EPOCHS"] + 1):
            # Training
            start = time.time()
            x_inputs, y_batches = self.dl.genEpochTrain()
            for batch in range(len(x_inputs)):
                loss, output = self.model.training_step(x_inputs[batch], y_batches[batch])


            # Testing
            x_inputs, y_batches = self.dl.genEpochTest()
            if (len(x_inputs) > 1):
                prntC(C.ERROR, "Batch size should be 1 !")
            for batch in range(len(x_inputs)):
                acc, res = self.model.compute_loss(x_inputs[batch], y_batches[batch])



            for i in range(len(res)):
                prntC(C.INFO, "pred : ", C.BLUE, res[i], C.RESET, " true : ", C.BLUE, y_batches[0][i])
            prntC(C.INFO, "Epoch : ", C.BLUE, ep, C.RESET, " acc : ", C.BLUE, acc * 100.0, C.RESET, " time : ", C.BLUE, time.time() - start, C.RESET, 's', flush=True)

        if (CTX["EPOCHS"]):
            self.save()


    ###################################################
    # Evaluation
    ###################################################



    def eval(self):

        x_inputs, y_batches, alterations = self.dl.genEval()
        if (len(x_inputs) > 1):
                prntC(C.ERROR, "Batch size should be 1 !")

        for batch in range(len(x_inputs)):
            acc, res = self.model.compute_loss(x_inputs[batch], y_batches[batch])

        for i in range(len(res)):
            print(y_batches[0][i], " with ", alterations[0][i], " pred : ", res[i])

        print("Eval",  "acc : ", acc * 100.0, flush=True)

        return {}

