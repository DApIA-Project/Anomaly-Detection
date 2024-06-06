import os
import time

from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.ReplaySolver.DataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


from   _Utils.DebugGui import GUI
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.numpy import np, ax
from   _Utils.save import write, load

# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"
EVAL_FOLDER = "./A_Dataset/ReplaySolver/"



# |====================================================================================================================
# | TRAINER CLASS
# |====================================================================================================================


class Trainer(AbstractTrainer):


# |====================================================================================================================
# |     INITIALIZATION
# |====================================================================================================================

    def __init__(self, CTX:dict, Model:"type[_Model_]") -> None:
        # Public attributes
        self.CTX = CTX
        self.model:_Model_ = Model(CTX)
        self.__makes_artifacts__()
        self.__init_GUI__()
        self.viz_model(self.ARTIFACTS)
        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")

        # Private attributes



    def __makes_artifacts__(self) -> None:
        self.ARTIFACTS = ARTIFACTS+PBM_NAME+"/"+self.model.name
        if not os.path.exists(ARTIFACTS):
            os.makedirs(ARTIFACTS)
        if not os.path.exists(ARTIFACTS+PBM_NAME):
            os.makedirs(ARTIFACTS+PBM_NAME)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)


    def __init_GUI__(self) -> None:
        GUI.visualize("/ReplaySolver", GUI.TEXT, "TODO")

# |====================================================================================================================
# |     SAVE & LOAD HASH DATABASE
# |====================================================================================================================


    def save(self) -> None:
        write(self.ARTIFACTS+"/w", self.model.get_variables())

    def load(self) -> None:
        self.model.set_variables(load(self.ARTIFACTS+"/w"))

# |====================================================================================================================
# |     TRAINING
# |====================================================================================================================

    def train(self) -> None:

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

