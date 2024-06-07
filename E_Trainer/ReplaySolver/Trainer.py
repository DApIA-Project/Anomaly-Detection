from _Utils.os_wrapper import os
import time

from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.ReplaySolver.DataLoader import DataLoader
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


from   _Utils.Chrono import Chrono
from   _Utils.DebugGui import GUI
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.numpy import np, ax
from   _Utils.save import write, load
from   _Utils.ProgressBar import ProgressBar

# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"
EVAL_FOLDER = "./A_Dataset/ReplaySolver/"

BAR = ProgressBar(max = 100)
CHRONO = Chrono()


# |====================================================================================================================
# | BEGIN OF TRAINER CLASS
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

            # Allocate batches
            x_train, y_train = self.dl.get_train()
            x_test,  y_test  = self.dl.get_test()

            # Training
            CHRONO.start()
            for batch in range(len(x_train)):
                acc, y_ = self.model.training_step(x_train[batch], y_train[batch])

            # Testing
            accuracy = 0.0
            for batch in range(len(x_test)):
                acc, y_ = self.model.compute_loss(x_test[batch], y_test[batch])
                accuracy += acc
            accuracy /= len(x_test)

            self.__epoch_stats__(ep, acc)

        self.save()

# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    def __epoch_stats__(self, ep:int, acc:float) -> None:
        prntC(C.INFO,
                "Epoch : ", C.BLUE, ep, C.RESET,
                " acc : ", C.BLUE, acc * 100.0, C.RESET,
                " time : ", C.BLUE, CHRONO, C.RESET,
                    flush=True)


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================


    def predict(self, x:"list[dict[str,object]]") -> np.bool_1d[ax.sample]:

        x_batch = np.zeros((len(x), self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), dtype=np.float64)

        for i in range(len(x)):
            x_batch[i] = self.dl.streamer.stream(x[i])

        y_ = self.model.predict(x_batch)
        return y_


# |====================================================================================================================
# |    EVALUATION
# |====================================================================================================================

    def __gen_eval_batch__(self) -> "tuple[list[pd.DataFrame], int]":

        files = os.listdir(EVAL_FOLDER)
        max_len = 0
        dfs = []
        for f in range(len(files)):
            df = U.read_trajectory(files[f])
            dfs.append(df)
            max_len = max(max_len, len(df))

        return dfs, max_len


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", t:int) -> "list[dict[str:float]]":
        x, files = [], []
        for f in range(len(dfs)):
            if (t < len(dfs[f])):
                msg = dfs[f].iloc[t].to_dict()
                x.append(msg)
                files.append(f)
        return x, files


    def eval(self) -> "dict[str, float]":

        dfs, max_len = self.__gen_eval_batch__()

        for t in range(max_len):
            x, files = self.__next_msgs__(dfs, t)
            matches = self.predict(x)


        return {}

