from _Utils.os_wrapper import os
import pandas as pd

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

TRAIN_FOLDER = "./A_Dataset/AircraftClassification/Train/"
EVAL_FOLDER = "./A_Dataset/ReplaySolver/Eval"

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
        self.dl = DataLoader(CTX, TRAIN_FOLDER)

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

    def load(self, path=None) -> None:

        if (path is None):
            path = self.ARTIFACTS

        self.model.set_variables(load(path+"/w"))

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
                self.model.training_step(x_train[batch], y_train[batch])

            # Testing
            accuracy = 0.0
            for batch in range(len(x_test)):
                acc, _ = self.model.compute_loss(x_test[batch], y_test[batch])
                accuracy += acc
            accuracy /= len(x_test)

            self.__epoch_stats__(ep, accuracy)

            if(len(x_train) == 0):
                prntC(C.INFO, "Whole dataset has been used, training is over.")
                break

        self.save()

# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    def __epoch_stats__(self, ep:int, acc:float) -> None:
        prntC(C.INFO,
                "Epoch : ", C.BLUE, ep, C.RESET,
                " accuracy : ", C.BLUE, round(acc*100, 2), C.RESET,
                " time : ", C.BLUE, CHRONO, C.RESET,
                    flush=True)


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================


    def predict(self, x:"list[dict[str,object]]") -> "list[str]":
        if(len(x) == 0): return []

        x_batch = np.zeros((len(x), self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]), dtype=np.float64)
        is_interesting:"list[int]" = []
        y_:"list[list[str]]" = [["none"] for _ in range(len(x))]

        for i in range(len(x)):
            x_sample, valid = self.dl.streamer.stream(x[i])
            if (valid):
                x_batch[i] = x_sample
                is_interesting.append(i)

        # TODO check x_batch size < MAX_BATCH_SIZE
        y_is_interesting = self.model.predict(x_batch[is_interesting])
        for i in range(len(is_interesting)):
            y_[is_interesting[i]] = y_is_interesting[i]

        for i in range(len(x)):
            tag = x[i].get("tag", x[i]["icao24"])

            predictions = self.dl.streamer.get_cache(f"y_{tag}")
            if (predictions is None):
                predictions = []
            predictions.append(y_[i])
            self.dl.streamer.cache(f"y_{tag}", predictions)


            # find the most frequent prediction
            count = {}
            for t in range(len(predictions)):
                for p in predictions[t]:
                    count[p] = count.get(p, 0) + 1

            prediction = "unknown"
            if (len(count) > 0):
                best = max(count, key=count.get)
                if (count[best] > 30):
                    prediction = best
            y_[i] = prediction

        return y_


# |====================================================================================================================
# |    EVALUATION
# |====================================================================================================================

    def __gen_eval_batch__(self) -> "tuple[list[pd.DataFrame], int, list[str]]":

        files = U.list_flights(EVAL_FOLDER)
        max_len = 0
        dfs = []
        for f in range(len(files)):
            df = U.read_trajectory(files[f])
            dfs.append(df)
            max_len = max(max_len, len(df))
            files[f] = files[f].split("/")[-1]

        return dfs, max_len, files


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", t:int, files:"list[str]") -> "list[dict[str:float]]":
        x, y = [], []
        for f in range(len(dfs)):
            if (t < len(dfs[f])):
                msg = dfs[f].iloc[t].to_dict()
                x.append(msg)
                y.append(files[f])
        return x, y


    def eval(self) -> "dict[str, float]":

        dfs, max_len, files = self.__gen_eval_batch__()

        preds_for_files:"dict[str, list[str]]" = {}

        for t in range(max_len):
            x, y = self.__next_msgs__(dfs, t, files)
            matches = self.predict(x)

            for i in range(len(y)):
                preds = preds_for_files.get(y[i], [])
                preds.append(matches[i])
                preds_for_files[y[i]] = preds

        self.__eval_stats__(preds_for_files)

        return {}


    def __eval_stats__(self, preds_for_files:"dict[str, list[str]]") -> None:

        for f in preds_for_files:
            count = {}
            for p in preds_for_files[f]:
                count[p] = count.get(p, 0) + 1

            sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)

            sorted_count = [(x[0], round((x[1] / len(preds_for_files[f])) * 100.0, 1)) for x in sorted_count]

            prntC(C.INFO, "Predictions for ", C.BLUE, f, C.RESET)
            for i in range(len(sorted_count)):
                prntC("\t-", C.YELLOW, sorted_count[i][0], C.RESET, " : ",
                      C.BLUE, sorted_count[i][1], C.RESET, "%")
            prntC()




