# This file contains the Trainer class for the AircraftClassification problem


# |====================================================================================================================
# | IMPORTS
# |====================================================================================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.AircraftClassification.DataLoader import DataLoader
import D_DataLoader.Utils as U
import D_DataLoader.AircraftClassification.Utils as SU
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer

import _Utils.Metrics as Metrics
from _Utils.save import write, load
import _Utils.Color as C
from _Utils.Color import prntC
# TODO from _Utils.plotADSB import plotADSB improve output from training, and visualization of the prediction
from _Utils.ProgressBar import ProgressBar
from _Utils.DebugGui import GUI
import _Utils.plotADSB as PLT




# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

# get problem name from parent folder for artifact saving
PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"

EVAL_FOLDER = "./A_Dataset/AircraftClassification/Eval/"
EVAL_FILES = U.listFlight(EVAL_FOLDER)[0:64]

H_TRAIN_LOSS = 0
H_TEST_LOSS = 1
H_TRAIN_ACC = 2
H_TEST_ACC = 3


BAR = ProgressBar(max = len(EVAL_FILES))


# |====================================================================================================================
# | UTILITY FUNCTIONS
# |====================================================================================================================


def __alloc_pred_batches__(CTX:dict, train_batches:int, train_size:int,
                                     test_batches :int, test_size :int) \
        ->"tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":

    return np.zeros((train_batches, train_size, CTX["FEATURES_OUT"]), dtype=np.float32), \
           np.zeros((test_batches,  test_size,  CTX["FEATURES_OUT"]), dtype=np.float32), \
           np.zeros(train_batches, dtype=np.float32), np.zeros(test_size, dtype=np.float32)


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
        GUI.visualize("/Model/Achitecture", GUI.IMAGE, self.ARTIFACTS+f"/{self.model.name}.png")

        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")

        # Private attributes
        self.__ep__ = -1
        self.__history__ = None
        self.__history_mov_avg__ = None





    def __makes_artifacts__(self) -> None:
        self.ARTIFACTS = ARTIFACTS+PBM_NAME+self.model.name

        if not os.path.exists(ARTIFACTS):
            os.makedirs(ARTIFACTS)
        if not os.path.exists(ARTIFACTS+PBM_NAME):
            os.makedirs(ARTIFACTS+PBM_NAME)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)
        if not os.path.exists(self.ARTIFACTS+"/weights"):
            os.makedirs(self.ARTIFACTS+"/weights")


# |====================================================================================================================
# |     SAVE & LOAD MODEL'S VARIABLES
# |====================================================================================================================

    def save(self) -> None:
        CTX = self.CTX
        write(self.ARTIFACTS+"/w", self.model.getVariables())
        write(self.ARTIFACTS+"/xs", self.dl.xScaler.getVariables())

        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            write(self.ARTIFACTS+"/xts", self.dl.xTakeOffScaler.getVariables())
        if (CTX["ADD_AIRPORT_CONTEXT"]):
            write(self.ARTIFACTS+"/xas", self.dl.xAirportScaler.getVariables())

        write(self.ARTIFACTS+"/pad", self.dl.PAD)



    def load(self) -> None:
        self.model.setVariables(load(self.ARTIFACTS+"/w"))
        self.dl.xScaler.setVariables(load(self.ARTIFACTS+"/xs"))

        self.dl.yScaler.setVariables(self.CTX["USED_LABELS"])

        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.dl.xTakeOffScaler.setVariables(load(self.ARTIFACTS+"/xts"))
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            self.dl.xAirportScaler.setVariables(load(self.ARTIFACTS+"/xas"))

        self.dl.PAD = load(self.ARTIFACTS+"/pad")


# |====================================================================================================================
# |     TRAINING FUNCTIONS
# |====================================================================================================================

    def train(self) -> dict:
        CTX = self.CTX

        for ep in range(1, CTX["EPOCHS"] + 1):

            # Data allocation
            elapsed = time.time()
            x_train, y_train = self.dl.genEpochTrain()
            x_test, y_test = self.dl.genEpochTest()

            _y_train, _y_test, loss_train, loss_test = __alloc_pred_batches__(
                CTX, len(x_train), len(x_train[0][0]), len(x_test),  len(x_test[0][0]))

            BAR.reset(max=len(x_train) + len(x_test))


            # Training
            for batch in range(len(x_train)):
                loss_train[batch], _y_train[batch] = self.model.training_step(x_train[batch], y_train[batch])
                BAR.update(batch+1)
            _y_train = _y_train.reshape(-1, _y_train.shape[-1])
            y_train = y_train.reshape(-1, y_train.shape[-1])


            # Testing
            for batch in range(len(x_test)):
                loss_test[batch], _y_test[batch] = self.model.compute_loss(x_test[batch], y_test[batch])
                BAR.update(len(x_train)+batch+1)
            _y_test = _y_test.reshape(-1, _y_test.shape[-1])
            y_test = y_test.reshape(-1, y_test.shape[-1])


            # Statistics
            elapsed = time.time() - elapsed
            self.__epoch_stats__(ep, elapsed, y_train, _y_train, y_test, _y_test)

        self.__load_best_model__()


# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    def __epoch_stats__(self, ep:int, elapsed:float,
                        y_train:np.ndarray, _y_train:np.ndarray,
                        y_test :np.ndarray, _y_test :np.ndarray) -> None:

        train_acc, train_loss = Metrics.spoofing_training_statistics(y_train, _y_train)
        test_acc,  test_loss  = Metrics.spoofing_training_statistics(y_test, _y_test)


        # |--------------------------
        # | On first epoch
        if (self.__ep__ == -1 or self.__ep__ > ep):
            self.__history__         = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float32)
            self.__history_mov_avg__ = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float32)

        # |--------------------------
        # | On epoch change
        if (self.__ep__ != ep):
            self.__ep__ = ep
            self.__history__[:, ep-1] = [train_loss, test_loss, train_acc, test_acc]
            for i in range(4):
                self.__history_mov_avg__[i, ep-1] = Metrics.moving_average_at(self.__history__[i], ep-1, w=5)

            write(self.ARTIFACTS+"/weights/"+str(ep)+".w", self.model.getVariables())

            per_class_acc = Metrics.perClassAccuracy(y_test, _y_test)
            self.__print_epoch_stats__(ep, elapsed, train_loss, test_loss, train_acc, test_acc, per_class_acc)
            self.__plot_epoch_stats__()
            self.__plot_train_exemple__(y_train, _y_train)



    def __print_epoch_stats__(self, ep:int, elapsed:float,
                              train_loss:float, test_loss:float,
                              train_acc:float, test_acc:float,
                              per_class_acc:np.ndarray) -> None:

        prntC(C.INFO,  "Epoch :", C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, round(elapsed, 2), "s")
        prntC(C.INFO_, "Train Loss :", C.BLUE, round(train_loss, 4), C.RESET,
                     "- Test  Loss :", C.BLUE, round(test_loss,  4))
        prntC(C.INFO_, "Train Acc  :", C.BLUE, round(train_acc,  4), C.RESET,
                     "- Test  Acc  :", C.BLUE, round(test_acc,   4))

        prntC(C.INFO_, "Per class accuracy : ")
        for i in range(len(per_class_acc)):
            prntC(C.INFO_, "\t-",
                  self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][i]], ":",
                  C.BLUE, round(per_class_acc[i],2))
        prntC()



    def __init_GUI__(self) -> None:
        GUI.visualize("/Model", GUI.COLAPSIING_HEADER)
        GUI.visualize("/Training/TableTitle", GUI.TEXT, "Training curves", opened=True)
        GUI.visualize("/Training/Table", GUI.TABLE, 2, opened=True)
        GUI.visualize("/Training/Table/0/0/loss", GUI.TEXT, "Loading...")
        GUI.visualize("/Training/PredTitle", GUI.TEXT, "Example of prediction :")
        GUI.visualize("/Training/PredPlot", GUI.TEXT, "Loading...")



    def __plot_epoch_stats__(self) -> None:

        # plot loss curves
        Metrics.plotLoss(self.__history__[H_TRAIN_LOSS], self.__history__[H_TEST_LOSS],
                         self.__history_mov_avg__[H_TRAIN_LOSS], self.__history_mov_avg__[H_TEST_LOSS],
                            type="loss", path=self.ARTIFACTS+"/loss.png")

        Metrics.plotLoss(self.__history__[H_TRAIN_ACC], self.__history__[H_TEST_ACC],
                         self.__history_mov_avg__[H_TRAIN_ACC], self.__history_mov_avg__[H_TEST_ACC],
                            type="accuracy", path=self.ARTIFACTS+"/accuracy.png")

        GUI.visualize("/Training/Table/0/0/loss", GUI.IMAGE, self.ARTIFACTS+"/loss.png")
        GUI.visualize("/Training/Table/1/0/acc", GUI.IMAGE, self.ARTIFACTS+"/accuracy.png")



    def __plot_train_exemple__(self, y_train:np.ndarray, _y_train:np.ndarray)->None:

        NAME = "train_example"
        filename = PLT.get_data(NAME)["filename"]

        true_label = self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][np.argmax(y_train[0])]]
        pred_label = self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][np.argmax(_y_train[0])]]
        confidence = round(_y_train[0][np.argmax(_y_train[0])] * 100, 1)
        PLT.title(NAME, f"{filename} -> True : {true_label} - Pred : {pred_label} ({confidence}%)")
        PLT.show(NAME, self.ARTIFACTS+"/train_example.png")

        GUI.visualize("/Training/PredPlot", GUI.IMAGE, self.ARTIFACTS+"/train_example.png")



# |--------------------------------------------------------------------------------------------------------------------
# |     FIND AND LOAD BEST MODEL WHEN TRAINING IS DONE
# |--------------------------------------------------------------------------------------------------------------------

    def __load_best_model__(self) -> None:

        if (len(self.__history__[1]) == 0):
            prntC(C.WARNING, "No history of training has been saved")
            return

        best_i = np.argmax(self.__history_mov_avg__[H_TEST_ACC]) + 1

        prntC(C.INFO, "load best model, epoch : ",
              C.BLUE, best_i, C.RESET, " with Acc : ",
              C.BLUE, self.__history__[H_TEST_ACC][best_i-1])

        self.model.setVariables(load(self.ARTIFACTS+"/weights/"+str(best_i)+".w"))
        self.save()


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================

    def predict(self, x:"list[dict[str,object]]") -> np.ndarray:

        x_inputs, isInteresting = None, []

        for i in range(len(x)):
            sample, valid = self.dl.streamer.stream(x[i])
            isInteresting.append(valid)

            if (x_inputs is None):
                x_inputs = [np.zeros((len(x),) + sample[d].shape[1:], dtype=np.float32)
                                for d in range(len(sample))]

            for input in range(len(x_inputs)):
                x_inputs[input][i] = sample[input][0]

        # add if interesting flag
        i_loc = np.arange(0, len(isInteresting), dtype=int)[isInteresting]
        x_batch =  [x_inputs[d][i_loc] for d in range(len(x_inputs))]
        y_ = np.zeros((len(x_inputs[0]), self.CTX["FEATURES_OUT"]), dtype=np.float32)
        # exit(0)
        y_[i_loc] = self.model.predict(x_batch)
        return y_


# |====================================================================================================================
# |     EVALUATION
# |====================================================================================================================

    def __nb_batch__(self, nb_flights):
        return (nb_flights-1) // self.CTX["MAX_BATCH_SIZE"] + 1


    def __batch_size__(self, ith_batch, NB_BATCH, nb_flights):
        batch_size = nb_flights//NB_BATCH
        if (ith_batch == 0):
            return nb_flights - batch_size*(NB_BATCH-1)
        return batch_size


    def __gen_eval_batch__(self, files):

        # load all ressources needed for the batch
        files_df, max_lenght = [], 0
        y, y_ = [], []
        for f in range(len(files)):

            df = pd.read_csv(EVAL_FOLDER+files[f], dtype={'icao24': str})
            files_df.append(df)
            y.append(SU.getLabel(self.CTX, df))
            y_.append(np.zeros((len(df), self.CTX["FEATURES_OUT"]), dtype=np.float32))

            max_lenght = max(max_lenght, len(df))

        y = self.dl.yScaler.transform(y)
        return files_df, max_lenght, y, y_


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", t):
        x, files = [], []
        for f in range(len(dfs)):
            if (t < len(dfs[f])):
                x.append(dfs[f].iloc[t])
                files.append(f)
        return x, files


    def eval(self)->dict:
        CTX = self.CTX
        NB_BATCH = (len(EVAL_FILES)-1) // CTX["MAX_BATCH_SIZE"] + 1
        BAR.reset(max=len(EVAL_FILES))
        file_i = 0

        y = np.zeros((len(EVAL_FILES), self.CTX["FEATURES_OUT"]), dtype=np.float32)
        y_ = [np.ndarray((0,)) for _ in range(len(EVAL_FILES))]

        for batch in range(NB_BATCH):
            BATCH_I = slice(file_i, file_i+CTX["MAX_BATCH_SIZE"], 1)
            BATCH_SIZE = self.__batch_size__(batch, NB_BATCH, len(EVAL_FILES))
            prntC(C.INFO, "Starting batch", batch, "with", BATCH_SIZE, "files")
            batch_files = EVAL_FILES[BATCH_I]
            dfs, max_len, y[BATCH_I], y_[BATCH_I] = self.__gen_eval_batch__(batch_files)

            for t in range(max_len):
                x, files = self.__next_msgs__(dfs, t)
                yt_ = self.predict(x)
                for i in range(len(files)):
                    y_[BATCH_I][files[i]][t] = yt_[i]

                # Show progression :
                __batch_realized = (t+1) / max_len + batch
                BAR.update(__batch_realized / NB_BATCH * len(EVAL_FILES), f"remaining files: {len(files)}")

            file_i += BATCH_SIZE

        self.__eval_stats__(y, y_)


# |====================================================================================================================
# |     EVALUATION STATISTICS
# |====================================================================================================================


    def __eval_stats__(self, y:np.ndarray, y_:np.ndarray) -> None:
        OUT = np.ndarray((len(y), self.CTX["FEATURES_OUT"]), dtype=np.float32)
        agg_mean, agg_max, agg_count, agg_nth_max = OUT.copy(), OUT.copy(), OUT.copy(), OUT.copy()

        for f in range(len(EVAL_FILES)):
            confidence = Metrics.confidence(y_[f])
            agg_mean[f] = np.mean(y_[f], axis=0)
            agg_max[f] = y_[f][np.argmax(confidence)]
            agg_count[f] = np.bincount(np.argmax(y_[f], axis=1), minlength=self.CTX["FEATURES_OUT"])
            agg_count[f] = agg_count[f] / np.sum(agg_count[f])
            # 20 most confident prediction
            agg_nth_max[f] = np.mean(y_[f][np.argsort(confidence)[-20:]])

        methods = ["mean", "max", "count", "nth_max"]
        prediction_methods = [agg_mean, agg_max, agg_count, agg_nth_max]
        accuracy_per_method = [
            Metrics.accuracy(y, method) for method in prediction_methods
        ]
        best_method = np.argmax(accuracy_per_method)

        for i in range(len(methods)):
            prntC(C.INFO, "with", methods[i], " aggregation, accuracy : ", accuracy_per_method[i])
        prntC(C.INFO, "Best method is :", methods[best_method])

        confusion_matrix = Metrics.confusionMatrix(y, prediction_methods[best_method])
        prntC(confusion_matrix)


        L = [self.CTX["LABEL_NAMES"][i] for i in self.CTX["USED_LABELS"]]
        Metrics.plotConfusionMatrix(confusion_matrix,
                                    self.ARTIFACTS+"/confusion_matrix.png",
                                    L)

        prntC(C.INFO, "Failed files : ")
        for f in range(len(EVAL_FILES)):
            true = np.argmax(y[f])
            pred = np.argmax(prediction_methods[best_method][f])
            if (true != pred):
                prntC(" -", C.CYAN, EVAL_FILES[f], C.RESET,
                      " - True label :", self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][true]],
                      " - Predicted :",  self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][pred]])

