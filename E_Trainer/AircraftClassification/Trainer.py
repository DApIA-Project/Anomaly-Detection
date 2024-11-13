# This file contains the Trainer class for the AircraftClassification problem


# |====================================================================================================================
# | IMPORTS
# |====================================================================================================================

from   _Utils.os_wrapper import os
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.AircraftClassification.DataLoader import DataLoader
import D_DataLoader.Utils as U
import D_DataLoader.AircraftClassification.Utils as SU
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


from numpy_typing import np, ax
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.FeatureGetter import FG_spoofing as FG
from   _Utils.DebugGui import GUI
from   _Utils.plotADSB import PLT
from   _Utils.Chrono import Chrono
import _Utils.Metrics as Metrics
from   _Utils.ProgressBar import ProgressBar
from   _Utils.save import write, load
from   _Utils.ADSB_Streamer import streamer


# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

# get problem name from parent folder for artifact saving
PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"

TRAIN_FOLDER = "./A_Dataset/AircraftClassification/Train/"
EVAL_FOLDER = "./A_Dataset/AircraftClassification/Eval/"

H_TRAIN_LOSS = 0
H_TEST_LOSS = 1
H_TRAIN_ACC = 2
H_TEST_ACC = 3


BAR = ProgressBar(max = 100)
CHRONO = Chrono()


# |====================================================================================================================
# | UTILITY FUNCTIONS
# |====================================================================================================================


def __alloc_pred_batches__(CTX:dict, train_batches:int, train_size:int,
                                     test_batches :int, test_size :int) ->"""tuple[
        np.float64_3d[ax.batch, ax.sample, ax.feature],
        np.float64_3d[ax.batch, ax.sample, ax.feature],
        np.float64_1d, np.float64_1d]""":

    return np.zeros((train_batches, train_size, CTX["LABELS_OUT"]), dtype=np.float64), \
           np.zeros((test_batches,  test_size,  CTX["LABELS_OUT"]), dtype=np.float64), \
           np.zeros(train_batches, dtype=np.float64), np.zeros(test_size, dtype=np.float64)


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
        FG.init(CTX)

        self.model:_Model_ = Model(CTX)
        if ("LIB" not in CTX):
            self.__makes_artifacts__()
            self.__init_GUI__()
            self.viz_model(self.ARTIFACTS)
            GUI.visualize("/Model/Achitecture", GUI.IMAGE, self.ARTIFACTS+f"/{self.model.name}.png")

        self.dl = DataLoader(CTX, TRAIN_FOLDER)

        # Private attributes
        self.__ep__ = -1
        self.__history__ = None
        self.__history_mov_avg__ = None
        # __eval_files__ is initialized only if eval is called (as there is no eval file in production)
        self.__eval_files__ = None


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


    def __init_GUI__(self) -> None:
        GUI.visualize("/Model", GUI.COLAPSIING_HEADER)
        GUI.visualize("/Training/TableTitle", GUI.TEXT, "Training curves", opened=True)
        GUI.visualize("/Training/Table", GUI.TABLE, 2, opened=True)
        GUI.visualize("/Training/Table/0/0/loss", GUI.TEXT, "Loading...")
        GUI.visualize("/Training/PredTitle", GUI.TEXT, "Example of prediction :")
        GUI.visualize("/Training/PredPlot", GUI.TEXT, "Loading...")


# |====================================================================================================================
# |     SAVE & LOAD MODEL'S VARIABLES
# |====================================================================================================================

    def save(self) -> None:
        CTX = self.CTX
        write(self.ARTIFACTS+"/w", self.model.get_variables())
        write(self.ARTIFACTS+"/xs", self.dl.xScaler.get_variables())

        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            write(self.ARTIFACTS+"/xts", self.dl.xTakeOffScaler.get_variables())
        if (CTX["ADD_AIRPORT_CONTEXT"]):
            write(self.ARTIFACTS+"/xas", self.dl.xAirportScaler.get_variables())

        write(self.ARTIFACTS+"/pad", self.dl.PAD)



    def load(self, path:str=None) -> None:
        if (path is None):
            path = self.ARTIFACTS

        self.model.set_variables(load(path+"/w"))
        self.dl.xScaler.set_variables(load(path+"/xs"))
        self.dl.yScaler.set_variables(self.CTX["USED_LABELS"])

        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.dl.xTakeOffScaler.set_variables(load(path+"/xts"))
        if (self.CTX["ADD_AIRPORT_CONTEXT"]):
            self.dl.xAirportScaler.set_variables(load(path+"/xas"))

        self.dl.PAD = load(path+"/pad")


# |====================================================================================================================
# |     TRAINING FUNCTIONS
# |====================================================================================================================

    def train(self) -> None:
        CTX = self.CTX

        for ep in range(1, CTX["EPOCHS"] + 1):

            # Allocate variables
            x_train, y_train = self.dl.get_train()
            x_test,  y_test  = self.dl.get_test()

            _y_train, _y_test, loss_train, loss_test = __alloc_pred_batches__(
                CTX, len(x_train), len(x_train[0][0]), len(x_test),  len(x_test[0][0]))


            CHRONO.start()
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
            self.__epoch_stats__(ep, y_train, _y_train, y_test, _y_test)

        self.__load_best_model__()


# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __prediction_statistics__(y:np.ndarray, y_:np.ndarray) -> "tuple[float, float]":
        acc  = Metrics.accuracy(y, y_)
        loss = Metrics.mse(y, y_)
        return acc, loss

    def __epoch_stats__(self, ep:int,
                        y_train:np.ndarray, _y_train:np.ndarray,
                        y_test :np.ndarray, _y_test :np.ndarray) -> None:


        train_acc, train_loss = Trainer.__prediction_statistics__(y_train, _y_train)
        test_acc,  test_loss  = Trainer.__prediction_statistics__(y_test,  _y_test )

        # On first epoch, initialize history
        if (self.__ep__ == -1 or self.__ep__ > ep):
            self.__history__         = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)
            self.__history_mov_avg__ = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)


        # Save epoch statistics
        self.__ep__ = ep
        self.__history__[:, ep-1] = [train_loss, test_loss, train_acc, test_acc]
        for i in range(4):
            self.__history_mov_avg__[i, ep-1] = Metrics.moving_average_at(self.__history__[i], ep-1, w=5)
        write(self.ARTIFACTS+"/weights/"+str(ep)+".w", self.model.get_variables())

        per_class_acc = Metrics.per_class_accuracy(y_test, _y_test)

        # Display statistics !
        self.__print_epoch_stats__(ep, train_loss, test_loss, train_acc, test_acc, per_class_acc)
        self.__plot_epoch_stats__()
        self.__plot_train_exemple__(y_train, _y_train)



    def __print_epoch_stats__(self, ep:int,
                              train_loss:float, test_loss:float,
                              train_acc:float, test_acc:float,
                              per_class_acc:np.ndarray) -> None:

        prntC(C.INFO,  "Epoch :", C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, CHRONO, "s")
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




    def __plot_epoch_stats__(self) -> None:

        # plot loss curves
        Metrics.plot_loss(self.__history__[H_TRAIN_LOSS], self.__history__[H_TEST_LOSS],
                         self.__history_mov_avg__[H_TRAIN_LOSS], self.__history_mov_avg__[H_TEST_LOSS],
                            type="loss", path=self.ARTIFACTS+"/loss.png")

        Metrics.plot_loss(self.__history__[H_TRAIN_ACC], self.__history__[H_TEST_ACC],
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

        self.model.set_variables(load(self.ARTIFACTS+"/weights/"+str(best_i)+".w"))
        self.save()


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================

    def predict(self, x:"list[dict[str,object]]") -> "tuple[np.ndarray, np.ndarray]":
        if (len(x) == 0): return np.ndarray((0,)), np.ndarray((0,))

        x_inputs, is_interesting = None, []

        for i in range(len(x)):
            sample, valid = self.dl.process_stream_of(x[i])
            is_interesting.append(valid)

            # Initialize batch size
            if (x_inputs is None):
                x_inputs = [np.zeros((len(x),) + sample[d].shape[1:], dtype=np.float64)
                                for d in range(len(sample))]

            for input in range(len(x_inputs)):
                x_inputs[input][i] = sample[input][0]

        # TODO check that batch size < MAX_BATCH_SIZE -> if not : split !
        # add if interesting flag
        i_loc = np.arange(0, len(is_interesting), dtype=int)[is_interesting]
        x_batch =  [x_inputs[d][i_loc] for d in range(len(x_inputs))]
        y_ = np.full((len(x_inputs[0]), self.CTX["LABELS_OUT"]), np.nan, dtype=np.float64)
        # exit(0)
        y_[i_loc] = self.model.predict(x_batch)
        y_agg = np.zeros((len(y_), self.CTX["LABELS_OUT"]), dtype=np.float64)
        for i in range(len(y_)):
            if not(np.isnan(y_[i]).all()):
                all_y_ = self.dl.prediction_cache.append(x[i]["icao24"], x[i]["tag"], y_[i])
                # use mean method for now
                y_agg[i] = np.nanmean(all_y_, axis=0)
            else:
                cache = self.dl.prediction_cache.get(x[i]["icao24"], x[i]["tag"])
                if (len(cache) > 0):
                    y_agg[i] = cache[-1]

        return y_, y_agg


# |====================================================================================================================
# |     EVALUATION
# |====================================================================================================================

    def __nb_batch__(self, nb_flights:int)->int:
        return (nb_flights-1) // self.CTX["MAX_BATCH_SIZE"] + 1


    def __batch_size__(self, ith_batch:int, NB_BATCH:int, nb_flights:int)->int:
        batch_size = nb_flights//NB_BATCH
        if (ith_batch == 0):
            return nb_flights - batch_size*(NB_BATCH-1)
        return batch_size


    def __gen_eval_batch__(self, files:"list[str]")->"""tuple[
            list[pd.DataFrame],
            int,
            np.float64_2d[ax.sample, ax.feature],
            list[np.ndarray]]""":

        # load all ressources needed for the batch
        files_df, max_lenght = [], 0
        y, y_ = [], []
        for f in range(len(files)):

            df = U.read_trajectory(files[f])
            df["tag"] = str(f)

            l = SU.getLabel(self.CTX, df["icao24"].iloc[0])
            if (l == 0):
                prntC(C.WARNING, "No label found for", files[f])


            files_df.append(df)

            y.append(l)
            y_.append(np.zeros((len(df), self.CTX["LABELS_OUT"]), dtype=np.float64))

            max_lenght = max(max_lenght, len(df))

        y = self.dl.yScaler.transform(y)
        return files_df, max_lenght, y, y_


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", y:np.ndarray, t:int)-> "tuple[list[dict[str:float]], list[str]]":
        x, files = [], []
        for f in range(len(dfs)):
            # only if there is a message at this time and if the aircraft has a label
            if (t < len(dfs[f]) and np.max(y[f]) > 0):
                msg = dfs[f].iloc[t].to_dict()
                x.append(msg)
                files.append(f)
        return x, files


    def eval(self)->dict:
        CTX = self.CTX
        if (self.__eval_files__ is None):
            self.__eval_files__ = U.list_flights(EVAL_FOLDER)[0:]

        # To avoid memory overflow, we limit the number of files loaded at once to MAX_BATCH_SIZE
        NB_BATCH = (len(self.__eval_files__)-1) // CTX["MAX_BATCH_SIZE"] + 1
        BAR.reset(max=len(self.__eval_files__))
        file_i = 0

        y = np.zeros((len(self.__eval_files__), self.CTX["LABELS_OUT"]), dtype=np.float64)
        y_ = [np.ndarray((0,)) for _ in range(len(self.__eval_files__))]

        for batch in range(NB_BATCH):
            BATCH_I = slice(file_i, file_i+CTX["MAX_BATCH_SIZE"], 1)
            BATCH_SIZE = self.__batch_size__(batch, NB_BATCH, len(self.__eval_files__))
            prntC(C.INFO, "Starting batch", batch, "with", BATCH_SIZE, "files")
            batch_files = self.__eval_files__[BATCH_I]
            dfs, max_len, y[BATCH_I], y_[BATCH_I] = self.__gen_eval_batch__(batch_files)


            for t in range(max_len):
                x, files = self.__next_msgs__(dfs, y[BATCH_I], t)
                for i in range(len(x)): streamer.add(x[i])
                yt_, _ = self.predict(x)
                for i in range(len(files)):
                    y_[BATCH_I][files[i]][t] = yt_[i]

                # Show progression :
                __batch_realized = (t+1) / max_len + batch
                BAR.update(__batch_realized / NB_BATCH * len(self.__eval_files__), f"remaining files: {len(files)}")

            file_i += BATCH_SIZE

        self.__eval_stats__(y, y_)


# |====================================================================================================================
# |     EVALUATION STATISTICS
# |====================================================================================================================


    def __compute_prediction__(self, y_:np.float64_2d[ax.time, ax.label])-> """tuple[
            np.float64_1d[ax.label],
            np.float64_1d[ax.label],
            np.float64_1d[ax.label],
            np.float64_1d[ax.label]]""":

        y_ = y_[~np.isnan(y_).any(axis=1)]

        confidence = Metrics.confidence(y_)
        agg_mean = np.mean(y_, axis=0)
        agg_max = y_[np.argmax(confidence)]
        agg_count = np.bincount(np.argmax(y_, axis=1), minlength=self.CTX["LABELS_OUT"])
        agg_count = agg_count / np.sum(agg_count)
        agg_nth_max = np.mean(y_[np.argsort(confidence)[-20:]], axis=0)
        return agg_mean, agg_max, agg_count, agg_nth_max


    def __eval_stats__(self, y:np.float64_2d[ax.sample, ax.label], y_:"list[np.float64_2d[ax.time, ax.label]]") -> None:
        OUT = np.ndarray((len(y), self.CTX["LABELS_OUT"]), dtype=np.float64)
        agg_mean, agg_max, agg_count, agg_nth_max = OUT.copy(), OUT.copy(), OUT.copy(), OUT.copy()

        for f in range(len(self.__eval_files__)):
            agg_mean[f], agg_max[f], agg_count[f], agg_nth_max[f] = self.__compute_prediction__(y_[f])

        methods = ["mean", "max", "count", "nth_max"]
        prediction_methods = [agg_mean, agg_max, agg_count, agg_nth_max]
        accuracy_per_method = [
            Metrics.accuracy(y, method) for method in prediction_methods
        ]
        best_method = np.argmax(accuracy_per_method)

        for i in range(len(methods)):
            prntC(C.INFO, "with", methods[i], " aggregation, accuracy : ", round(accuracy_per_method[i] * 100, 2))
        prntC(C.INFO, "Best method is :", methods[best_method])

        confusion_matrix = Metrics.confusion_matrix(y, prediction_methods[best_method])
        prntC(confusion_matrix)


        labels = [self.CTX["LABEL_NAMES"][i] for i in self.CTX["USED_LABELS"]]
        Metrics.plot_confusion_matrix(confusion_matrix,
                                    self.ARTIFACTS+"/confusion_matrix.png", labels)

        prntC(C.INFO, "Failed files : ")
        for f in range(len(self.__eval_files__)):
            true = np.argmax(y[f])
            pred = np.argmax(prediction_methods[best_method][f])
            if (true != pred):
                prntC(" -", C.CYAN, self.__eval_files__[f], C.RESET,
                      " - True label :", self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][true]],
                      " - Predicted :",  self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][pred]])

