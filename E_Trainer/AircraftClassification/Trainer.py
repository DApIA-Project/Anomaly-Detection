# This file contains the Trainer class for the AircraftClassification problem


# |====================================================================================================================
# | IMPORTS
# |====================================================================================================================

from   _Utils.os_wrapper import os
import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< HEAD
from   matplotlib.backends.backend_pdf import PdfPages
=======
from sklearn import metrics
>>>>>>> master

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
<<<<<<< HEAD
=======
from   _Utils.Metrics import accuracy
>>>>>>> master


# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

# get problem name from parent folder for artifact saving
PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"

<<<<<<< HEAD
TRAIN_FOLDER = "./A_Dataset/AircraftClassification/Train/"
EVAL_FOLDER = "./A_Dataset/AircraftClassification/Eval/"
=======
TRAIN_FOLDER = "./A_Dataset/V1/Train/"
EVAL_FOLDER = "./A_Dataset/V1/Eval/"
>>>>>>> master

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
        
        print(self.model.get_variables())



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
        if ("START_FROM_MODEL" in self.CTX):
            self.model.set_variables(load(self.ARTIFACTS+"/weights/"+str(self.CTX["START_FROM_MODEL"])+".w"))
        else:
            # clean previous weights
            os.system("rm "+self.ARTIFACTS+"/weights/*")
            
        CTX = self.CTX
        acc_per_class = [0] * CTX["LABELS_OUT"]

        for ep in range(1, CTX["EPOCHS"] + 1):

            # Allocate variables
            x_train, y_train = self.dl.get_train(acc_per_class)
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
            acc_per_class = self.__epoch_stats__(ep, y_train, _y_train, y_test, _y_test)

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

        acc_per_class = Metrics.accuracy_per_class(y_test, _y_test)

        # Display statistics !
        self.__print_epoch_stats__(ep, train_loss, test_loss, train_acc, test_acc, acc_per_class)
        self.__plot_epoch_stats__()
        self.__plot_train_exemple__(y_train, _y_train)
        return acc_per_class



    def __print_epoch_stats__(self, ep:int,
                              train_loss:float, test_loss:float,
                              train_acc:float, test_acc:float,
                              acc_per_class:np.ndarray) -> None:

        prntC(C.INFO,  "Epoch :", C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, CHRONO, "s")
        prntC(C.INFO_, "Train Loss :", C.BLUE, round(train_loss, 4), C.RESET,
                     "- Test  Loss :", C.BLUE, round(test_loss,  4))
        prntC(C.INFO_, "Train Acc  :", C.BLUE, round(train_acc,  4), C.RESET,
                     "- Test  Acc  :", C.BLUE, round(test_acc,   4))

        prntC(C.INFO_, "Per class accuracy : ")
        for i in range(len(acc_per_class)):
            prntC(C.INFO_, "\t-",
                  self.CTX["LABEL_NAMES"][self.CTX["USED_LABELS"][i]], ":",
                  C.BLUE, round(acc_per_class[i],2))
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

    def predict(self, x_:"list[dict[str,object]]") -> "tuple[np.ndarray, np.ndarray]":
        if (len(x_) == 0): return np.ndarray((0,)), np.ndarray((0,))

        x_inputs, is_interesting = None, []

        for i in range(len(x_)):
            sample, valid = self.dl.process_stream_of(x_[i])
            is_interesting.append(valid)

            # Initialize batch size
            if (x_inputs is None):
                x_inputs = [np.zeros((len(x_),) + sample[d].shape[1:], dtype=np.float64)
                                for d in range(len(sample))]

            for input in range(len(x_inputs)):
                x_inputs[input][i] = sample[input][0]

        y_ = np.full((len(x_inputs[0]), self.CTX["LABELS_OUT"]), np.nan, dtype=np.float64)
        y_agg = np.zeros((len(y_), self.CTX["LABELS_OUT"]), dtype=np.float64)
        
        i_loc_ = np.arange(0, len(is_interesting), dtype=int)[is_interesting]    
        
        for s in range(0, len(i_loc_), self.CTX["MAX_BATCH_SIZE"]):
            start = s
            end = min(s+self.CTX["MAX_BATCH_SIZE"], len(i_loc_))
            pad = max(self.CTX["MIN_BATCH_SIZE"] - (end-start), 0)
            
            i_loc = i_loc_[start:end]

            x_batch = []
            for d in range(len(x_inputs)):
                batch = x_inputs[d][i_loc]
                batch = np.concatenate([batch, np.empty((pad,) + batch.shape[1:])])
                x_batch.append(batch)
            
            y_batch_ = self.model.predict(x_batch)
            
            if (pad > 0):
                y_batch_ = y_batch_[:-pad]
            y_[i_loc] = y_batch_
                
                
        for i in range(len(y_)):
            if not(np.isnan(y_[i]).all()):
                all_y_ = self.dl.prediction_cache.append(x_[i]["icao24"], x_[i]["tag"], y_[i])
            else:
                all_y_ = self.dl.prediction_cache.get(x_[i]["icao24"], x_[i]["tag"])
<<<<<<< HEAD
            # use nth max method
            l = min(len(all_y_), 20)
=======
            if (all_y_ is None):
                continue
            
            # use nth max method
            l = min(len(all_y_), self.CTX["LOSS_MOVING_AVERAGE"])
>>>>>>> master
            if (l > 0):
                confidence = Metrics.confidence(all_y_)
                # TODO performance improvment with already sorted list
                y_agg[i] = np.nanmean(all_y_[np.argsort(confidence)[-l:]], axis=0)

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


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", y:np.ndarray, t:int)-> "tuple[list[dict[str:float]], list[int]]":
        x, files = [], []
        for f in range(len(dfs)):
            # only if there is a message at this time and if the aircraft has a label
            if (t < len(dfs[f]) and np.max(y[f]) > 0):
                msg = dfs[f].iloc[t].to_dict()
                x.append(msg)
                files.append(f)
        return x, files


    def eval(self)->dict:
        if (self.__eval_files__ is None):
            self.__eval_files__ = U.list_flights(EVAL_FOLDER)[0:]
            # self.__eval_files__ = ["./A_Dataset/AircraftClassification/Eval/2022-01-01_15-15-33_FJDGY_3a2cbc.csv"]
            
        print(self.__eval_files__)

        BAR.reset(max=len(self.__eval_files__))

        dfs, max_len, y, y_ = self.__gen_eval_batch__(self.__eval_files__)


        CHRONO.start()
        NB_MESSAGE = 0
        for t in range(max_len):
            x, files = self.__next_msgs__(dfs, y, t)
            NB_MESSAGE += len(x)

            for i in range(len(x)): streamer.add(x[i])
            yt_, _ = self.predict(x)
            for i in range(len(files)):
                y_[files[i]][t] = yt_[i]

            # Show progression :
            BAR.update((t+1) / max_len * len(self.__eval_files__), f"remaining files: {len(files)}")
        
        CHRONO.stop()
        
        print("TOTAL NUM OF MESSAGES : ", NB_MESSAGE)

<<<<<<< HEAD
        acc = self.__eval_stats__(y, y_)
        return {"ACCURACY": round(acc*100, 2), "TIME": round(CHRONO.get_time_s(),1)}
=======
        acc, _, precision, recall, f1, roc_auc = self.__eval_stats__(y, y_)
        return {"ACCURACY": round(acc*100, 2), "TIME": round(CHRONO.get_time_s(),1), 
                "PRECISION": round(precision*100,2), "RECALL": round(recall*100,2),
                "F1_SCORE": round(f1*100,2), "ROC_AUC": round(roc_auc*100,2)}
>>>>>>> master


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
<<<<<<< HEAD

        for f in range(len(self.__eval_files__)):
            agg_mean[f], agg_max[f], agg_count[f], agg_nth_max[f] = self.__compute_prediction__(y_[f])
=======
        
        per_message_correct = 0
        total_messages = 0
        

        for f in range(len(self.__eval_files__)):
            per_message_correct += np.sum(np.argmax(y[f]) == np.argmax(y_[f], axis=1))
            total_messages += len(y_[f])
            
            agg_mean[f], agg_max[f], agg_count[f], agg_nth_max[f] = self.__compute_prediction__(y_[f])
        acc_per_messages = per_message_correct / total_messages
>>>>>>> master

        methods = ["mean", "max", "count", "nth_max"]
        prediction_methods = [agg_mean, agg_max, agg_count, agg_nth_max]
        accuracy_per_method = [
            Metrics.accuracy(y, method) for method in prediction_methods
        ]
        best_method = np.argmax(accuracy_per_method)

        for i in range(len(methods)):
            prntC(C.INFO, "with", methods[i], " aggregation, accuracy : ", round(accuracy_per_method[i] * 100, 2))
        prntC(C.INFO, "Best method is :", methods[best_method])
<<<<<<< HEAD

        confusion_matrix = Metrics.confusion_matrix(y, prediction_methods[best_method])
        prntC(confusion_matrix)

=======
        
        
        trues_label = np.argmax(y, axis=1)
        preds_label = np.argmax(prediction_methods[best_method], axis=1)
        
        
        confusion_matrix = Metrics.confusion_matrix(y, prediction_methods[best_method])
        
        accuracy_score = metrics.accuracy_score(trues_label, preds_label)
        precision = metrics.precision_score(trues_label, preds_label, average='macro')
        recall = metrics.recall_score(trues_label, preds_label, average='macro')
        f1 = metrics.f1_score(trues_label, preds_label, average='macro')
        roc_auc = metrics.roc_auc_score(y, prediction_methods[best_method], average='macro', multi_class='ovo')
        
        
        print(accuracy_per_method[best_method], accuracy_score)
        print(precision, recall, f1, roc_auc)
>>>>>>> master

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
        
<<<<<<< HEAD
        return accuracy_per_method[best_method]
=======
        return accuracy_per_method[best_method], acc_per_messages, precision, recall, f1, roc_auc
>>>>>>> master

