from _Utils.os_wrapper import os
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics


from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.InterpolationDetector.DataLoader import DataLoader
import D_DataLoader.InterpolationDetector.Utils as SU
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


from numpy_typing import np, ax
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.FeatureGetter import FG_interp as FG
from   _Utils.Chrono import Chrono
from   _Utils.DebugGui import GUI
import _Utils.geographic_maths as GEO
import _Utils.Metrics as Metrics
from   _Utils.plotADSB import PLT
from   _Utils.ProgressBar import ProgressBar
from   _Utils.save import write, load
from   _Utils.ADSB_Streamer import streamer

# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================


# get problem name from parent folder for artifact saving
PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]+"/"
ARTIFACTS = "./_Artifacts/"

TRAIN_FOLDER = ["./A_Dataset/V1/Train/", "./A_Dataset/InterpolationDetector/Train/interp_0/"]
EVAL_FOLDER = "./A_Dataset/InterpolationDetector/Eval/"

H_TRAIN_LOSS = 0
H_TEST_LOSS  = 1
H_TRAIN_ACC = 2
H_TEST_ACC  = 3

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

    return np.zeros((train_batches, train_size, 1), dtype=np.float64), \
           np.zeros((test_batches,  test_size,  1), dtype=np.float64), \
           np.zeros(train_batches, dtype=np.float64), np.zeros(test_size, dtype=np.float64)


# |====================================================================================================================
# | BEGIN OF TRAINER CLASS
# |====================================================================================================================


class Trainer(AbstractTrainer):

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



# |====================================================================================================================
# |     SAVE & LOAD MODEL'S VARIABLES
# |====================================================================================================================

    def save(self) -> None:
        write(self.ARTIFACTS+"/w", self.model.get_variables())
        write(self.ARTIFACTS+"/xs", self.dl.xScaler.get_variables())
        write(self.ARTIFACTS+"/pad", self.dl.PAD)


    def load(self, path:str=None) -> None:
        if (path is None):
            path = self.ARTIFACTS

        if ("LOAD_EP" in self.CTX and self.CTX["LOAD_EP"] != -1):
            self.model.set_variables(load(path+"/weights/"+str(self.CTX["LOAD_EP"])+".w"))
        else:
            self.model.set_variables(load(path+"/w"))
        self.dl.xScaler.set_variables(load(path+"/xs"))
        self.dl.PAD = load(path+"/pad")


# |====================================================================================================================
# |     TRAINING FUNCTIONS
# |====================================================================================================================

    def train(self) -> None:
        CTX = self.CTX
        prntC(C.INFO, "Training model : ", C.BLUE, self.model.name,
              C.RESET, " for ", C.BLUE, CTX["EPOCHS"], C.RESET, " epochs")

        for ep in range(1, CTX["EPOCHS"] + 1):

            # Allocate batches
            x_train, y_train = self.dl.get_train()
            x_test,  y_test  = self.dl.get_test()

            _y_train, _y_test, loss_train, loss_test = __alloc_pred_batches__(
                CTX, len(x_train), len(x_train[0]), len(x_test),  len(x_test[0]))

            CHRONO.start()
            BAR.reset(max=len(x_train) + len(x_test))

            # Training
            for batch in range(len(x_train)):
                loss_train[batch], _y_train[batch] = self.model.training_step(x_train[batch], y_train[batch])
                BAR.update()
            _y_train:np.float64_2d[ax.sample, ax.feature] = _y_train.reshape(-1, _y_train.shape[-1])
            y_train :np.float64_2d[ax.sample, ax.feature] =  y_train.reshape(-1,  y_train.shape[-1])

            # Testing
            for batch in range(len(x_test)):
                loss_test[batch], _y_test[batch] = self.model.compute_loss(x_test[batch], y_test[batch])
                BAR.update()
            _y_test:np.float64_2d[ax.sample, ax.feature] = _y_test.reshape(-1, _y_test.shape[-1])
            y_test :np.float64_2d[ax.sample, ax.feature] =  y_test.reshape(-1,  y_test.shape[-1])

            self.__epoch_stats__(ep, y_train, _y_train, y_test, _y_test)

        self.__load_best_model__()

# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    def __prediction_statistics__(self,
                                  y:np.float64_2d[ax.sample, ax.feature],
                                  y_:np.float64_2d[ax.sample, ax.feature])\
            -> "tuple[float, float]":

        accuracy = Metrics.binary_accuracy(y, y_)
        loss = Metrics.mse(y, y_)
        return accuracy[0], loss


    def __epoch_stats__(self, ep:int,
                        y_train:np.float64_2d[ax.sample, ax.feature],
                        _y_train:np.float64_2d[ax.sample, ax.feature],
                        y_test :np.float64_2d[ax.sample, ax.feature],
                        _y_test :np.float64_2d[ax.sample, ax.feature]) -> None:

        train_acc, train_loss = self.__prediction_statistics__(y_train, _y_train)
        test_acc,  test_loss  = self.__prediction_statistics__(y_test,  _y_test )

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

        # Display statistics !
        self.__print_epoch_stats__(ep, train_loss, test_loss, train_acc, test_acc)
        self.__plot_epoch_stats__()


    def __print_epoch_stats__(self, ep:int,
                              train_loss:float, test_loss:float,
                              train_acc:float, test_acc:float) -> None:


        prntC(C.INFO,  "Epoch :", C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, CHRONO, "s")
        prntC(C.INFO_, "Train Loss :", C.BLUE, round(train_loss, 4), C.RESET,
                     "- Test  Loss :", C.BLUE, round(test_loss,  4))
        prntC(C.INFO_, "Train Accuracy :", C.BLUE, round(train_acc, 4), C.RESET,
                     "- Test  Accuracy :", C.BLUE, round(test_acc,  4))
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
        







    def __load_best_model__(self) -> None:

        if (len(self.__history__[1]) == 0):
            prntC(C.WARNING, "No history of training has been saved")
            return

        best_i = np.argmax(self.__history_mov_avg__[H_TEST_ACC]) + 1

        prntC(C.INFO, "load best model, epoch : ",
              C.BLUE, best_i, C.RESET, " with accuracy : ",
              C.BLUE, round(self.__history__[H_TEST_ACC][best_i-1]*100.0, 2),"%")

        self.model.set_variables(load(self.ARTIFACTS+"/weights/"+str(best_i)+".w"))
        self.save()


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================


    def predict(self, x:"list[dict[str,object]]") -> """tuple[
            np.float64_2d[ax.sample, ax.feature],
            np.float64_1d[ax.sample],
            np.bool_1d[ax.sample]]""":
                
        if (len(x) == 0): return np.zeros((0, 1)), np.zeros((0,)), np.zeros((0, ))

        # allocate memory
        x_batch  = np.zeros((len(x), self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]))
        y_batch_ = np.full((len(x), 1), np.nan, dtype=np.float64)
        is_interesting = np.zeros(len(x), dtype=bool)
        is_interp = np.zeros(len(x), dtype=bool)

        # stream message and build input batch
        for i in range(len(x)):
            x_sample, valid = self.dl.process_stream_of(x[i])
            x_batch[i] = x_sample[0]
            is_interesting[i] = valid



        # predict only on interesting samples
        x_preds = x_batch[is_interesting]
        y_preds_ = np.zeros((len(x_preds), 1), dtype=np.float64)
        for s in range(0, len(x_preds), self.CTX["MAX_BATCH_SIZE"]):
            start = s
            end = min(s + self.CTX["MAX_BATCH_SIZE"], len(x_preds))
            pad = max(0, self.CTX["MIN_BATCH_SIZE"] - (end - start))
            
            x_batch_ =  np.concatenate([x_preds[start:end], np.empty((pad, ) + x_preds.shape[1:])], axis=0)
            output = self.model.predict(x_batch_)
            if (pad > 0):
                output = output[0:-pad]
            y_preds_[start:end] = output
            
        y_batch_[is_interesting] = y_preds_



        # compute loss
        y_mean = np.full(len(x), np.nan, dtype=np.float64)
        for i in range(len(x)):
            if (is_interesting[i]):
                loss_win = self.dl.preds_cache.append(x[i]["icao24"], x[i]["tag"], y_batch_[i])
            else:
                loss_win = self.dl.preds_cache.append(x[i]["icao24"], x[i]["tag"], 0)

            # prntC(C.DEBUG, loss_win[-1])
            
            if (len(loss_win) > self.CTX["LOSS_MOVING_AVERAGE"]):
                y_mean[i] = np.mean(loss_win[-self.CTX["LOSS_MOVING_AVERAGE"]:])
                is_interp[i] = y_mean[i] > self.CTX["THRESHOLD"]


        return y_batch_, y_mean, is_interp



# |====================================================================================================================
# |     EVALUATION
# |====================================================================================================================

    def __gen_eval_batch__(self, files:"list[str]")->"""tuple[
            list[pd.DataFrame], int,
            list[bool],
            list[np.float64_2d[ax.time, ax.feature]],
            list[np.float64_1d[ax.time]],
            list[int]]""":

        # load all ressources needed for the batch
        files_df, max_lenght, y, y_, loss, acc, glob_pred = [], 0, [], [], [], [], []
        for f in range(len(files)):
            df = U.read_trajectory(files[f])
            files_df.append(df)
            y.append("interp" in files[f])
            y_.append(np.full((len(df), 1), np.nan, dtype=np.float64))
            loss.append(np.full(len(df), np.nan, dtype=np.float64))

            max_lenght = max(max_lenght, len(df))
            acc.append(0)
            glob_pred.append(0)

        return files_df, max_lenght, y, y_, loss, acc, glob_pred


    def __next_msgs__(self, dfs:"list[pd.DataFrame]", t:int)-> "tuple[list[dict[str:float]], list[int]]":
        x, files = [], []
        for f in range(len(dfs)):
            # if there is a message at this time
            if (t < len(dfs[f])):
                msg = dfs[f].iloc[t].to_dict()
                x.append(msg)
                files.append(f)
        return x, files


    def eval(self) -> "dict[str, float]":

        if (self.__eval_files__ is None):
            self.__eval_files__ = U.list_flights(EVAL_FOLDER)



        dfs, max_len, y, y_, mean_y_, acc, glob_pred = self.__gen_eval_batch__(self.__eval_files__)

        BAR.reset(max=max_len)
        prntC(C.INFO, "Evaluating model on : ", C.BLUE, self.__eval_files__[0].split("/")[-2])

        CHRONO.start()
        NB_MESSAGE = 0
        for t in range(max_len):
            x, files = self.__next_msgs__(dfs, t)
            NB_MESSAGE += len(x)
            for i in range(len(x)): streamer.add(x[i])

            yt_, mean_yt_, is_interp = self.predict(x)
            
            
            

            for i in range(len(files)):
                y_[files[i]][t] = yt_[i]
                mean_y_[files[i]][t] = mean_yt_[i]
                acc[files[i]] += (is_interp[i] == y[files[i]])
                
                if (is_interp[i]):
                    glob_pred[files[i]] += 1
                
            BAR.update()
            
        CHRONO.stop()

        mean_acc = 0
        acc_on_glob = 0
        
        detection_capacity = 0
        nb_iterp = 0
        
        y_true_per_message = [np.array([y[i]]*len(dfs[i])) for i in range(len(dfs))]
        y_true_per_message = np.concatenate(y_true_per_message, axis=0)
        mean_y_ = np.concatenate(mean_y_, axis=0)
        mean_y_ = np.nan_to_num(mean_y_, nan=0.0)
        y_ = np.concatenate(y_, axis=0)
        y_ = np.nan_to_num(y_, nan=0.0)
        y_bin = [(mean_y_[i] > self.CTX["THRESHOLD"]) for i in range(len(mean_y_))]
        # remplace nan with 0
        
        
        confusion_matrix = metrics.confusion_matrix(y_true_per_message, y_bin)
        Metrics.plot_confusion_matrix(confusion_matrix, self.ARTIFACTS+"/confusion_matrix.png", ["Normal", "Interpolated"])
        
       
        
        
        for i in range(len(dfs)):
            name = self.__eval_files__[i].split("/")[-1]
            prntC(C.INFO, "Accuracy for ", C.BLUE, name, C.RESET, " : ", C.BLUE, round(acc[i]/len(dfs[i])*100, 2), glob_pred[i])
            mean_acc += acc[i]/len(dfs[i])
            
            if ("interp" in name):
                detection_capacity += glob_pred[i]
                nb_iterp += len(dfs[i])
                if (glob_pred[i] > 0):
                    acc_on_glob += 1
                
            elif (glob_pred[i] == 0):
                acc_on_glob += 1
                
            
        mean_acc /= len(dfs)
        
        print("TOTAL NUM OF MESSAGES : ", NB_MESSAGE)
        accuracy = metrics.accuracy_score(y_true_per_message, y_bin)
        precision = metrics.precision_score(y_true_per_message, y_bin)
        recall = metrics.recall_score(y_true_per_message, y_bin)
        f1 = metrics.f1_score(y_true_per_message, y_bin)
        
        y_true_per_message = y_true_per_message.astype(np.float64)
        # print(list(zip(y_true_per_message, mean_y_)))
        roc_auc = metrics.roc_auc_score(y_true_per_message, y_)
        fpr = confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[0][0])
        
        prntC(C.INFO, "Overall accuracy : ", C.BLUE, round(mean_acc*100, 2), "%")
        prntC(C.INFO, "Overall accuracy : ", C.BLUE, round(accuracy*100, 2), "%")
        prntC(C.INFO, "Overall precision : ", C.BLUE, round(precision*100, 2), "%")
        prntC(C.INFO, "Overall recall : ", C.BLUE, round(recall*100, 2), "%")
        prntC(C.INFO, "Overall F1-score : ", C.BLUE, round(f1*100, 2), "%")
        prntC(C.INFO, "Overall ROC-AUC : ", C.BLUE, round(roc_auc*100, 2), "%")
        prntC(C.INFO, "Overall FPR : ", C.BLUE, round(fpr*100, 2), "%")

        return {"ACCURACY": round(accuracy*100, 2), "GLOBAL_ACC": round(acc_on_glob/len(dfs)*100, 2), "DETECTION_CAPACITY": round(detection_capacity/nb_iterp*100, 2), "TIME": round(CHRONO.get_time_s()/NB_MESSAGE*1000,2),
                "PRECISION": round(precision*100, 2), "RECALL": round(recall*100, 2), "F1": round(f1*100, 2), "ROC_AUC": round(roc_auc*100, 2), "FPR": round(fpr*100, 2)}

