from _Utils.os_wrapper import os
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

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

TRAIN_FOLDER = "./A_Dataset/InterpolationDetector/Train/"
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
                is_interp[i] = y_mean[i] > 0.9


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



        dfs, max_len, y, y_, loss_, acc, glob_pred = self.__gen_eval_batch__(self.__eval_files__)

        BAR.reset(max=max_len)
        prntC(C.INFO, "Evaluating model on : ", C.BLUE, self.__eval_files__[0].split("/")[-2])

        CHRONO.start()
        NB_MESSAGE = 0
        for t in range(max_len):
            x, files = self.__next_msgs__(dfs, t)
            NB_MESSAGE += len(x)
            for i in range(len(x)): streamer.add(x[i])

            yt_, losst_, is_interp = self.predict(x)
            
            
            

            for i in range(len(files)):
                y_[files[i]][t] = yt_[i]
                loss_[files[i]][t] = losst_[i]
                acc[files[i]] += (is_interp[i] == y[files[i]])
                
                if (is_interp[i]):
                    glob_pred[files[i]] += 1
                
            BAR.update()
            
        CHRONO.stop()

        mean_acc = 0
        acc_on_glob = 0
        
        detection_capacity = 0
        nb_iterp = 0
        
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

        return {"ACCURACY": round(mean_acc*100, 2), "GLOBAL_ACC": round(acc_on_glob/len(dfs)*100, 2), "DETECTION_CAPACITY": round(detection_capacity/nb_iterp*100, 2), "TIME": round(CHRONO.get_time_s()/NB_MESSAGE*1000,2)}



# |====================================================================================================================
# |     EVALUATION STATISTICS
# |====================================================================================================================


    def __eval_stats__(self,
                       loss:"list[np.float64_1d]",
                       loss_:"list[np.float64_1d]",
                       y_:"list[np.float64_2d[ax.time, ax.feature]]",
                       y:"list[np.float64_2d[ax.time, ax.feature]]",
                       dfs:"list[pd.DataFrame]",
                       max_len:int, name:str) -> float:

        # plot mean loss (along flights), per timestamp
        mean_loss = np.zeros(max_len, dtype=np.float64)
        mean_loss_ = np.zeros(max_len, dtype=np.float64)
        for t in range(max_len):
            files = [f for f in range(len(loss)) if t < len(loss[f])]
            mean_loss[t] = np.nanmean([loss[f][t] for f in files])
            mean_loss_[t] = np.nanmean([loss_[f][t] for f in files])


        return self.__plot_eval__(dfs, y_, y, loss, loss_, mean_loss, mean_loss_, max_len, name)



    def __plot_eval__(self, dfs:"list[pd.DataFrame]",
                      y_:"list[np.float64_2d[ax.time, ax.feature]]", y:"list[np.float64_2d[ax.time, ax.feature]]",
                      loss:"list[np.float64_1d]", loss_:"list[np.float64_1d]", mean_loss:np.float64_1d, mean_loss_:np.float64_1d,
                      max_len:int, name:str) -> float:

        self.__plot_loss_curves__(loss, loss_, mean_loss, mean_loss_, max_len, name)
        self.__plot_predictions_on_saturation__(dfs, y_, y, loss, max_len, name)
        self.__plot_prediction_on_error_spikes__(dfs, y_, y, loss, loss_, max_len, name)
        return self.__plot_safe_icao24__(dfs, loss, name)


# |====================================================================================================================
# | SUB FUNCTIONS FOR PLOTTING
# |====================================================================================================================


    def __plot_loss_curves__(self, loss:"list[np.float64_1d]", loss_:"list[np.float64_1d]",
                             mean_loss:np.float64_1d, mean_loss_:np.float64_1d,
                             max_len:int, name:str) -> None:
        COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        plt.figure(figsize=(24 * max_len / 500, 12))
        for f in range(len(loss)):
            plt.plot(loss[f], color=COLORS[f%len(COLORS)], linestyle="--")
            plt.plot(loss_[f], color=COLORS[f%len(COLORS)])
        plt.plot(mean_loss, color="black", linestyle="--", linewidth=1)
        plt.plot(mean_loss_, color="black", linewidth=2)
        plt.plot([0, max_len], [self.CTX["THRESHOLD"]]*2, color="black", linestyle="--", linewidth=2)
        plt.title("Mean Loss along timestamps")
        plt.xlabel("Timestamp")
        plt.ylabel("Distance Loss (m)")
        plt.grid()
        plt.savefig(self.ARTIFACTS+f"/eval_loss_{name}.png")





    def __plot_predictions_on_saturation__(self, dfs:"list[pd.DataFrame]",
                                            y_:"list[np.float64_2d[ax.time, ax.feature]]",
                                            y:"list[np.float64_2d[ax.time, ax.feature]]",
                                            loss:"list[np.float64_1d]", max_len:int, name:str) -> None:

        if (len(dfs) == 1): return
        # find when the saturation is reached
        attack_t = 0
        while(loss[0][attack_t] == loss[1][attack_t] or np.isnan(loss[0][attack_t]) or np.isnan(loss[1][attack_t])):
            attack_t += 1
        s = slice(max(0, attack_t-20), min(max_len, attack_t+20))

        # plot the trajectory and the prediction in a map
        min_lat, min_lon = np.inf, np.inf
        max_lat, max_lon = -np.inf, -np.inf
        for i in range(len(dfs)):
            lat = dfs[i]["latitude"].to_numpy()[s]
            lon = dfs[i]["longitude"].to_numpy()[s]
            min_lat = min(min_lat, lat.min())
            min_lon = min(min_lon, lon.min())
            max_lat = max(max_lat, lat.max())
            max_lon = max(max_lon, lon.max())


        nb_flight = min(9, len(dfs))
        side = int(np.ceil(np.sqrt(nb_flight)))
        col = side
        row = int(np.ceil(nb_flight / side))


        box = [min_lat, min_lon, max_lat, max_lon]
        PLT.figure (name, box[0], box[1], box[2], box[3], figsize=(15*row, 15*col), sub_plots=(row, col))


        for i in range(nb_flight):
            r, c = i // col, i % col

            lat = dfs[i]["latitude"].to_numpy()
            lon = dfs[i]["longitude"].to_numpy()

            laty_ = y_[i][:, 0]
            lony_ = y_[i][:, 1]
            laty  = y [i][:, 0]
            lony  = y [i][:, 1]

            PLT.subplot(name, r, c).plot   (lon[s],  lat[s],  color="tab:blue")
            PLT.subplot(name, r, c).scatter(lon[s],  lat[s],  color="tab:blue", marker="x")
            PLT.subplot(name, r, c).scatter(lon[attack_t],  lat[attack_t],  color="tab:red", marker="x")
            PLT.subplot(name, r, c).scatter(lony_[s], laty_[s], color="tab:purple", marker="x")
            PLT.subplot(name, r, c).scatter(lony[s], laty[s], color="tab:green", marker="+")

            for t in range(s.start, s.stop):
                PLT.subplot(name, r, c).plot([lon[t], lony_[t]], [lat[t], laty_[t]], color="black", linestyle="--")
                PLT.subplot(name, r, c).plot([lony[t], lony_[t]], [laty[t], laty_[t]], color="black", linestyle="--")

        PLT.show(name, self.ARTIFACTS+f"/predictions_{name}.png")


    def __plot_prediction_on_error_spikes__(self, dfs:"list[pd.DataFrame]",
                                            y_:"list[np.float64_2d[ax.time, ax.feature]]",
                                            y:"list[np.float64_2d[ax.time, ax.feature]]",
                                            loss:"list[np.float64_1d]", loss_:"list[np.float64_1d]",
                                            max_len:int, name:str) -> None:

        # find error spikes
        spikes = []
        for t in range(max_len):
            remaining = [i for i in range(len(dfs)) if t < len(dfs[i])]
            max_i = remaining[np.argmax([loss_[f][t] for f in remaining])]
            if (loss_[max_i][t] > self.CTX["THRESHOLD"]):
                spikes.append([t, max_i])

        if (len(spikes) == 0):
            return


        GAP = 15
        gaps = []
        gap = []
        start = 0
        i = start
        while (i < len(spikes)-1):

            if (spikes[i][1] == spikes[start][1] and spikes[i][0] - spikes[start][0] <= GAP and i-start < 100):
                gap.append(spikes[i])
            else:
                gaps.append(gap)
                start = i
                gap = [spikes[i]]
            i += 1
        gaps.append(gap)
        gaps = [(g[0][0], g[-1][0], g[0][1]) for g in gaps]

        pdf = PdfPages(self.ARTIFACTS+f"/eval_mistakes_{name}.pdf")
        for g in gaps:
            flight = g[2]
            gap_start = g[0]
            gap_end = g[1]

            traj_slice = slice(max(0, gap_start-GAP), min(len(dfs[flight]), g[1]+2))
            pred_start = g[0]
            pred_end = min(max_len, gap_end+1)
            pred_slice = slice(pred_start, pred_end)

            lat = dfs[flight]["latitude"].to_numpy()
            lon = dfs[flight]["longitude"].to_numpy()
            laty_ = y_[flight][:, 0]
            lony_ = y_[flight][:, 1]

            box = [lat[traj_slice].min(), lon[traj_slice].min(),
                   lat[traj_slice].max(), lon[traj_slice].max()]
            box = [min(box[0], laty_[traj_slice].min())-0.001, min(box[1], lony_[traj_slice].min()-0.001),
                   max(box[2], laty_[traj_slice].max()+0.001), max(box[3], lony_[traj_slice].max()+0.001)]

            PLT.figure(name, box[0], box[1], box[2], box[3],
                        figsize=(15, 15),
                        sub_plots=(2, 1), display_map=[[True], [False]])

            PLT.subplot(name, 0, 0).plot   (lon[traj_slice],  lat[traj_slice],  color="tab:blue", label="trajectory")
            PLT.subplot(name, 0, 0).scatter(lon[traj_slice],  lat[traj_slice],  color="tab:blue", marker="x")
            PLT.subplot(name, 0, 0).scatter(lony_[traj_slice], laty_[traj_slice], color="tab:green",
                                            marker="x", label="prediction")
            PLT.subplot(name, 0, 0).scatter(lony_[pred_slice], laty_[pred_slice], color="tab:purple",
                                            marker="x", label="wrong")
            # plot actual position
            PLT.subplot(name, 0, 0).scatter(lon[traj_slice.stop-1], lat[traj_slice.stop-1], color="tab:red",
                                            marker="o", label="actual position")
            for t in range(traj_slice.start, traj_slice.stop):
                PLT.subplot(name, 0, 0).plot([lon[t], lony_[t]], [lat[t], laty_[t]], color="black", linestyle="--")


            # plot loss_
            PLT.subplot(name, 1, 0).plot(list(range(traj_slice.start, traj_slice.stop)),
                                         loss_[flight][traj_slice],
                                         label="loss_")
            PLT.subplot(name, 1, 0).scatter(list(range(traj_slice.start, traj_slice.stop)),
                                            loss_[flight][traj_slice],
                                            color="tab:green", marker="x")

            PLT.subplot(name, 1, 0).scatter(list(range(pred_start, pred_end)),
                                            loss_[flight][pred_slice],
                                            color="tab:purple", marker="x",
                                            label="prediction")

            # plot loss
            PLT.subplot(name, 1, 0).plot(list(range(traj_slice.start, traj_slice.stop)),
                                         loss[flight][traj_slice],
                                         label="loss")


            PLT.subplot(name, 1, 0).plot([traj_slice.start, traj_slice.stop-1],
                                         [self.CTX["THRESHOLD"]]*2, label="limit", linestyle="--", color="black")
            PLT.legend(name)
            PLT.show(name, pdf=pdf)
        pdf.close()


    def __plot_safe_icao24__(self, dfs:"list[pd.DataFrame]", loss:"list[np.float64_1d]", name:str) -> float:
        if (len(dfs) == 1): return 1,  [1, 1, 1]
        icaos24 = [df["icao24"].iloc[0] for df in dfs]

        # find when the saturation is reached
        attack_t = 0
        while(loss[0][attack_t] == loss[1][attack_t] or np.isnan(loss[0][attack_t]) or np.isnan(loss[1][attack_t])):
            attack_t += 1

        short_slice = slice(attack_t, attack_t+self.CTX["LOSS_MOVING_AVERAGE"])
        long_slice = slice(attack_t, attack_t+self.CTX["HISTORY"]//2)


        mean_loss_per_flights =   [np.nanmean(loss[f])              for f in range(len(loss))]
        mean_loss_at_attack =     [np.nanmean(loss[f][short_slice]) for f in range(len(loss))]
        mean_loss_around_attack = [np.nanmean(loss[f][long_slice])  for f in range(len(loss))]

        LABELS = ["Mean Loss", "Mean Loss at Attack", "Mean Loss around Attack"]

        fig, ax = plt.subplots(1, len(LABELS), figsize=(15 * len(dfs) / 10.0, 5))

        colors = ["tab:blue", "tab:green", "tab:purple", "tab:red", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        best_acc = 0
        acc = [0, 0, 0]
        for i, loss in enumerate([mean_loss_per_flights, mean_loss_at_attack, mean_loss_around_attack]):

            order = np.argsort(loss)

            sorted_icaos24 = [icaos24[i] for i in order]
            sorted_loss = [loss[i] for i in order]

            # true icao i is the index of the longuest icaco24
            true_icao_i = np.argmax([len(sorted_icaos24[i]) for i in range(len(sorted_icaos24))])
            acc[i] = 1-true_icao_i/(len(sorted_icaos24)-1)
            if (acc[i] > best_acc):
                best_acc = acc[i]

            for j in range(len(sorted_loss)):
                col = colors[order[j] % len(colors)]
                ax[i].bar(j, sorted_loss[j], color=col, label=sorted_icaos24[j])

            ax[i].set_xticks(range(len(icaos24)))
            ax[i].set_xticklabels(sorted_icaos24, rotation=70)
            ax[i].set_title(LABELS[i])
            ax[i].grid()
        fig.tight_layout()
        plt.savefig(self.ARTIFACTS+f"/safe_icao24_{name}.png")

        return best_acc, acc


