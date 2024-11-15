from _Utils.os_wrapper import os
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.FloodingSolver.DataLoader import DataLoader
import D_DataLoader.FloodingSolver.Utils as SU
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


from numpy_typing import np, ax
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.FeatureGetter import FG_flooding as FG
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

TRAIN_FOLDER = "./A_Dataset/AircraftClassification/Train/"
EVAL_FOLDER = "./A_Dataset/FloodingSolver/Eval/"

H_TRAIN_LOSS = 0
H_TEST_LOSS  = 1
H_TRAIN_DIST = 2
H_TEST_DIST  = 3

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

    return np.zeros((train_batches, train_size, CTX["FEATURES_OUT"]), dtype=np.float64), \
           np.zeros((test_batches,  test_size,  CTX["FEATURES_OUT"]), dtype=np.float64), \
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
        write(self.ARTIFACTS+"/ys", self.dl.yScaler.get_variables())
        write(self.ARTIFACTS+"/pad", self.dl.PAD)


    def load(self, path:str=None) -> None:
        if (path is None):
            path = self.ARTIFACTS

        if ("LOAD_EP" in self.CTX and self.CTX["LOAD_EP"] != -1):
            self.model.set_variables(load(path+"/weights/"+str(self.CTX["LOAD_EP"])+".w"))
        else:
            self.model.set_variables(load(path+"/w"))
        self.dl.xScaler.set_variables(load(path+"/xs"))
        self.dl.yScaler.set_variables(load(path+"/ys"))
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

        y_unscaled  = self.dl.yScaler.inverse_transform(y)
        y_unscaled_ = self.dl.yScaler.inverse_transform(y_)
        dist = GEO.np.distance(y_unscaled[:, 0], y_unscaled[:, 1], y_unscaled_[:, 0], y_unscaled_[:, 1])
        dist = np.mean(dist)
        loss = Metrics.mse(y, y_)
        return dist, loss


    def __epoch_stats__(self, ep:int,
                        y_train:np.float64_2d[ax.sample, ax.feature],
                        _y_train:np.float64_2d[ax.sample, ax.feature],
                        y_test :np.float64_2d[ax.sample, ax.feature],
                        _y_test :np.float64_2d[ax.sample, ax.feature]) -> None:

        train_dist, train_loss = self.__prediction_statistics__(y_train, _y_train)
        test_dist,  test_loss  = self.__prediction_statistics__(y_test,  _y_test )

        # On first epoch, initialize history
        if (self.__ep__ == -1 or self.__ep__ > ep):
            self.__history__         = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)
            self.__history_mov_avg__ = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)


        # Save epoch statistics
        self.__ep__ = ep
        self.__history__[:, ep-1] = [train_loss, test_loss, train_dist, test_dist]
        for i in range(4):
            self.__history_mov_avg__[i, ep-1] = Metrics.moving_average_at(self.__history__[i], ep-1, w=5)
        write(self.ARTIFACTS+"/weights/"+str(ep)+".w", self.model.get_variables())

        # Display statistics !
        self.__print_epoch_stats__(ep, train_loss, test_loss, train_dist, test_dist)
        self.__plot_epoch_stats__()
        self.__plot_train_exemple__(y_train, _y_train)



    def __print_epoch_stats__(self, ep:int,
                              train_loss:float, test_loss:float,
                              train_dist:float, test_dist:float) -> None:


        prntC(C.INFO,  "Epoch :", C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, CHRONO, "s")
        prntC(C.INFO_, "Train Loss :", C.BLUE, round(train_loss, 4), C.RESET,
                     "- Test  Loss :", C.BLUE, round(test_loss,  4))
        prntC(C.INFO_, "Train Dist :", C.BLUE, round(train_dist, 4), C.RESET,
                     "- Test  Dist :", C.BLUE, round(test_dist,  4))
        prntC()

    def __plot_epoch_stats__(self) -> None:

        # plot loss curves
        Metrics.plot_loss(self.__history__[H_TRAIN_LOSS], self.__history__[H_TEST_LOSS],
                         self.__history_mov_avg__[H_TRAIN_LOSS], self.__history_mov_avg__[H_TEST_LOSS],
                            type="loss", path=self.ARTIFACTS+"/loss.png")

        Metrics.plot_loss(self.__history__[H_TRAIN_DIST], self.__history__[H_TEST_DIST],
                         self.__history_mov_avg__[H_TRAIN_DIST], self.__history_mov_avg__[H_TEST_DIST],
                            type="distance", path=self.ARTIFACTS+"/distance.png")

        GUI.visualize("/Training/Table/0/0/loss", GUI.IMAGE, self.ARTIFACTS+"/loss.png")
        GUI.visualize("/Training/Table/1/0/acc", GUI.IMAGE, self.ARTIFACTS+"/distance.png")

    def __plot_train_exemple__(self, y_train:np.float64_2d[ax.sample, ax.feature],
                                    _y_train:np.float64_2d[ax.sample, ax.feature]) -> None:
        NAME = "train_example"
        y_sample  = self.dl.yScaler.inverse_transform(np.array([y_train[-1]]))[0]
        y_sample_ = self.dl.yScaler.inverse_transform(np.array([_y_train[-1]]))[0]

        o_lat, o_lon, o_track = PLT.get_data(NAME+"Origin")
        y_sample  = U.denormalize_trajectory(self.CTX, [y_sample[0]], [y_sample[1]],
                                             o_lat, o_lon, o_track)
        y_sample_ = U.denormalize_trajectory(self.CTX, [y_sample_[0]], [y_sample_[1]],
                                             o_lat, o_lon, o_track)

        PLT.scatter(NAME, y_sample[1],  y_sample[0],  color="tab:green", marker="x")
        PLT.scatter(NAME, y_sample_[1], y_sample_[0], color="tab:purple", marker="x")

        loss = GEO.distance(y_sample[0], y_sample[1], y_sample_[0], y_sample_[1])
        PLT.title  (NAME, "Flooding Solver - Prediction on a training sample - Loss : "+ str(round(loss, 2)) + "m")

        PLT.show(NAME, self.ARTIFACTS+"/train_example.png")





# |--------------------------------------------------------------------------------------------------------------------
# |     FIND AND LOAD BEST MODEL WHEN TRAINING IS DONE
# |--------------------------------------------------------------------------------------------------------------------

    def __load_best_model__(self) -> None:

        if (len(self.__history__[1]) == 0):
            prntC(C.WARNING, "No history of training has been saved")
            return

        best_i = np.argmin(self.__history_mov_avg__[H_TEST_DIST]) + 1

        prntC(C.INFO, "load best model, epoch : ",
              C.BLUE, best_i, C.RESET, " with distance loss of : ",
              C.BLUE, self.__history__[H_TEST_DIST][best_i-1],"m")

        self.model.set_variables(load(self.ARTIFACTS+"/weights/"+str(best_i)+".w"))
        self.save()


# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================


    def predict(self, x:"list[dict[str,object]]") -> """tuple[
            np.float64_2d[ax.sample, ax.feature],
            np.float64_1d[ax.sample],
            np.bool_1d[ax.sample]]""":
        if (len(x) == 0): return np.zeros((0, self.CTX["FEATURES_OUT"])), np.zeros((0, self.CTX["FEATURES_OUT"])), np.zeros((0, ))

        # allocate memory
        x_batch  = np.zeros((len(x), self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"]))
        y_batch  = np.full((len(x), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64)
        y_batch_ = np.full((len(x), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64)
        y_       = np.full((len(x), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64)
        y        = np.full((len(x), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64)
        is_interesting = np.zeros(len(x), dtype=bool)
        is_flooding = np.zeros(len(x), dtype=bool)
        origin   = np.zeros((len(x), 3), dtype=np.float64)

        # stream message and build input batch
        for i in range(len(x)):
            x_sample, y_sample, valid, o = self.dl.process_stream_of(x[i])
            x_batch[i] = x_sample[0]
            if (valid): y_batch[i] = y_sample[0]

            is_interesting[i] = valid
            origin[i] = o


        # predict only on interesting samples
        x_preds = x_batch[is_interesting]
        y_preds_ = np.zeros((len(x_preds), self.CTX["FEATURES_OUT"]), dtype=np.float64)
        for i in range(0, len(x_preds), self.CTX["MAX_BATCH_SIZE"]):
            s = slice(i, i+self.CTX["MAX_BATCH_SIZE"])
            y_preds_[s] = self.model.predict(x_preds[s])
        y_batch_[is_interesting] = y_preds_


        # denormalize predictions
        y_batch_ = self.dl.yScaler.inverse_transform(y_batch_)
        y_batch = self.dl.yScaler.inverse_transform(y_batch)


        for i in range(len(x)):
            y_lat, y_lon = U.denormalize_trajectory(self.CTX, [y_batch_[i, 0]], [y_batch_[i, 1]],
                                                    origin[i, 0], origin[i, 1], origin[i, 2])
            ylat, ylon   = U.denormalize_trajectory(self.CTX, [y_batch [i, 0]], [y_batch [i, 1]],
                                                    origin[i, 0], origin[i, 1], origin[i, 2])
            y_[i] = [y_lat[0], y_lon[0]]
            y[i]  = [ylat [0], ylon [0]]





        # DEBUG (comment this line to remove debug plot)
        # self.__debug_plot_predictions__(x_batch, y_batch, y_batch_, y_, y, is_interesting, origin)


        # compute loss
        loss = np.full(len(x), np.nan, dtype=np.float64)
        for i in range(len(x)):
            if (is_interesting[i]):
                message_distance = GEO.distance(origin[i, 0], origin[i, 1], y[i][0], y[i][1])
                pred_distance =  GEO.distance(y_[i][0], y_[i][1], y[i][0], y[i][1])

                l = pred_distance #/message_distance * 250


                loss_win = self.dl.loss_cache.append(x[i]["icao24"], x[i]["tag"], l)
            else:
                loss_win = self.dl.loss_cache.append(x[i]["icao24"], x[i]["tag"], 0)

            loss[i] = np.mean(loss_win[-self.CTX["LOSS_MOVING_AVERAGE"]:])

        # compute flooding
        flooding:"dict[str, dict[str, [float, int]]]" = {}
        for i in range(len(x)):
            icao = x[i]["icao24"]
            tag = x[i]["tag"]
            traj = streamer.get(icao, tag)
            if (streamer.ended_flooding(traj, x[i]["timestamp"], self.CTX["HORIZON"])):
                if (icao not in flooding):
                    flooding[icao] = {}
                flooding[icao][tag] = [loss[i], i]

        for icao, tags in flooding.items():
            min_tag = min(tags, key=lambda x: tags[x][0])
            for tag in tags:
                if (tag != min_tag):
                    streamer.setAbnormal(icao, tag, x[tags[tag][1]]["timestamp"])

        # set is_flooding to True if the message is flooded
        for i in range(len(x)):
            icao = x[i]["icao24"]
            tag = x[i]["tag"]
            timestamp = x[i]["timestamp"]
            is_flooding[i] = streamer.isAbnormal(icao, tag, timestamp)

        return y_, loss, is_flooding



    def __debug_plot_predictions__(self, x_batch :np.float64_3d[ax.sample, ax.time, ax.feature],
                                         y_batch :np.float64_2d[ax.sample, ax.feature],
                                         y_batch_:np.float64_2d[ax.sample, ax.feature],
                                         y_      :np.float64_2d[ax.sample, ax.feature],
                                         y       :np.float64_2d[ax.sample, ax.feature],
                                         is_interesting:np.bool_1d,
                                         origin  :np.float64_2d[ax.sample, ax.feature]) -> None:

        interesting = np.arange(len(x_batch))[is_interesting]
        if (len(interesting) > 0):
            i = 2
            it = interesting[i]

            x_batch_norm = self.dl.xScaler.inverse_transform(x_batch)
            x_batch_norm   = x_batch_norm[it, :, 0:2]
            x_batch_denorm = U.denormalize_trajectory(self.CTX, x_batch_norm[:, 0], x_batch_norm[:, 1],
                                                   origin[it, 0], origin[it, 1], origin[it, 2])

            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            ax[0].plot   (x_batch_norm[:, 1], x_batch_norm[:, 0], color="tab:blue")
            ax[0].scatter(x_batch_norm[:, 1], x_batch_norm[:, 0], color="tab:blue", marker="x")
            ax[0].scatter(y_batch_[it, 1], y_batch_[it, 0], color="tab:purple", marker="x")
            ax[0].scatter(y_batch [it, 1], y_batch [it, 0], color="tab:green", marker="+")
            ax[0].title.set_text("Prediction before denormalization")
            ax[0].axis('equal')

            ax[1].plot   (x_batch_denorm[1], x_batch_denorm[0], color="tab:blue")
            ax[1].scatter(x_batch_denorm[1], x_batch_denorm[0], color="tab:blue", marker="x")
            ax[1].scatter(y_[it, 1], y_[it, 0], color="tab:purple", marker="x")
            ax[1].scatter(y [it, 1], y [it, 0], color="tab:green",  marker="+")
            ax[1].title.set_text("Prediction after denormalization")
            ax[1].axis('equal')


            plt.savefig(self.ARTIFACTS+f"/debug_prediction.png")
            plt.close()
            plt.clf()
            input("press enter to continue")


# |====================================================================================================================
# |     EVALUATION
# |====================================================================================================================

    def __gen_eval_batch__(self, files:"list[str]")->"""tuple[
            list[pd.DataFrame], int,
            list[np.float64_2d[ax.time, ax.feature]],
            list[np.float64_2d[ax.time, ax.feature]],
            list[np.float64_1d[ax.time]],
            list[np.float64_1d[ax.time]]]""":

        # load all ressources needed for the batch
        files_df, max_lenght, y, y_, loss, loss_ = [], 0, [], [], [], []
        for f in range(len(files)):
            df = U.read_trajectory(files[f])
            files_df.append(df)
            y_.append(np.full((len(df), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64))
            y .append(np.full((len(df), self.CTX["FEATURES_OUT"]), np.nan, dtype=np.float64))
            loss.append(np.full(len(df), np.nan, dtype=np.float64))
            loss_.append(np.full(len(df), np.nan, dtype=np.float64))

            max_lenght = max(max_lenght, len(df))

        return files_df, max_lenght, y, y_, loss, loss_


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
            self.__eval_files__ = [U.list_flights(f"{EVAL_FOLDER}{f}") for f in os.listdir(EVAL_FOLDER)]

        to_remove=[f for f in os.listdir(self.ARTIFACTS) if f.startswith("prediction")]
        for f in to_remove:
            os.remove(self.ARTIFACTS+"/"+f)

        # only keep folder named exp_solo
        # self.__eval_files__ = [f for f in self.__eval_files__ if f[0].split("/")[-2] == "exp_solo"]

        mean_acc = np.zeros(3, dtype=np.float64)
        for folder in self.__eval_files__:

            dfs, max_len, y, y_, loss, loss_ = self.__gen_eval_batch__(folder)

            BAR.reset(max=max_len)
            prntC(C.INFO, "Evaluating model on : ", C.BLUE, folder[0].split("/")[-2])

            first = True

            for t in range(max_len):
                x, files = self.__next_msgs__(dfs, t)
                for i in range(len(x)): streamer.add(x[i])

                yt_, losst_, _ = self.predict(x)

                for i in range(len(files)):
                    y_[files[i]][t] = yt_[i]
                    y [files[i]][t] = [x[i]["latitude"], x[i]["longitude"]]
                    loss_[files[i]][t] = losst_[i]
                    loss [files[i]][t] = GEO.distance(x[i]["latitude"], x[i]["longitude"], yt_[i][0], yt_[i][1])
                BAR.update()

            name = folder[0].split("/")[-2]
            best, acc = self.__eval_stats__(loss, loss_, y_, y, dfs, max_len, name=name)
            mean_acc += acc
            prntC(C.INFO, "Accuracy for ", C.BLUE, name, C.RESET, " : ", C.BLUE, round(best*100, 2), "% (", acc,")" )

        mean_acc /= len(self.__eval_files__)

        prntC(C.INFO, "accuracy by mean loss : ", C.BLUE, round(mean_acc[0]*100, 2), "%")
        prntC(C.INFO, "accuracy by loss at attack : ", C.BLUE, round(mean_acc[1]*100, 2), "%")
        prntC(C.INFO, "accuracy by loss around attack : ", C.BLUE, round(mean_acc[2]*100, 2), "%")

        prntC(C.INFO, "best accuracy : ", C.BLUE, round(mean_acc.max()*100, 2), "%")

        return {"mean_acc": mean_acc}



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


