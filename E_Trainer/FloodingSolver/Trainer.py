import os
import pandas as pd
import numpy as np

from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.FloodingSolver.DataLoader import DataLoader
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer

import _Utils.Metrics as Metrics
from   _Utils.save import write, load
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.ProgressBar import ProgressBar
from   _Utils.Chrono import Chrono
from   _Utils.DebugGui import GUI
import _Utils.plotADSB as PLT
from   _Utils.Typing import NP, AX
import _Utils.geographic_maths as GEO


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
        NP.float32_3d[AX.batch, AX.sample, AX.feature],
        NP.float32_3d[AX.batch, AX.sample, AX.feature],
        NP.float32_1d, NP.float32_1d]""":

    return np.zeros((train_batches, train_size, CTX["FEATURES_OUT"]), dtype=np.float32), \
           np.zeros((test_batches,  test_size,  CTX["FEATURES_OUT"]), dtype=np.float32), \
           np.zeros(train_batches, dtype=np.float32), np.zeros(test_size, dtype=np.float32)


# |====================================================================================================================
# | BEGIN OF TRAINER CLASS
# |====================================================================================================================


class Trainer(AbstractTrainer):

    def __init__(self, CTX:dict, Model:"type[_Model_]") -> None:

        # Public attributes
        self.CTX = CTX
        self.model:_Model_ = Model(CTX)
        self.__makes_artifacts__()
        self.__init_GUI__()
        self.viz_model(self.ARTIFACTS)
        GUI.visualize("/Model/Achitecture", GUI.IMAGE, self.ARTIFACTS+f"/{self.model.name}.png")

        self.dl = DataLoader(CTX, TRAIN_FOLDER)

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


    def __init_GUI__(self) -> None:
        GUI.visualize("/Model", GUI.COLAPSIING_HEADER)



# |====================================================================================================================
# |     SAVE & LOAD MODEL'S VARIABLES
# |====================================================================================================================
    def save(self) -> None:
        write(self.ARTIFACTS+"/w", self.model.getVariables())
        write(self.ARTIFACTS+"/xs", self.dl.xScaler.getVariables())
        write(self.ARTIFACTS+"/ys", self.dl.yScaler.getVariables())
        write(self.ARTIFACTS+"/pad", self.dl.PAD)

    def load(self, path:str) -> None:
        if (path is None):
            path = self.ARTIFACTS

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

            # Allocate variables
            x_train, y_train = self.dl.genEpochTrain()
            x_test,  y_test  = self.dl.genEpochTest()

            _y_train, _y_test, loss_train, loss_test = __alloc_pred_batches__(
                CTX, len(x_train), len(x_train[0]), len(x_test),  len(x_test[0]))


            CHRONO.start()
            BAR.reset(max=len(x_train) + len(x_test))

            # Training
            for batch in range(len(x_train)):
                loss_train[batch], _y_train[batch] = self.model.training_step(x_train[batch], y_train[batch])
                BAR.update()
            _y_train:NP.float32_2d[AX.sample, AX.feature] = _y_train.reshape(-1, _y_train.shape[-1])
            y_train :NP.float32_2d[AX.sample, AX.feature] =  y_train.reshape(-1,  y_train.shape[-1])

            # Testing
            for batch in range(len(x_test)):
                loss_test[batch], _y_test[batch] = self.model.compute_loss(x_test[batch], y_test[batch])
                BAR.update()
            _y_test:NP.float32_2d[AX.sample, AX.feature] = _y_test.reshape(-1, _y_test.shape[-1])
            y_test :NP.float32_2d[AX.sample, AX.feature] =  y_test.reshape(-1,  y_test.shape[-1])

            self.__epoch_stats__(ep, y_train, _y_train, y_test, _y_test)

        self.__load_best_model__()

# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------

    def __prediction_statistics__(self,
                                  y:NP.float32_2d[AX.sample, AX.feature],
                                  y_:NP.float32_2d[AX.sample, AX.feature])\
            -> "tuple[float, float]":

        y_unscaled  = self.dl.yScaler.inverse_transform(y)
        y_unscaled_ = self.dl.yScaler.inverse_transform(y_)
        dist = GEO.np.distance(y_unscaled[:, 0], y_unscaled[:, 1], y_unscaled_[:, 0], y_unscaled_[:, 1])
        dist = np.mean(dist)
        loss = Metrics.mse(y, y_)
        return dist, loss


    def __epoch_stats__(self, ep:int,
                        y_train:NP.float32_2d[AX.sample, AX.feature],
                        _y_train:NP.float32_2d[AX.sample, AX.feature],
                        y_test :NP.float32_2d[AX.sample, AX.feature],
                        _y_test :NP.float32_2d[AX.sample, AX.feature]) -> None:

        train_dist, train_loss = self.__prediction_statistics__(y_train, _y_train)
        test_dist,  test_loss  = self.__prediction_statistics__(y_test,  _y_test )

        # On first epoch, initialize history
        if (self.__ep__ == -1 or self.__ep__ > ep):
            self.__history__         = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float32)
            self.__history_mov_avg__ = np.full((4, self.CTX["EPOCHS"]), np.nan, dtype=np.float32)


        # Save epoch statistics
        self.__ep__ = ep
        self.__history__[:, ep-1] = [train_loss, test_loss, train_dist, test_dist]
        for i in range(4):
            self.__history_mov_avg__[i, ep-1] = Metrics.moving_average_at(self.__history__[i], ep-1, w=5)
        write(self.ARTIFACTS+"/weights/"+str(ep)+".w", self.model.getVariables())

        # Print & Display statistics !
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
        Metrics.plotLoss(self.__history__[H_TRAIN_LOSS], self.__history__[H_TEST_LOSS],
                         self.__history_mov_avg__[H_TRAIN_LOSS], self.__history_mov_avg__[H_TEST_LOSS],
                            type="loss", path=self.ARTIFACTS+"/loss.png")

        Metrics.plotLoss(self.__history__[H_TRAIN_DIST], self.__history__[H_TEST_DIST],
                         self.__history_mov_avg__[H_TRAIN_DIST], self.__history_mov_avg__[H_TEST_DIST],
                            type="distance", path=self.ARTIFACTS+"/distance.png")

        GUI.visualize("/Training/Table/0/0/loss", GUI.IMAGE, self.ARTIFACTS+"/loss.png")
        GUI.visualize("/Training/Table/1/0/acc", GUI.IMAGE, self.ARTIFACTS+"/distance.png")

    def __plot_train_exemple__(self, y_train:NP.float32_2d[AX.sample, AX.feature],
                                    _y_train:NP.float32_2d[AX.sample, AX.feature]) -> None:
        NAME = "train_example"
        y_sample  = self.dl.yScaler.inverse_transform([y_train[-1]])[0]
        y_sample_ = self.dl.yScaler.inverse_transform([_y_train[-1]])[0]

        o_lat, o_lon = PLT.get_data(NAME+"Origin")
        y_sample  = U.denormalize_trajectory(self.CTX, [y_sample[0]], [y_sample[1]], o_lat, o_lon, 0)
        y_sample_ = U.denormalize_trajectory(self.CTX, [y_sample_[0]], [y_sample_[1]], o_lat, o_lon, 0)

        PLT.scatter(NAME, y_sample[0],  y_sample[1],  color="tab:purple", marker="x")
        PLT.scatter(NAME, y_sample_[0], y_sample_[1], color="tab:red", marker="x")

        PLT.show(NAME, self.ARTIFACTS+"/train_example.png")





# |--------------------------------------------------------------------------------------------------------------------
# |     FIND AND LOAD BEST MODEL WHEN TRAINING IS DONE
# |--------------------------------------------------------------------------------------------------------------------

    def __load_best_model__(self) -> None:

        if (len(self.__history__[1]) == 0):
            prntC(C.WARNING, "No history of training has been saved")
            return

        best_i = np.argmax(self.__history_mov_avg__[H_TEST_DIST]) + 1

        prntC(C.INFO, "load best model, epoch : ",
              C.BLUE, best_i, C.RESET, " with Acc : ",
              C.BLUE, self.__history__[H_TEST_DIST][best_i-1])

        self.model.set_variables(load(self.ARTIFACTS+"/weights/"+str(best_i)+".w"))
        self.save()






    def eval(self):
        """
        Evaluate the model and return metrics


        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        if not(os.path.exists("./A_Dataset/FloodingSolver/Outputs/Eval")):
                os.makedirs("./A_Dataset/FloodingSolver/Outputs/Eval")

        CTX = self.CTX
        FOLDER = "./A_Dataset/FloodingSolver/Eval"
        PRED_FEATURES = CTX["PRED_FEATURES"]
        PRED_LAT = CTX["PRED_FEATURE_MAP"]["latitude"]
        PRED_LON = CTX["PRED_FEATURE_MAP"]["longitude"]



        # clear Outputs/path folder
        os.system("rm -rf ./A_Dataset/FloodingSolver/Outputs/*")

        # list all folder in the eval folder
        folders = os.listdir(FOLDER)
        folders = [folder for folder in folders if os.path.isdir(os.path.join(FOLDER, folder))]
        for folder in folders:

            path = os.path.join(FOLDER, folder)

            files = os.listdir(path)
            files = [file for file in files if file.endswith(".csv")]

            # check Outputs/path folder exists
            if not(os.path.exists("./A_Dataset/FloodingSolver/Outputs/"+folder)):
                os.makedirs("./A_Dataset/FloodingSolver/Outputs/"+folder)

            print("EVAL : "+folder+" : "+str(len(files))+" files", flush=True)


            mean_distances = []

            for i in range(len(files)):
                LEN = 20
                # nb = int((i+1)/len(files)*LEN)
                # print("EVAL : |", "-"*(nb)+" "*(LEN-nb)+"| "+str(i + 1).rjust(len(str(len(files))), " ") + "/" + str(len(files)), end="\r", flush=True)

                file = files[i]
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path, sep=",",dtype={"callsign":str, "icao24":str})
                df_timestamp = df["timestamp"] - df["timestamp"][0]
                df_ts_map = dict([[df_timestamp[i], i] for i in range(len(df))])
                x_inputs, y_batches, ts = self.dl.genEval(file_path)


                if (len(x_inputs) == 0): # skip empty file (no label)
                    continue

                y_batches_ = np.zeros((len(x_inputs), CTX["FEATURES_OUT"]), dtype=np.float32)
                jumps = 256
                for b in range(0, len(x_inputs),jumps):
                    x_batch = x_inputs[b:b+jumps]
                    pred =  self.model.predict(x_batch).numpy()
                    y_batches_[b:b+jumps] = pred


                y_batches_ = self.dl.yScaler.inverse_transform(y_batches_)
                y_batches = self.dl.yScaler.inverse_transform(y_batches)

                d = distance(CTX, y_batches_, y_batches)
                mean_distances.append(d)

                # analyse outputs and give the ghost aircraft

                pred_lat = y_batches_[:,PRED_LAT]
                pred_lon = y_batches_[:,PRED_LON]
                true_lat = y_batches[:,PRED_LAT]
                true_lon = y_batches[:,PRED_LON]


                pred_lat = pad_pred(ts, df_ts_map, pred_lat, CTX)
                pred_lon = pad_pred(ts, df_ts_map, pred_lon, CTX)
                true_lat = pad_pred(ts, df_ts_map, true_lat, CTX)
                true_lon = pad_pred(ts, df_ts_map, true_lon, CTX)



                traj_lat = df["latitude"].values
                traj_lon = df["longitude"].values

                pred_lat, pred_lon = self.un_transform(traj_lat, traj_lon, pred_lat, pred_lon)
                true_lat, true_lon = self.un_transform(traj_lat, traj_lon, true_lat, true_lon)


                out_df = pd.DataFrame()
                out_df["timestamp"] = df["timestamp"].values
                out_df["df_latitude"] = df["latitude"].values
                out_df["df_longitude"] = df["longitude"].values
                out_df["true_latitude"] = true_lat
                out_df["true_longitude"] = true_lon
                out_df["pred_latitude"] = pred_lat
                out_df["pred_longitude"] = pred_lon

                out_df.to_csv("./A_Dataset/FloodingSolver/Outputs/"+folder+"/"+file, index=False)


            print("Mean distance : ", np.mean(mean_distances), flush=True)


            # analyse outputs and give the ghost aircraft

        return {}

