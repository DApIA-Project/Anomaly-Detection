
# MDSM : Mean Dense Simple Model

import _Utils.Metrics as Metrics
from _Utils.save import write, load
import _Utils.Color as C
from _Utils.Color import prntC
# from _Utils.plotADSB import plotADSB

from D_DataLoader.FloodingSolver.Utils import distance, undo_batchPreProcess


from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.FloodingSolver.DataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


import os
import time
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages



def reshape(x):
    """
    x = [batch size][[x],[takeoff],[map]]
    x = [[x of batch size], [takeoff of batch size], [map of batch size]]
    """
    x_reshaped = []
    for i in range(len(x[0])):
        x_reshaped.append([])

        for j in range(len(x)):
            x_reshaped[i].append(x[j][i])

        x_reshaped[i] = np.array(x_reshaped[i])

    return x_reshaped


def pad_pred(ts, df_ts_map, pred, CTX):
    """
    Pad the prediction with nan values
    """
    
    pred_pad = np.full(len(df_ts_map), np.nan, dtype=np.float32)
    for i in range(len(ts)):
        pred_pad[df_ts_map[ts[i]]] = pred[i]
    return pred_pad

class Trainer(AbstractTrainer):
    """"
    Manage the whole training of a Direct model.
    (A model that can directly output the desired result from a dataset)

    Parameters :
    ------------

    CTX : dict
        The hyperparameters context
    
    model : type[Model]
        The model class of the model we want to train

    Attributes :
    ------------

    CTX : dict
        The hyperparameters context

    dl : DataLoader
        The data loader corresponding to the problem
        we want to solve

    model : Model
        The model instance we want to train   

    Methods :
    ---------

    run(): Inherited from AbstractTrainer
        Run the whole training pipeline
        and give metrics about the model's performance

    train():
        Manage the training loop

    eval():
        Evaluate the model and return metrics
    """

    def __init__(self, CTX:dict, Model:"type[_Model_]"):
        super().__init__(CTX, Model)
        self.CTX = CTX

        self.model:_Model_ = Model(CTX)
        
        try:
            self.model.visualize("./")
        except Exception as e:
            print("WARNING : visualization of the model failed")
            print(e)


        raise Exception("Stop here")

        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")
        
        # If "_Artifactss/" folder doesn't exist, create it.
        if not os.path.exists("./_Artifacts"):
            os.makedirs("./_Artifacts")



    def train(self):
        """
        Train the model.
        Plot the loss curves into Artefacts folder.
        """
        CTX = self.CTX
        
        history = [[], [], [], []]

        best_variables = None

        # if _Artifacts/modelsW folder exists and is not empty, clear it
        if os.path.exists("./_Artifacts/modelsW"):
            if (len(os.listdir("./_Artifacts/modelsW")) > 0):
                os.system("rm ./_Artifacts/modelsW/*")
        else:
            os.makedirs("./_Artifacts/modelsW")

        for ep in range(1, CTX["EPOCHS"] + 1):
            ##############################
            #         Training           #
            ##############################
            start = time.time()
            x_inputs, y_batches = self.dl.genEpochTrain(CTX["NB_BATCH"], CTX["BATCH_SIZE"])

            # print(x_inputs.shape)
            
            train_loss = 0
            train_distance = 0
            for batch in range(len(x_inputs)):
                loss, output = self.model.training_step(x_inputs[batch], y_batches[batch])
                train_loss += loss

                output = self.dl.yScaler.inverse_transform(output.numpy())
                true = self.dl.yScaler.inverse_transform(y_batches[batch])
                train_distance += distance(CTX, output, true)

            train_loss /= len(x_inputs)
            train_distance /= len(x_inputs)

            ##############################
            #          Testing           #
            ##############################
            # if (test_save_x is None):
            #     test_save_x, test_save_y = self.dl.genEpochTest()
            x_inputs, test_y = self.dl.genEpochTest()

            test_loss = 0
            test_distance = 0
            n = 0
            for batch in range(0, len(x_inputs), CTX["BATCH_SIZE"]):
                sub_test_x = x_inputs[batch:batch+CTX["BATCH_SIZE"]]
                sub_test_y = test_y[batch:batch+CTX["BATCH_SIZE"]]

                sub_loss, sub_output = self.model.compute_loss(sub_test_x, sub_test_y)

                test_loss += sub_loss

                sub_output = self.dl.yScaler.inverse_transform(sub_output.numpy())
                sub_true = self.dl.yScaler.inverse_transform(sub_test_y)
                test_distance += distance(CTX, sub_output, sub_true)

                n += 1

            test_loss /= n
            test_distance /= n


            # Verbose area
            print()
            print(f"Epoch {ep}/{CTX['EPOCHS']} - train_loss: {train_loss:.4f} - train_distance: {train_distance:.4f} - test_loss: {test_loss:.4f} - test_distance: {test_distance:.4f} - time: {time.time() - start:.2f}s", flush=True)

            # Save the model loss
            history[0].append(train_loss)
            history[1].append(test_loss)
            history[2].append(train_distance)
            history[3].append(test_distance)
            
            # Save the model weights
            write("./_Artifacts/modelsW/"+self.model.name+"_"+str(ep)+".w", self.model.getVariables())


        # Compute the moving average of the loss for a better visualization
        history_avg = [[], [], [], []]
        window_len = 5
        for i in range(len(history[0])):
            min_ = max(0, i - window_len)
            max_ = min(len(history[0]), i + window_len)
            history_avg[0].append(np.mean(history[0][min_:max_]))
            history_avg[1].append(np.mean(history[1][min_:max_]))
            history_avg[2].append(np.mean(history[2][min_:max_]))
            history_avg[3].append(np.mean(history[3][min_:max_]))

        Metrics.plotLoss(history[0], history[1], history_avg[0], history_avg[1])
        Metrics.plotLoss(history[2], history[3], history_avg[2], history_avg[3], filename="distance.png")

        # #  load back best model
        if (len(history[1]) > 0):
            # find best model epoch
            best_i = np.argmin(history_avg[3])

            print("load best model, epoch : ", best_i+1, " with distance : ", history[3][best_i], flush=True)
            
            best_variables = load("./_Artifacts/modelsW/"+self.model.name+"_"+str(best_i+1)+".w")
            self.model.setVariables(best_variables)
        else:
            print("WARNING : no history of training has been saved")


        write("./_Artifacts/"+self.model.name+".w", self.model.getVariables())
        write("./_Artifacts/"+self.model.name+".xs", self.dl.xScaler.getVariables())
        write("./_Artifacts/"+self.model.name+".ys", self.dl.yScaler.getVariables())
        write("./_Artifacts/"+self.model.name+".min", self.dl.FEATURES_MIN_VALUES)

    def load(self):
        """
        Load the model's weights from the _Artifacts folder
        """
        self.model.setVariables(load("./_Artifacts/"+self.model.name+".w"))
        self.dl.xScaler.setVariables(load("./_Artifacts/"+self.model.name+".xs"))
        self.dl.yScaler.setVariables(load("./_Artifacts/"+self.model.name+".ys"))
        self.dl.FEATURES_MIN_VALUES = load("./_Artifacts/"+self.model.name+".min")



    def un_transform(self, lats, lons, lats_preds, lons_preds):
        CTX = self.CTX
        for t in range(CTX["HORIZON"], len(lats)):
            o_lat = lats[t - CTX["HORIZON"]]
            o_lon = lons[t - CTX["HORIZON"]]
            track = 0

            lats_preds[t], lons_preds[t] = undo_batchPreProcess(CTX, o_lat, o_lon, track, lats_preds[t], lons_preds[t], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])

        return lats_preds, lons_preds


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

