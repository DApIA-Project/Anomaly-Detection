
# MDSM : Mean Dense Simple Model

import _Utils.mlflow as mlflow
import _Utils.Metrics as Metrics
from _Utils.save import write, load, formatJson
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.plotADSB import plotADSB


from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.AircraftClassification.DataLoader import DataLoader
from D_DataLoader.AircraftClassification.Utils import angle_diff
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
            self.model.visualize()
        except:
            print("WARNING : visualization of the model failed")

        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")
        
        # If "_Artefacts/" folder doesn't exist, create it.
        if not os.path.exists("./_Artefact"):
            os.makedirs("./_Artefact")





    def train(self):
        """
        Train the model.
        Plot the loss curves into Artefacts folder.
        """
        CTX = self.CTX
        
        history = [[], [], [], []]

        best_variables = None
        best_loss= 10000000

        test_save_x, test_save_y = None, None

        # if _Artefact/modelsW folder exists and is not empty, clear it
        if os.path.exists("./_Artefact/modelsW"):
            if (len(os.listdir("./_Artefact/modelsW")) > 0):
                os.system("rm ./_Artefact/modelsW/*")
        else:
            os.makedirs("./_Artefact/modelsW")

        for ep in range(1, CTX["EPOCHS"] + 1):
            ##############################
            #         Training           #
            ##############################
            start = time.time()
            x_inputs, y_batches = self.dl.genEpochTrain(CTX["NB_BATCH"], CTX["BATCH_SIZE"])

            # count the number of sample in each class
            nb_sample_per_class = np.zeros((CTX["FEATURES_OUT"]), dtype=np.int32)
            for batch in range(len(y_batches)):
                for t in range(len(y_batches[batch])):
                    nb_sample_per_class[np.argmax(y_batches[batch][t])] += 1

            train_loss = 0
            train_y_ = []
            train_y = []
            for batch in range(len(x_inputs)):
                loss, output = self.model.training_step(x_inputs[batch], y_batches[batch])
                train_loss += loss

                train_y_.append(output)
                train_y.append(y_batches[batch])

            train_loss /= len(x_inputs)
            train_y_ = np.concatenate(train_y_, axis=0)
            train_y = np.concatenate(train_y, axis=0)

            train_acc = Metrics.perClassAccuracy(train_y, train_y_)

            ##############################
            #          Testing           #
            ##############################
            # if (test_save_x is None):
            #     test_save_x, test_save_y = self.dl.genEpochTest()
            x_inputs, test_y = self.dl.genEpochTest()

            test_loss = 0
            n = 0
            test_y_ = np.zeros((len(x_inputs), CTX["FEATURES_OUT"]), dtype=np.float32)
            for batch in range(0, len(x_inputs), CTX["BATCH_SIZE"]):
                sub_test_x = x_inputs[batch:batch+CTX["BATCH_SIZE"]]
                sub_test_y = test_y[batch:batch+CTX["BATCH_SIZE"]]

                sub_loss, sub_output = self.model.compute_loss(reshape(sub_test_x), sub_test_y)

                test_loss += sub_loss
                n += 1
                test_y_[batch:batch+CTX["BATCH_SIZE"]] = sub_output

            test_loss /= n
            test_acc = Metrics.perClassAccuracy(test_y, test_y_)


            # Verbose area
            print()
            print(f"Epoch {ep}/{CTX['EPOCHS']} - train_loss: {train_loss:.4f} - test_loss: {test_loss:.4f} - time: {time.time() - start:.0f}s" , flush=True)
            print("-----------" + "-"*(len(self.dl.yScaler.classes_)*4-1))
            prntC("classes   :",C.BLUE, (C.RESET+"|"+C.BLUE).join([str(int(round(v, 0))).rjust(3, " ") for v in self.dl.yScaler.classes_]))
            print("-----------" + "-"*(len(self.dl.yScaler.classes_)*4-1))
            print("train_acc :", "|".join([str(int(round(v, 0))).rjust(3, " ") for v in train_acc]))
            print("train_nb  :", "|".join([str(int(round(v, 0))).rjust(3, " ") for v in nb_sample_per_class]))
            print("-----------" + "-"*(len(self.dl.yScaler.classes_)*4-1))
            print("test_acc  :", "|".join([str(int(round(v, 0))).rjust(3, " ") for v in test_acc]))
            print("-----------" + "-"*(len(self.dl.yScaler.classes_)*4-1))

            train_acc, test_acc = Metrics.accuracy(train_y, train_y_), Metrics.accuracy(test_y, test_y_)
            print(f"train acc: {train_acc:.1f}")
            print(f"test acc : {test_acc:.1f}")
            print()

            # Save the model loss
            history[0].append(train_loss)
            history[1].append(test_loss)
            history[2].append(train_acc)
            history[3].append(test_acc)
            
            # Log metrics to mlflow
            mlflow.log_metric("train_loss", train_loss, step=ep)
            mlflow.log_metric("test_loss", test_loss, step=ep)
            mlflow.log_metric("epoch", ep, step=ep)

            # Save the model weights
            write("./_Artefact/modelsW/"+self.model.name+"_"+str(ep)+".w", self.model.getVariables())

        # load best model


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


        Metrics.plotLoss(history, history_avg)
        Metrics.plotAccuracy(history, history_avg)

        # #  load back best model
        if (len(history[1]) > 0):
            # find best model epoch with history_avg_accuracy 
            best_i = np.argmax(history_avg[3]) + 1

            print("load best model, epoch : ", best_i, " with Acc : ", history[3][best_i-1], flush=True)
            
            best_variables = load("./_Artefact/modelsW/"+self.model.name+"_"+str(best_i)+".w")
            self.model.setVariables(best_variables)
        else:
            print("WARNING : no history of training has been saved")


        write("./_Artefact/"+self.model.name+".w", self.model.getVariables())
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            write("./_Artefact/"+self.model.name+".xts", self.dl.xTakeOffScaler.getVariables())
        write("./_Artefact/"+self.model.name+".xs", self.dl.xScaler.getVariables())
        write("./_Artefact/"+self.model.name+".min", self.dl.FEATURES_MIN_VALUES)

    def load(self):
        """
        Load the model's weights from the _Artefact folder
        """
        self.model.setVariables(load("./_Artefact/"+self.model.name+".w"))
        # self.model.setVariables(load("./_Artefact/modelsW/CNN_80.w"))
        self.dl.xScaler.setVariables(load("./_Artefact/"+self.model.name+".xs"))
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            self.dl.xTakeOffScaler.setVariables(load("./_Artefact/"+self.model.name+".xts"))
        self.dl.yScaler.setVariables(self.CTX["USED_LABELS"])
        self.dl.FEATURES_MIN_VALUES = load("./_Artefact/"+self.model.name+".min")


    def eval(self):
        """
        Evaluate the model and return metrics


        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """
        CTX = self.CTX
        FOLDER = "./A_Dataset/AircraftClassification/Eval"
        files = os.listdir(FOLDER)
        files = [file for file in files if file.endswith(".csv")]


        nb_classes = self.dl.yScaler.classes_.shape[0]

        global_nb = 0
        global_correct_mean = 0
        global_correct_count = 0
        global_correct_max = 0
        global_ts_confusion_matrix = np.zeros((nb_classes, nb_classes), dtype=int)
        global_confusion_matrix = np.zeros((nb_classes, nb_classes), dtype=int)

        # create a plotting pdf
        pdf = PdfPages("./_Artefact/tmp")

        failed_files = []

        # clear output eval folder
        os.system("rm ./A_Dataset/AircraftClassification/Outputs/Eval/*")

        for i in range(len(files)):
            LEN = 20
            nb = int((i+1)/len(files)*LEN)
            print("EVAL : |", "-"*(nb)+" "*(LEN-nb)+"| "+str(i + 1).rjust(len(str(len(files))), " ") + "/" + str(len(files)), end="\r", flush=True)


            file = files[i]
            x_inputs, y_batches = self.dl.genEval(os.path.join(FOLDER, file))
            if (len(x_inputs) == 0): # skip empty file (no label)
                continue

            start = time.time()
            y_batches_ = np.zeros((len(x_inputs), nb_classes), dtype=np.float32)

            jumps = CTX["BATCH_SIZE"] * CTX["NB_BATCH"]
            for b in range(0, len(x_inputs),jumps):
                x_batch = x_inputs[b:b+jumps]
                pred =  self.model.predict(reshape(x_batch)).numpy()
                y_batches_[b:b+jumps] = pred



            global_ts_confusion_matrix = global_ts_confusion_matrix + Metrics.confusionMatrix(y_batches, y_batches_)

            pred_mean = np.argmax(np.mean(y_batches_, axis=0))
            pred_count = np.argmax(np.bincount(np.argmax(y_batches_, axis=1), minlength=nb_classes))
            pred_max = np.argmax(y_batches_[np.argmax(np.max(y_batches_, axis=1))])
            true = np.argmax(np.mean(y_batches, axis=0))
            

            global_nb += 1
            global_correct_mean += 1 if (pred_mean == true) else 0
            global_correct_count += 1 if (pred_count == true) else 0
            global_correct_max += 1 if (pred_max == true) else 0

            global_confusion_matrix[true, pred_max] += 1



            
            # compute binary (0/1) correct prediction
            correct_predict = np.full((len(x_inputs)), np.nan, dtype=np.float32)
            #   start at history to remove padding
            for t in range(0, len(x_inputs)):
                correct_predict[t] = np.argmax(y_batches_[t]) == np.argmax(y_batches[t])
            # check if A_dataset/output/ doesn't exist, create it
            if not os.path.exists("./A_Dataset/AircraftClassification/Outputs/Eval"):
                os.makedirs("./A_Dataset/AircraftClassification/Outputs/Eval")


            # save the input df + prediction in A_dataset/output/
            df = pd.read_csv(os.path.join("./A_Dataset/AircraftClassification/Eval", file),dtype={'icao24': str})


            if (pred_max != true):
                failed_files.append((file, str(self.dl.yScaler.classes_[true]), str(self.dl.yScaler.classes_[pred_max])))
                track = df["track"].values
                relative_track = track.copy()
                for i in range(1, len(relative_track)):
                    relative_track[i] = angle_diff(track[i], track[i-1])
                relative_track[0] = 0

                y_batches_ = Metrics.inv_sigmoid(y_batches_)


                fig, ax = plotADSB(CTX, self.dl.yScaler.classes_, 
                                   f"{file}    Y : {self.dl.yScaler.classes_[true]} Ŷ : {self.dl.yScaler.classes_[pred_max]}", 
                                   df["timestamp"].values, df['latitude'].values, df['longitude'].values, 
                                   df['groundspeed'].values, df['track'].values, df['vertical_rate'].values, 
                                   df['altitude'].values, df['geoaltitude'].values, 
                                   y_batches_, self.dl.yScaler.classes_[true], [(relative_track, "relative_track")])
                pdf.savefig(fig)
                plt.close(fig)



            if (CTX["INPUT_PADDING"] != "valid"):
                # list all missing timestep that are not in the real dataset

                missing_timestamp = []
                ind = 0
                for t in range(0, len(df["timestamp"])-1):
                    if df["timestamp"][t + 1] != df["timestamp"][t] + 1:
                        nb_missing_timestep = df["timestamp"][t + 1] - df["timestamp"][t] - 1
                        for j in range(nb_missing_timestep):
                            missing_timestamp.append(df["timestamp"][t] + 1 + j - df["timestamp"][0])
                            ind += 1
                    ind += 1

                # remove those timestep from the prediction
                correct_predict = np.delete(correct_predict, missing_timestamp)
                y_batches_ = np.delete(y_batches_, missing_timestamp, axis=0)

            correct_predict = ["" if np.isnan(x) else "True" if x else "False" for x in correct_predict]
            df_y_ = [";".join([str(x) for x in y]) for y in y_batches_]
            df["prediction"] = correct_predict
            df["y_"] = df_y_
            df.to_csv(os.path.join("./A_Dataset/AircraftClassification/Outputs/Eval", file), index=False)
          
            


        pdf.close()
        os.rename("./_Artefact/tmp", "./_Artefact/eval.pdf")


        self.CTX["LABEL_NAMES"] = np.array(self.CTX["LABEL_NAMES"])
        Metrics.plotConusionMatrix("./_Artefact/confusion_matrix.png", global_confusion_matrix, self.CTX["LABEL_NAMES"][self.dl.yScaler.classes_])
        Metrics.plotConusionMatrix("./_Artefact/ts_confusion_matrix.png", global_ts_confusion_matrix, self.CTX["LABEL_NAMES"][self.dl.yScaler.classes_])


        accuracy_per_class = np.diag(global_confusion_matrix) / np.sum(global_confusion_matrix, axis=1)
        accuracy_per_class = np.nan_to_num(accuracy_per_class, nan=0)
        nbSample = np.sum(global_confusion_matrix, axis=1)
        accuracy = np.sum(np.diag(global_confusion_matrix)) / np.sum(global_confusion_matrix)

        print("class              : ", "|".join([str(a).rjust(6, " ") for a in self.dl.yScaler.classes_]))
        print("accuracy per class : ", "|".join([str(round(a * 100)).rjust(6, " ") for a in accuracy_per_class]))
        print("nbSample per class : ", "|".join([str(a).rjust(6, " ") for a in nbSample]))
        print("accuracy : ", accuracy)

        print("global accuracy mean : ", round(global_correct_mean / global_nb*100.0, 1), "(", global_correct_mean, "/", global_nb, ")")
        print("global accuracy count : ", round(global_correct_count / global_nb*100.0, 1), "(", global_correct_count, "/", global_nb, ")")
        print("global accuracy max : ", round(global_correct_max / global_nb*100.0, 1), "(", global_correct_max, "/", global_nb, ")")

        # print files of failed predictions
        print("failed files : ")
        for i in range(len(failed_files)):
            print("\t-",failed_files[i][0], "\tY : ", failed_files[i][1], " Ŷ : ", failed_files[i][2], sep="", flush=True)

        # fail counter
        if os.path.exists("./_Artefact/"+self.model.name+".fails.json"):
            file = open("./_Artefact/"+self.model.name+".fails.json", "r")
            json_ = file.read()
            file.close()
            fails = json.loads(json_)
            # print(fails)
        else:
            fails = {}

        for i in range(len(failed_files)):
            if (failed_files[i][0] not in fails):
                fails[failed_files[i][0]] = {"Y":failed_files[i][1]}
            if (failed_files[i][2] not in fails[failed_files[i][0]]):
                fails[failed_files[i][0]][failed_files[i][2]] = 1
            else:
                fails[failed_files[i][0]][failed_files[i][2]] += 1
        # sort by nb of fails
        fails_counts = {}
        for file in fails:
            fails_counts[file] = 0
            for pred in fails[file]:
                if (pred != "Y"):
                    fails_counts[file] += fails[file][pred]
        fails = {k: v for k, v in sorted(fails.items(), key=lambda item: fails_counts[item[0]], reverse=True)}
        json_ = json.dumps(fails)
        

        file = open("./_Artefact/"+self.model.name+".fails.json", "w")
        file.write(formatJson(json_))


        print("", flush=True)


        # self.eval_alterated()

        return {
            "accuracy": accuracy, 
            "mean accuracy":global_correct_mean / global_nb,
            "count accuracy":global_correct_count / global_nb,
            "max accuracy":global_correct_max / global_nb,
        }

