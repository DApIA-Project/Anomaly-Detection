
# MDSM : Mean Dense Simple Model

import _Utils.mlflow as mlflow
import _Utils.Metrics as Metrics
import _Utils.Color as Color


from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.AircraftClassification.RawDataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt



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
        self.model.visualize()

        self.dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")
        

        # If "_Artefacts/" folder doesn't exist, create it.
        if not os.path.exists("./_Artefact"):
            os.makedirs("./_Artefact")


    def train(self):
        """
        Train the model.
        Plot the loss curves into Artefacts folder.
        """
        
        history = [[], []]

        best_variables = None
        best_loss= 10000000

        for ep in range(1, self.CTX["EPOCHS"] + 1):
            ##############################
            #         Training           #
            ##############################
            batch_x, batch_y = self.dl.genEpochTrain(self.CTX["NB_BATCH"], self.CTX["BATCH_SIZE"])
            # print(batch_x.shape)

            train_loss = 0
            train_y_ = []
            train_y = []
            for batch in range(len(batch_x)):
                loss, output = self.model.training_step(batch_x[batch], batch_y[batch])
                train_loss += loss

                train_y_.append(output)
                train_y.append(batch_y[batch])

            train_loss /= len(batch_x)
            train_y_ = np.concatenate(train_y_, axis=0)
            train_y = np.concatenate(train_y, axis=0)


            train_acc = Metrics.perClassAccuracy(train_y, train_y_)

            ##############################
            #          Testing           #
            ##############################
            test_x, test_y = self.dl.genEpochTest()
            test_loss, test_y_ = self.model.compute_loss(test_x, test_y)

            test_acc = Metrics.perClassAccuracy(test_y, test_y_)


            # Verbose area
            print()
            print(f"Epoch {ep}/{self.CTX['EPOCHS']} - train_loss: {train_loss:.4f} - test_loss: {test_loss:.4f}", flush=True)
            
            print("classes  : ", "|".join([str(int(round(v, 0))).zfill(3) for v in self.dl.yScaler.classes_]))
            print("train_acc: ", "|".join([str(int(round(v, 0))).zfill(3) for v in train_acc]))
            print("test_acc : ", "|".join([str(int(round(v, 0))).zfill(3) for v in test_acc]))

            print("train acc: ", Metrics.accuracy(train_y, train_y_))
            print("test acc : ", Metrics.accuracy(test_y, test_y_))

            if test_loss < best_loss:
                best_loss = test_loss
                best_variables = self.model.getVariables()

            # Save the model loss
            history[0].append(train_loss)
            history[1].append(test_loss)
            
            # Log metrics to mlflow
            mlflow.log_metric("train_loss", train_loss, step=ep)
            mlflow.log_metric("test_loss", test_loss, step=ep)
            mlflow.log_metric("epoch", ep, step=ep)

        # load best model


        # Compute the moving average of the loss for a better visualization
        history_avg = [[], []]
        window_len = 5
        for i in range(len(history[0])):
            min_ = max(0, i - window_len)
            max_ = min(len(history[0]), i + window_len)
            history_avg[0].append(np.mean(history[0][min_:max_]))
            history_avg[1].append(np.mean(history[1][min_:max_]))


        # Plot the loss curves
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.grid()
        ax.plot(np.array(history[0]) * 100.0, c="tab:blue", linewidth=0.5)
        ax.plot(np.array(history[1]) * 100.0, c="tab:orange", linewidth=0.5)
        ax.plot(np.array(history_avg[0]) * 100.0, c="tab:blue", ls="--", label="train loss")
        ax.plot(np.array(history_avg[1]) * 100.0, c="tab:orange", ls="--", label="test loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (%)")
        ax.legend()
        fig.savefig("./_Artefact/loss.png")

        
        # load back best model
        if (len(history[1]) > 0):
            print("load best model, epoch : ", np.argmin(history[1]) + 1, " with loss : ", np.min(history[1]), flush=True)
            self.model.setVariables(best_variables)
        else:
            print("WARNING : no history of training has been saved")


        # save weights
        w = open("./_Artefact/"+self.model.name+".w", "w")
        w.write(self.model.getVariables())

        # save x scaler
        w = open("./_Artefact/"+self.model.name+".xs", "w")
        w.write(self.dl.xScaler.getVariables())

        # save y scaler
        w = open("./_Artefact/"+self.model.name+".ys", "w")
        w.write(self.dl.yScaler.getVariables())


    def eval(self):
        """
        Evaluate the model and return metrics


        Returns:
        --------

        metrics : dict
            The metrics dictionary of the model's performance
        """


        x_batches, y_batches, associated_files = self.dl.genEval("./A_Dataset/AircraftClassification/Eval")
        nb_classes = self.dl.yScaler.classes_.shape[0]

        global_confusion_matrix = np.zeros((nb_classes, nb_classes), dtype=int)

        i = 0

        global_nb = 0
        global_correct_mean = 0
        global_correct_count = 0
        global_correct_max = 0


        failed_files = []

        while i < len(x_batches):

            # make a batch with all the time windows of the same file
            batch_length = 1
            i += 1
            while i < len(x_batches) and (i==0 or associated_files[i-1] == associated_files[i]):
                batch_length += 1
                i += 1


            # get the batch
            file = associated_files[i-1]
            batch_x = x_batches[i-batch_length:i]
            batch_y = y_batches[i-batch_length:i]

            # predict with sub batches to avoid memory issues
            batch_y_ = np.zeros((len(batch_x), nb_classes), dtype=np.float32)
            MAX_BATCH_SIZE = self.CTX["BATCH_SIZE"]
            for b in range(0, len(batch_x), MAX_BATCH_SIZE):
                batch_y_[b:b+MAX_BATCH_SIZE] = self.model.predict(batch_x[b:b+MAX_BATCH_SIZE]).numpy()
            
            global_confusion_matrix = global_confusion_matrix + Metrics.confusionMatrix(batch_y, batch_y_)


            global_pred_mean = np.argmax(np.mean(batch_y_, axis=0))
            global_pred_count = np.argmax(np.bincount(np.argmax(batch_y_, axis=1), minlength=nb_classes))
            global_pred_max = np.argmax(batch_y_[np.argmax(np.max(batch_y_, axis=1))])


            global_true = np.argmax(np.mean(batch_y, axis=0))
            

            global_nb += 1
            global_correct_mean += 1 if (global_pred_mean == global_true) else 0
            global_correct_count += 1 if (global_pred_count == global_true) else 0
            global_correct_max += 1 if (global_pred_max == global_true) else 0

            
            if (global_pred_max != global_true):
                failed_files.append((file, self.dl.yScaler.classes_[global_true]))

            
            # compute binary (0/1) correct prediction
            correct_predict = np.full((len(batch_x)), np.nan, dtype=np.float32)
            #   start at history to remove padding
            for t in range(0, len(batch_x)):
                correct_predict[t] = np.argmax(batch_y_[t]) == np.argmax(batch_y[t])
            # check if A_dataset/output/ doesn't exist, create it
            if not os.path.exists("./A_Dataset/AircraftClassification/Outputs/Eval"):
                os.makedirs("./A_Dataset/AircraftClassification/Outputs/Eval")


            # save the input df + prediction in A_dataset/output/
            df = pd.read_csv(os.path.join("./A_Dataset/AircraftClassification/Eval", file))

            if (self.CTX["PAD_MISSING_INPUT_LEN"]):
                # list all missing timestep in df["timestamp"] (sec)
                print("pad missing timesteps for ", file, " ...")
                missing_timestep_i = []
                ind = 0
                for t in range(1, len(df["timestamp"])):
                    if df["timestamp"][t - 1] != df["timestamp"][t] - 1:
                        nb_missing_timestep = df["timestamp"][t] - df["timestamp"][t - 1] - 1
                        for _ in range(nb_missing_timestep):
                            missing_timestep_i.append(ind)
                            ind += 1
                    ind += 1

                correct_predict = np.delete(correct_predict, missing_timestep_i)
                batch_y_ = np.delete(batch_y_, missing_timestep_i, axis=0)


            correct_predict = ["" if np.isnan(x) else "True" if x else "False" for x in correct_predict]
            df_y_ = [";".join([str(x) for x in y]) for y in batch_y_]
            df["prediction"] = correct_predict
            df["y_"] = df_y_
            df.to_csv(os.path.join("./A_Dataset/AircraftClassification/Outputs/Eval", file), index=False)


        print(global_confusion_matrix)
        accuracy_per_class = np.diag(global_confusion_matrix) / np.sum(global_confusion_matrix, axis=1)
        nbSample = np.sum(global_confusion_matrix, axis=1)
        accuracy = np.sum(np.diag(global_confusion_matrix)) / np.sum(global_confusion_matrix)

    
        print("accuracy per class : ", accuracy_per_class)
        print("nbSample per class : ", nbSample)
        print("accuracy : ", accuracy)

        print("global accuracy mean : ", global_correct_mean / global_nb, "(", global_correct_mean, "/", global_nb, ")")
        print("global accuracy count : ", global_correct_count / global_nb, "(", global_correct_count, "/", global_nb, ")")
        print("global accuracy max : ", global_correct_max / global_nb, "(", global_correct_max, "/", global_nb, ")")

        # print files of failed predictions
        print("failed files : ")
        for i in range(len(failed_files)):
            print("\t-",failed_files[i][0], " label : "+Color.CYAN, self.CTX["LABEL_NAMES"][failed_files[i][1]], "("+str(failed_files[i][1])+")")

        print("", flush=True)

        self.eval_alterated()

        return {
            "accuracy": accuracy, 
            "mean accuracy":global_correct_mean / global_nb,
            "count accuracy":global_correct_count / global_nb,
            "max accuracy":global_correct_max / global_nb,
        }





    def eval_alterated(self):

        # get folder list in A_Dataset/AircraftClassification/Alterations/
        alterations = os.listdir("./A_Dataset/AircraftClassification/Alterations/")

        # keep only folder
        alterations = [x for x in alterations if os.path.isdir(os.path.join("./A_Dataset/AircraftClassification/Alterations/", x))]

        # for each folder
        for alteration in alterations:
            
            print(alteration, " : ", flush=True)
            x_batches, y_batches, associated_files = self.dl.genEval(os.path.join("./A_Dataset/AircraftClassification/Alterations/", alteration))
            nb_classes = self.dl.yScaler.classes_.shape[0]


            i = 0
            while i < len(x_batches):

                # make a batch with all the time windows of the same file
                batch_length = 1
                i += 1
                while i < len(x_batches) and (associated_files[i-1] == associated_files[i]):
                    batch_length += 1
                    i += 1

                # get the batch
                file = associated_files[i-1]
                batch_x = x_batches[i-batch_length:i]
                batch_y = y_batches[i-batch_length:i]


                # predict with sub batches to avoid memory issues
                batch_y_ = np.zeros((len(batch_x), nb_classes), dtype=np.float32)
                MAX_BATCH_SIZE = self.CTX["BATCH_SIZE"]
                for b in range(0, len(batch_x), MAX_BATCH_SIZE):
                    batch_y_[b:b+MAX_BATCH_SIZE] = self.model.predict(batch_x[b:b+MAX_BATCH_SIZE]).numpy()
                

                # compute binary (0/1) correct prediction
                correct_predict = np.full((len(batch_x)), np.nan, dtype=np.float32)
                #   start at history to remove padding
                for t in range(0, len(batch_x)):
                    correct_predict[t] = np.argmax(batch_y_[t]) == np.argmax(batch_y[t])
                # check if A_dataset/output/ doesn't exist, create it
                if not os.path.exists("./A_Dataset/AircraftClassification/Outputs/Alterations/"+alteration):
                    os.makedirs("./A_Dataset/AircraftClassification/Outputs/Alterations/"+alteration)

                # save the input df + prediction in A_dataset/output/
                df = pd.read_csv(os.path.join("./A_Dataset/AircraftClassification/Alterations/"+alteration, file))

                print(batch_y_.shape, batch_y.shape, flush=True)

                if (self.CTX["PAD_MISSING_INPUT_LEN"]):
                    # list all missing timestep in df["timestamp"] (sec)
                    print("pad missing timesteps for ", file, " ...")
                    missing_timestep_i = []
                    ind = 0
                    for t in range(1, len(df["timestamp"])):
                        if df["timestamp"][t - 1] != df["timestamp"][t] - 1:
                            nb_missing_timestep = df["timestamp"][t] - df["timestamp"][t - 1] - 1
                            for _ in range(nb_missing_timestep):
                                missing_timestep_i.append(ind)
                                ind += 1
                        ind += 1

                    # remove missing timestep from correct_predict
                    correct_predict = np.delete(correct_predict, missing_timestep_i)
                    batch_y_ = np.delete(batch_y_, missing_timestep_i, axis=0)


                correct_predict = ["" if np.isnan(x) else "True" if x else "False" for x in correct_predict]
                df_y_ = [";".join([str(x) for x in y]) for y in batch_y_]
                df["prediction"] = correct_predict
                df["y_"] = df_y_
                df.to_csv(os.path.join("./A_Dataset/AircraftClassification/Outputs/Alterations/"+alteration, file), index=False)

                global_true = np.argmax(np.mean(batch_y, axis=0))
                global_pred_mean = np.argmax(np.mean(batch_y_, axis=0))
                global_pred_count = np.argmax(np.bincount(np.argmax(batch_y_, axis=1), minlength=nb_classes))
                global_pred_max = np.argmax(batch_y_[np.argmax(np.max(batch_y_, axis=1))])

                print("File : ", file)
                print("Expected class : ", self.dl.yScaler.classes_[global_true])
                print("Ŷ: ", np.mean(batch_y_, axis=0))
                print("Label : mean :", self.dl.yScaler.classes_[global_pred_mean], "   count :", self.dl.yScaler.classes_[global_pred_count], "   max :", self.dl.yScaler.classes_[global_pred_max]) 

                print("Annomaly ?")
                print("mean :", (global_pred_mean != global_true), "   count :", (global_pred_count != global_true), "   max :", (global_pred_max != global_true))
                print()

    
    






    





            
            
