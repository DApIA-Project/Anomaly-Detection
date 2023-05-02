
# MDSM : Mean Dense Simple Model

from B_Model.AbstractModel import Model
from D_DataLoader.ClassifyHelico import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer


import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import mlflow


def RMSE(y_true, y_pred):
    mse = 0
    for i in range(len(y_true)):
        if y_true[i] > 0:
            mse += (y_true[i] - y_pred[i])**2
    return np.sqrt(mse/y_true.shape[0])


class Trainer(AbstractTrainer):
    """"
    Managing the training of the model.

    Attributes :
    ------------

        CTX : dict
            The hyperparameters context

        dl : DataLoader
            The data loader corresponding to the problem
            we want to solve

        model : Model
            The model we want to train   

    Methods :
    ---------

    __init__(CTX, model):
        Constructor

    run():
        Shortcut that train the model completely
        and then give the evaluation of the performance
        (code in AbstractTrainer)

    train():
        Train the model and return the history of the training

    eval():
        Evaluate the model and return metrics
    
    """

    def __init__(self, CTX:dict, model:"type[Model]", iteration:int = 1):
        self.CTX = CTX

        self.model:Model = model(CTX)
        self.model.visualize()

        self.dl = DataLoader(CTX, "./A_Dataset/ClassifyHelico/Train")
        
        self.iteration = iteration



    def train(self):
        """
        train the model and return the history of the training

            Returns:
                list[float, float]: list of [train_loss, test_loss]
        """
        
        history = [[], []]

        print(self.CTX["EPOCHS"] )

        for ep in range(1, self.CTX["EPOCHS"] + 1):
            # training
            batch_x, batch_y = self.dl.genEpochTrain(self.CTX["NB_BATCH"], self.CTX["BATCH_SIZE"])


            train_loss = 0
            for batch in range(len(batch_x)):
                loss, output = self.model.training_step(batch_x[batch], batch_y[batch])
                train_loss += loss
            train_loss /= len(batch_x)

            # testing
            x_test, y_test = self.dl.genEpochTest()
            test_loss, y_pred = self.model.compute_loss(x_test, y_test)

            # print the loss
            print(f"Epoch {ep}/{self.CTX['EPOCHS']} - train_loss: {train_loss:.4f} - test_loss: {test_loss:.4f}", flush=True)


            history[0].append(train_loss)
            history[1].append(test_loss)


            mlflow.log_metric("train_loss-"+str(self.iteration), train_loss, step=ep)
            mlflow.log_metric("test_loss-"+str(self.iteration), test_loss, step=ep)

        history_avg = [[], []]
        window_len = 5
        for i in range(len(history[0]) - window_len):
            min_ = max(0, i - window_len)
            max_ = min(len(history[0]), i + window_len)
            history_avg[0].append(np.mean(history[0][min_:max_]))
            history_avg[1].append(np.mean(history[1][min_:max_]))


        # plot the loss curve to Output_artefacts/loss.png
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # # show grid
        ax.grid()
        ax.plot(np.array(history[0]) * 100.0, c="tab:blue", linewidth=0.5)
        ax.plot(np.array(history[1]) * 100.0, c="tab:orange", linewidth=0.5)
        ax.plot(np.array(history_avg[0]) * 100.0, c="tab:blue", ls="--", label="train loss")
        ax.plot(np.array(history_avg[1]) * 100.0, c="tab:orange", ls="--", label="test loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (%)")
        ax.legend()
        fig.savefig("./Output_artefact/loss.png")


        # evaluate the modelnb_units
        return history

    def eval(self):
        """
        test the model in REAL CONDITIONS and return a score linked to the real condition (not a loss !!!!)

            Returns:
                dict[str, float]: the final metrics of model evaluation
        """


        batch_x, batch_y = self.dl.genEval("./A_Dataset/ClassifyHelico/Eval")
        output = self.model.predict(batch_x)

        output = output.numpy()



        # convert the output to a binary vector max = 1 and the others = 0
        indx = np.argmax(output, axis=1)
        output = np.zeros(output.shape)
        for i in range(len(indx)):
            output[i][indx[i]] = 1

        # accuracy
        accuracy = 0
        for i in range(len(output)):
            # check if the output is the same as the batch_y
            if np.array_equal(output[i], batch_y[i]):
                accuracy += 1
        accuracy /= len(output)
        accuracy *= 100

        return {"accuracy": accuracy}



    
    






    





            
            
