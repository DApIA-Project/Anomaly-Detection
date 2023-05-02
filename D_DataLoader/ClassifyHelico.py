
import pandas as pd
import numpy as np
from _Utils.MinMaxScaler3D import MinMaxScaler3D
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader
import os
import math
# MultiLabelBinirazer
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

# managing the data preprocessing
class DataLoader(AbstractDataLoader):
    """
    Class for managing the data processing

    Attributes :
    ------------

    xScaler : Scaler
    yScaler : Scaler

    x : np.array
        represent the input data
    y : np.array
        represent the output data

    TEST_SIZE : float
        represent the size of the test set
    x_train : np.array
    y_train : np.array
    x_test : np.array
    y_test : np.array
    


    Methods :
    ---------
    

    __load_dataset__(path):
        return from a path the x and y dataset in the format of your choice
        define here the preprocessing you want to do
        no need to call directly this method, use instead __get_dataset__ with a cache

    __init__(self, CTX, path):
        Constructor of the class, generate x_train, y_train, x_test, y_test
        3 tasks :
            1. load dataset
            2. create and fit the scalers on x and y
            3. split the data into train and test

    genEpochTrain(nb_batch, batch_size):
        return x, y batch list in format [nb_batch, batch_size, ...]
        The output must be directly usable by the model for the training
    
    genEpochTest():
        return x, y batch list in format [nb_row, ...]
        The output must be directly usable by the model for the testing
    """


    @staticmethod
    def __load_dataset__(CTX, path):
        """
        preprocess data
        """


        # path is the folder containing the data
        # but the labels are in the parent folder


        labels_file = os.path.join(os.path.dirname(path), "labels.csv")
        labels = pd.read_csv(labels_file, sep=",", header=None, dtype={"callsign":str, "icao24":str})
        labels.columns = ["callsign", "icao24", "label"]

        labels = labels.fillna("NULL")

        # load the data
        data_files = os.listdir(path)
        x = []
        y = []



        for file in data_files:
            df = pd.read_csv(os.path.join(path, file), sep=",",dtype={"callsign":str, "icao24":str})

            if (len(df) < CTX["HISTORY"]):
                print(df["callsign"][0], df["icao24"][0], "is too short")
                continue
            
            # remplace NaN by NULL for callsign
            df["callsign"] = df["callsign"].fillna("NULL")


            callsign = df["callsign"].iloc[0]
            icao24 = df["icao24"].iloc[0]


            # get the label
            label = labels[(labels["callsign"] == callsign)
                         & (labels["icao24"]   == icao24  )]["label"]
            
            if (len(label) == 0):
                print("no label for", callsign, icao24)
                print(df)
                continue

            label = label.iloc[0]
            # only keep airplane, small aircraft and helicopter
            if (label != 1 and label != 3 and label != 5):
                continue

            x_df = df[CTX["USED_FEATURES"]]

            
            
            # change all boolean to int
            for col in x_df.columns:
                if (x_df[col].dtype == bool):
                    x_df[col] = x_df[col].astype(int)
                
            x_df = x_df.fillna(-1)
            x_df = x_df.to_numpy()

            x.append(x_df)
            y.append(label)

        return x, y


    def __init__(self, CTX, path) -> None:    
        self.CTX = CTX
        self.x, self.y = self.__get_dataset__(path)

        # create the scaler
        self.xScaler = MinMaxScaler3D()
        self.yScaler = LabelBinarizer()



        # fit the scaler
        self.x = self.xScaler.fit_transform(self.x)
        self.y = self.yScaler.fit_transform(self.y)

        self.y = np.array(self.y, dtype=np.float32)


        # split the data
        # concat
        ratio = self.CTX["TEST_SIZE"]
        split_index = int(len(self.x) * (1 - ratio))
        self.x_train = self.x[:split_index]
        self.y_train = self.y[:split_index]
        self.x_test = self.x[split_index:]
        self.y_test = self.y[split_index:]



    def genEpochTrain(self, nb_batch, batch_size):
        """
        Generate the x train and y train batches.
        The returned format is usually [nb_batch, batch_size, ...]
        But it can be different depending on you're model implementation
        The output must be directly usable by the model for the training

        Called between each epoch by the trainer
        """

        # split the data into batches
        x_batches = []
        y_batches = []
        for n in range(nb_batch):

            x_batches.append([])
            y_batches.append([])

            for b in range(batch_size):
                # standardize the ratio of each class
                label = np.random.randint(0, self.yScaler.classes_.shape[0])
                i = -1
                while i == -1 or self.y_train[i, label] != 1:
                    i = np.random.randint(0, len(self.x_train))
                
                t = np.random.randint(0, len(self.x_train[i]) - self.CTX["HISTORY"])

                x_batches[n].append(self.x_train[i][t:t+self.CTX["HISTORY"]])
                y_batches[n].append(self.y_train[i])

        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)

        # check if there is nan in the data
        if (np.isnan(x_batches).any()):
            print("x_batches contains nan")
            exit(1)
        if (np.isnan(y_batches).any()):
            print("y_batches contains nan")
            exit(1)


        return x_batches, y_batches


    def genEpochTest(self):
        """
        Generate the x test and y test dataset.
        The returned format is usually [nb_row, ...]
        But it can be different depending on you're model implementation
        The output must be directly usable by the model for the testing

        Called between each epoch by the trainer
        """

                # split the data into batches
        x_batches = []
        y_batches = []
        nb_data = int(self.CTX["BATCH_SIZE"] * self.CTX["NB_BATCH"] * self.CTX["TEST_SIZE"])
        for _ in range(nb_data):

            # standardize the ratio of each class
            label = np.random.randint(0, self.yScaler.classes_.shape[0])
            i = -1
            while i == -1 or self.y_train[i, label] != 1:
                i = np.random.randint(0, len(self.x_train))

            t = np.random.randint(0, len(self.x_train[i]) - self.CTX["HISTORY"])

            x_batches.append(self.x_train[i][t:t+self.CTX["HISTORY"]])
            y_batches.append(self.y_train[i])

        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)

        return x_batches, y_batches
    

    def genEval(self, path):
        """
        Generate the x eval and y eval batches.

        called by the trainer at the end, to get the final evaluation in the real world condition
        """


        x, y = self.__load_dataset__(self.CTX, path)
        x = self.xScaler.transform(x)
        y = self.yScaler.transform(y)
        y = np.array(y, dtype=np.float32)


        x_eval = np.zeros((0, self.CTX["HISTORY"], len(self.CTX["USED_FEATURES"])))
        y_eval = np.zeros((0, len(self.yScaler.classes_)))


        for s in range(len(x)):

            # split x[s] in multiple self.CTX["HISTORY"] sub arrays
            x_sub = np.zeros((0, self.CTX["HISTORY"], len(self.CTX["USED_FEATURES"])))
            
            maxi = len(x[s]) // self.CTX["HISTORY"] * self.CTX["HISTORY"]
            for i in range(0, maxi, self.CTX["HISTORY"]):
                x_sub = np.concatenate((x_sub, [x[s][i:i+self.CTX["HISTORY"]]]), axis=0)

            
            # repeat y[s] len(x_sub) times
            y_sub = np.repeat([y[s]], len(x_sub), axis=0)


            x_eval = np.concatenate((x_eval, x_sub), axis=0)
            y_eval = np.concatenate((y_eval, y_sub), axis=0)
                

        return x_eval, y_eval
             


