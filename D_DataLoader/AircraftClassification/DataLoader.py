 
import pandas as pd
import numpy as np
from _Utils.MinMaxScaler3D import MinMaxScaler3D
from _Utils.SparceLabelBinarizer import SparceLabelBinarizer
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader
import os
import math
# MultiLabelBinirazer
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

from PIL import Image

import D_DataLoader.AircraftClassification.Utils as U

import time

from _Utils.Metrics import computeTimeserieVarienceRate
from _Utils import Color


__icao_db__ = None

# managing the data preprocessing
class DataLoader(AbstractDataLoader):
    """
    Data manager for aircraft classification
    loads ADS-B reccords and format them for training
    

    Pipeline :
    ----------

    1. Load all the flights in the dataset folder
        one flight = one list of adsb features. (__load_dataset__)

    2. Preprocess flights globaly (__load_dataset__)

    3. Split the dataset into train and test (__init__)

    4. Split training batches (__genEpochTrain__, __genEpochTest__)

    5. Preprocess batches (__genEpochTrain__, __genEpochTest__)

    6. Scale batches


    Parameters :
    ------------

    CTX : dict
        The hyperparameters context

    path : str
        The path to the dataset

        
    Attributes :
    ------------

    xScaler: Scaler
        Scaler for the input data

    yScaler: Scaler
        Scaler for the output data

    x: np.array
        The input data

    y: np.array
        The associated output desired to be predicted

        
    x_train: np.array
        isolated x train dataset

    y_train: np.array
        isolated y train dataset

    x_test: np.array
        isolated x test dataset

    y_test: np.array
        isolated y test dataset
    

    Methods :
    ---------

    static __load_dataset__(CTX, path): x, y
        Read all flights into the defined folder and do some global preprocessing
        as filtering interisting variables, computing some features (eg: vectorial speed repr)

        WARNING :
        This function is generally heavy, and if you want to make
        several training on the same dataset, USE the __get_dataset__ method
        wich save the dataset on the first call and return it on the next calls

        For evaluation, generally the dataset
        you want to use is independant. Hence, you 
        can use the __load_dataset__ method directly on the Eval folder

    __get_dataset__(path): x, y (Inherited)
        Return dataset with caching
        it will save the dataset on the first call and return it on the next calls
        so you MUST ONLY use it on one dataset (generally training dataset)
        

    genEpochTrain(nb_batches, batch_size):
        Generate the x and y input, directly usable by the model.
        Pick randoms flights from train sub-dataset, and takes a
        somes fragements of it to compose batches
    
    genEpochTest():
        Generate the x and y test.
        Pick randoms flights from train sub-dataset, and takes a
        somes fragements of it to compose batches

    genEval(path):
        Load evaluation flights in the folder of desired path.
        Preprocess them same way as training flights, keep the full
        sliding window along the whole flight, and finally
        it keep a trace of the orriginal flight associated with each
        fragment of sliding window to be able to compute the accuracy
        and the final label for the complete flight
    """

        


    @staticmethod
    def __load_dataset__(CTX, path):
        """
        Read all flights into the defined folder and do some global preprocessing
        as filtering interisting variables, computing some features (eg: vectorial speed repr)

        Parameters:
        -----------

        CTX: dict
            The hyperparameters context

        path: str
            The path to the dataset

        Returns:
        --------
        x, y: list(np.array) 
            The input and output data.
            We use list because time series lenght is 
            variable because each flight has a different 
            duration.
        """


        data_files = os.listdir(path)
        data_files = [f for f in data_files if f.endswith(".csv")]
        x = []
        y = []

        print("Loading dataset :")

        # Read each file
        for f in range(len(data_files)):
            file = data_files[f]
            # set time as index
            df = pd.read_csv(os.path.join(path, file), sep=",",dtype={"callsign":str, "icao24":str})

            array = U.dfToFeatures(df, CTX)

            # Get the aircraft right label for his imatriculation
            icao24 = df["icao24"].iloc[0]
            callsign = df["callsign"].iloc[0]
            label = U.getLabel(CTX, icao24, callsign)
            if (label == 0):
                continue

            # Add the flight to the dataset
            x.append(array)
            y.append(label)

            if (f % 20 == (len(data_files)-1) % 20):
                done_20 = int(((f+1)/len(data_files)*20))
                print("\r|"+done_20*"="+(20-done_20)*" "+f"| {(f+1)}/{len(data_files)}", end=" "*20)
        print("\n", flush=True)
        return x, y
    


    def __init__(self, CTX, path) -> None:    
        self.CTX = CTX

        
        if (CTX["EPOCHS"]):
            self.x, self.y = self.__get_dataset__(path)
        else:
            self.x, self.y = [], []

        print(self.y)

        # Create the scalers
        self.xScaler = MinMaxScaler3D()
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler = MinMaxScaler3D()
        self.yScaler = SparceLabelBinarizer()


        # Fit the y scaler
        # x scaler will be fitted later after batch preprocessing
        if (CTX["EPOCHS"]):
            self.y = self.yScaler.fit_transform(self.y)
            self.y = np.array(self.y, dtype=np.float32)



        # Split the dataset into train and test according to the ratio in context
        ratio = self.CTX["TEST_RATIO"]
        split_index = int(len(self.x) * (1 - ratio))
        self.x_train = self.x[:split_index]
        self.y_train = self.y[:split_index]
        self.x_test = self.x[split_index:]
        self.y_test = self.y[split_index:]

        # self.x_test = self.x_train.copy()
        # self.y_test =  self.y_train.copy()

        print("Train dataset size :", len(self.x_train))
        print("Test dataset size :", len(self.x_test))

        # fit the scalers and define the min values
        self.FEATURES_MIN_VALUES = np.full((CTX["FEATURES_IN"],), np.nan)
        if (CTX["EPOCHS"]):
            self.genEpochTrain(1, 4096)

        print("="*100)



    
    


    def genEpochTrain(self, nb_batches, batch_size):
        """
        Generate the x and y input, directly usable by the model.
        Pick randoms flights from train sub-dataset, and takes a
        somes fragements of the same flight in a defined area to compose a batch.

        Called between each epoch by the trainer
        """

        CTX = self.CTX
        LON_I = self.CTX["FEATURE_MAP"]["longitude"]
        LAT_I = self.CTX["FEATURE_MAP"]["latitude"]
        ALT_I = self.CTX["FEATURE_MAP"]["altitude"]
        GEO_I = self.CTX["FEATURE_MAP"]["geoaltitude"]

        # Allocate memory for the batches
        x_batches = np.zeros((nb_batches * batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
        y_batches = np.zeros((nb_batches * batch_size, self.yScaler.classes_.shape[0]))
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = np.zeros((nb_batches * batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
        if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = np.zeros((nb_batches * batch_size, self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"],3), dtype=np.float32)

        for n in range(len(x_batches)):

            # Pick a random label
            label_i = np.random.randint(0, self.yScaler.classes_.shape[0])
            flight_i, t = U.pick_an_interesting_aircraft(CTX, self.x_train, self.y_train, label_i)
                    
            # compute the bounds of the fragment
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
            
            # shift to always have the last timestep as part of the fragment !!
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
            # build the batch

            
            x_batch = self.x_train[flight_i][start+shift:end:CTX["DILATION_RATE"]]
            x_batches[n, :pad_lenght] = self.FEATURES_MIN_VALUES
            x_batches[n, pad_lenght:] = x_batch


            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if (self.x_train[flight_i][0,ALT_I] > 1000 or self.x_train[flight_i][0,GEO_I] > 1000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_MIN_VALUES)
                else:
                    takeoff = self.x_train[flight_i][start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[n, :pad_lenght] = self.FEATURES_MIN_VALUES
                x_batches_takeoff[n, pad_lenght:] = takeoff



            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = x_batches[n, -1, LAT_I], x_batches[n, -1, LON_I]
                x_batches_map[n] = U.genMap(lat, lon, self.CTX["IMG_SIZE"]) / 255.0

            # get label
            y_batches[n] = self.y_train[flight_i]

        # fit the min values before preprocessing
        if not(self.xScaler.isFitted()):
            self.FEATURES_MIN_VALUES = np.nanmin(x_batches, axis=(0,1))
   

        # preprocess the batches
        for n in range(len(x_batches)):
            x_batches[n] = U.batchPreProcess(CTX, x_batches[n], CTX["RELATIVE_POSITION"], CTX["RELATIVE_HEADING"], CTX["RANDOM_HEADING"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[n] = U.batchPreProcess(CTX, x_batches_takeoff[n])

  
        # fit the scaler on the first epoch
        if not(self.xScaler.isFitted()):
            self.xScaler.fit(x_batches)
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler.fit(x_batches_takeoff)

        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)



        # noise the data and the output for a more continuous probability output (avoid only 1 and 0 output (binary))
        for i in range(len(x_batches)):
            # TODO : add noise include the takeoff context
            x_batches[i], y_batches[i] = U.add_noise(x_batches[i], y_batches[i], CTX["TRAINING_NOISE"])

        # Reshape the data into [nb_batches, batch_size, timestep, features]
        x_batches = x_batches.reshape(nb_batches, batch_size, self.CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        if CTX["ADD_TAKE_OFF_CONTEXT"]: x_batches_takeoff = x_batches_takeoff.reshape(nb_batches, batch_size, self.CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        if CTX["ADD_MAP_CONTEXT"]: x_batches_map = x_batches_map.reshape(nb_batches, batch_size, self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)
        y_batches = y_batches.reshape(nb_batches, batch_size, self.yScaler.classes_.shape[0])


        x_inputs = []
        for i in range(nb_batches):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            x_inputs.append(x_input)

        return x_inputs, y_batches


    def genEpochTest(self):
        """
        Generate the x and y test.
        Pick randoms flights from train sub-dataset, and takes a
        somes fragements of it to compose batches

        Called at the end of each epoch by the trainer

        Returns:
        -------

        x_batches, y_batches: np.array
        """
        
        CTX = self.CTX
        NB_BATCHES = int(CTX["BATCH_SIZE"] * CTX["NB_BATCH"] * CTX["TEST_RATIO"])
        LON_I = self.CTX["FEATURE_MAP"]["longitude"]
        LAT_I = self.CTX["FEATURE_MAP"]["latitude"]
        ALT_I = self.CTX["FEATURE_MAP"]["altitude"]
        GEO_I = self.CTX["FEATURE_MAP"]["geoaltitude"]

        # Allocate memory for the batches
        x_batches = np.zeros((NB_BATCHES, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
        y_batches = np.zeros((NB_BATCHES, self.yScaler.classes_.shape[0]))
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = np.zeros((NB_BATCHES, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
        if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = np.zeros((NB_BATCHES, self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"],3), dtype=np.float32)

        for n in range(len(x_batches)):

            # Pick a random label
            label_i = np.random.randint(0, self.yScaler.classes_.shape[0])
            flight_i, t = U.pick_an_interesting_aircraft(CTX, self.x_test, self.y_test, label_i)
                    
            # compute the bounds of the fragment
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
            
            # shift to always have the last timestep as part of the fragment !!
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
            # build the batch

            x_batch = self.x_test[flight_i][start+shift:end:CTX["DILATION_RATE"]]
            x_batches[n, :pad_lenght] = self.FEATURES_MIN_VALUES
            x_batches[n, pad_lenght:] = x_batch


            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if (self.x_test[flight_i][0,ALT_I] > 1000 or self.x_test[flight_i][0,GEO_I] > 1000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_MIN_VALUES)
                else:
                    takeoff = self.x_test[flight_i][start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[n, :pad_lenght] = self.FEATURES_MIN_VALUES
                x_batches_takeoff[n, pad_lenght:] = takeoff
                

            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = x_batches[n, -1, LAT_I], x_batches[n, -1, LON_I]
                x_batches_map[n] = U.genMap(lat, lon, self.CTX["IMG_SIZE"]) / 255.0

            # get label
            y_batches[n] = self.y_test[flight_i]
        
        # preprocess the batches
        for n in range(len(x_batches)):
            x_batches[n] = U.batchPreProcess(CTX, x_batches[n], CTX["RELATIVE_POSITION"], CTX["RELATIVE_HEADING"], CTX["RANDOM_HEADING"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[n] = U.batchPreProcess(CTX, x_batches_takeoff[n])

            
        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)

        x_inputs = []
        for i in range(NB_BATCHES):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            x_inputs.append(x_input)

        return x_inputs, y_batches

    def genEval(self, path):
        """
        Load a flight for the evaluation process.
        CSV are managed there one by one for memory issue.
        As we need for each ads-b message a sliding window of
        ~ 128 timesteps it can generate large arrays
        Do the Preprocess in the same way as training flights

        Called automatically by the trainer after the training phase.


        Parameters:
        ----------

        path : str
            Path to the csv

        Returns:
        -------
        x : np.array[flight_lenght, history, features]
            Inputs data for the model

        y : np.array
            True labels associated with x batches
        """

        CTX = self.CTX
        LON_I = self.CTX["FEATURE_MAP"]["longitude"]
        LAT_I = self.CTX["FEATURE_MAP"]["latitude"]
        ALT_I = self.CTX["FEATURE_MAP"]["altitude"]
        GEO_I = self.CTX["FEATURE_MAP"]["geoaltitude"]


        df = pd.read_csv(path, sep=",",dtype={"callsign":str, "icao24":str})
        icao = df["icao24"].iloc[0]
        callsign = df["callsign"].iloc[0]

        # preprocess the trajectory
        array = U.dfToFeatures(df, CTX)
        label = U.getLabel(CTX, icao, callsign)
        if (label == 0):
            return [], []
        y = self.yScaler.transform([label])[0]

        # allocate the required memory
        x_batches = np.zeros((len(array), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        y_batches = np.full((len(array), len(self.yScaler.classes_)), y)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = np.zeros((len(array), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = np.zeros((len(array), self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"],3), dtype=np.float32)

        # generate the sub windows
        for t in range(0, len(array)):
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]

            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

            x_batch = array[start+shift:end:CTX["DILATION_RATE"]]

            x_batches[t, :pad_lenght] = self.FEATURES_MIN_VALUES
            x_batches[t, pad_lenght:] = x_batch
            
            
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if (array[0,ALT_I] > 1000 or array[0,GEO_I] > 1000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_MIN_VALUES)
                else:
                    takeoff = array[start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[t, :pad_lenght] = self.FEATURES_MIN_VALUES
                x_batches_takeoff[t, pad_lenght:] = takeoff
                
            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = x_batches[t, -1, LAT_I], x_batches[t, -1, LON_I]
                x_batches_map[t] = U.genMap(lat, lon, self.CTX["IMG_SIZE"]) / 255.0


        # preprocess the batches
        for n in range(len(x_batches)):
            x_batches[n] = U.batchPreProcess(CTX, x_batches[n], CTX["RELATIVE_POSITION"], CTX["RELATIVE_HEADING"], CTX["RANDOM_HEADING"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[n] = U.batchPreProcess(CTX, x_batches_takeoff[n])



        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)

        x_inputs = []
        for i in range(len(x_batches)):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            x_inputs.append(x_input)

        return x_inputs, y_batches

