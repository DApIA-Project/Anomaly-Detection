 

from _Utils.Scaler3D import MinMaxScaler3D, StandardScaler3D, fillNaN3D
from _Utils.SparceLabelBinarizer import SparceLabelBinarizer
from _Utils.Metrics import computeTimeserieVarienceRate
import _Utils.Color as C
from _Utils.Color import prntC

from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader
import D_DataLoader.AircraftClassification.Utils as U

import os
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import _Utils.mlviz as MLviz

from _Utils.DataFrame import DataFrame


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
        data_files.sort()

        data_files = data_files[:]

        x = []
        zoi = []
        y = []

        print("Loading dataset :")

        # Read each file
        for f in range(len(data_files)):
            file = data_files[f]
            # set time as index
            df = pd.read_csv(os.path.join(path, file), sep=",",dtype={"callsign":str, "icao24":str})
            
            # Get the aircraft right label for his imatriculation
            icao24 = df["icao24"].iloc[0]
            callsign = df["callsign"].iloc[0]
            df.drop(["icao24", "callsign"], axis=1, inplace=True)

            if ("prediction" in df.columns):
                zoi.append(df["prediction"].values)
                df.drop(["prediction"], axis=1, inplace=True)
                if ("y_" in df.columns):
                    df.drop(["y_"], axis=1, inplace=True)
            else:
                zoi.append(np.full((len(df),), True))
            

            label = U.getLabel(CTX, icao24, callsign)
            if (label == 0):
                continue
            
            print(df)
            df = DataFrame(df)
            # print(df)
            array = U.dfToFeatures(df, label, CTX)
            
            
            # Add the flight to the dataset
            x.append(array)
            y.append(label)

            if (f % 20 == (len(data_files)-1) % 20):
                done_20 = int(((f+1)/len(data_files)*20))
                print("\r|"+done_20*"="+(20-done_20)*" "+f"| {(f+1)}/{len(data_files)}", end=" "*20)
        print("\n", flush=True)


        x_log = x[0]
        x_lat = x_log[:, CTX["FEATURE_MAP"]["latitude"]]
        x_lon = x_log[:, CTX["FEATURE_MAP"]["longitude"]]
        MLviz.log("trajectory", {"lat":x_lat, "lon":x_lon})
        # reshape (timesteps, features) to (features, timesteps) to 
        x_log = x_log.transpose(1,0)
        MLviz.log("data", x_log)

        return x, zoi, y
    


    def __init__(self, CTX, path) -> None:    
        self.CTX = CTX

        U.resetICAOdb()

        
        if (CTX["EPOCHS"]):
            if self.CTX["CHANGED"]:
                self.uncacheDataset()
            self.x, self.y = self.__get_dataset__(path)
        else:
            self.x, self.y = [], []


        self.FEATURES_MIN_VALUES = np.full((CTX["FEATURES_IN"],), np.nan)
        self.FEATURES_MAX_VALUES = np.full((CTX["FEATURES_IN"],), np.nan)
        for i in range(len(self.x)):
            self.FEATURES_MIN_VALUES = np.nanmin([self.FEATURES_MIN_VALUES, np.nanmin(self.x[i], axis=0)], axis=0)
            self.FEATURES_MAX_VALUES = np.nanmax([self.FEATURES_MAX_VALUES, np.nanmax(self.x[i], axis=0)], axis=0)


        
        # fit the scalers and define the min values
        self.FEATURES_PAD_VALUES = self.FEATURES_MIN_VALUES.copy()
        for f in range(len(CTX["USED_FEATURES"])):
            feature = CTX["USED_FEATURES"][f]

            if (feature == "latitude"):
                self.FEATURES_PAD_VALUES[f] = 0
            elif (feature == "longitude"):
                self.FEATURES_PAD_VALUES[f] = 0

            elif (feature == "altitude" 
                  or feature == "geoaltitude" 
                  or feature == "vertical_rate" 
                  or feature == "groundspeed" 
                  or feature == "track" 
                  or feature == "relative_track" 
                  or feature == "timestamp"):
                
                self.FEATURES_PAD_VALUES[f] = 0




        self.x = fillNaN3D(self.x, self.FEATURES_PAD_VALUES)

        # Create the scalers
        self.xScaler = StandardScaler3D()
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler = StandardScaler3D()
        self.yScaler = SparceLabelBinarizer()
        self.yScaler.setVariables(self.CTX["USED_LABELS"])
        print(self.yScaler.classes_)


        # Fit the y scaler
        # x scaler will be fitted later after batch preprocessing
        if (CTX["EPOCHS"]):
            self.y = self.yScaler.transform(self.y)
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

        prntC("Train dataset size :", C.BLUE, len(self.x_train))
        prntC("Test dataset size :", C.BLUE, len(self.x_test))


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

        NB=self.CTX["NB_TRAIN_SAMPLES"]

        log_n = -1

        for n in range(0, len(x_batches), NB):

            # Pick a random label
            label_i = np.random.randint(0, self.yScaler.classes_.shape[0])
            nb = min(NB, len(x_batches) - n)
            flight_i, ts = U.pick_an_interesting_aircraft(CTX, self.x_train, self.y_train, label_i, n=nb)


            for i in range(len(ts)):
                t = ts[i]       
            
                # compute the bounds of the fragment
                start = max(0, t+1-CTX["HISTORY"])
                end = t+1
                length = end - start
                pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
                
                # shift to always have the last timestep as part of the fragment !!
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
                # build the batch

                
                x_batch = self.x_train[flight_i][start+shift:end:CTX["DILATION_RATE"]]
                x_batches[n+i, :pad_lenght] = self.FEATURES_PAD_VALUES
                x_batches[n+i, pad_lenght:] = x_batch


                if CTX["ADD_TAKE_OFF_CONTEXT"]:
                    # compute the bounds of the fragment
                    start = 0
                    end = length
                    shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                    # build the batch
                    if(self.x_train[flight_i][0,ALT_I] > 2000 or self.x_train[flight_i][0,GEO_I] > 2000):
                        takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_PAD_VALUES)
                    else:
                        takeoff = self.x_train[flight_i][start+shift:end:CTX["DILATION_RATE"]]
                        if (log_n == -1): log_n = n+i
                    
                    # add padding and add to the batch
                    x_batches_takeoff[n+i, :pad_lenght] = self.FEATURES_PAD_VALUES
                    x_batches_takeoff[n+i, pad_lenght:] = takeoff


                y_batches[n+i] = self.y_train[flight_i]

            


                if CTX["ADD_MAP_CONTEXT"]:
                    lat, lon = U.getAircraftPosition(CTX, x_batches[n+i])
                    x_batches_map[n+i] = U.genMap(lat, lon, self.CTX["IMG_SIZE"])

                # if (n == 0 and i == 0):
                    # for feature in CTX["USED_FEATURES"]:
                    #     MLviz.log(feature, x_batches[n+i, :, CTX["FEATURE_MAP"][feature]])
                if (n+i == log_n):
                    lat = x_batches[n+i, :, LAT_I]
                    lon = x_batches[n+i, :, LON_I]
                    MLviz.log("0-x_btch_trajectory", {"lat":lat, "lon":lon})
                    MLviz.log("1-x_btch", x_batches[n+i].transpose(1,0))
                x_batches[n+i, pad_lenght:] = U.batchPreProcess(CTX, x_batches[n+i, pad_lenght:], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
                lat = x_batches[n+i, :, LAT_I]
                lon = x_batches[n+i, :, LON_I]
                if (n+i == log_n):
                    MLviz.log("2-x_btch_trajectory_preprocessed", {"lat":lat, "lon":lon})
                if CTX["ADD_TAKE_OFF_CONTEXT"]:
                    if (n+i == log_n):
                        MLviz.log("3-x_btch_takeoff", x_batches_takeoff[n+i].transpose(1,0))
                    x_batches_takeoff[n+i, pad_lenght:] = U.batchPreProcess(CTX, x_batches_takeoff[n+i, pad_lenght:])
                # get label

        

        # reorder the batches TODO check insterest of that
        # combinations = np.arange(len(x_batches))
        # np.random.shuffle(combinations)
        # x_batches = x_batches[combinations]
        # y_batches = y_batches[combinations]
        # if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = x_batches_takeoff[combinations]
        # if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = x_batches_map[combinations]


        # fit the scaler on the first epoch
        if not(self.xScaler.isFitted()):
            self.xScaler.fit(x_batches)

            if (CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler.fit(x_batches_takeoff)
            print("DEBUG SCALLERS : ")
            prntC("feature:","|".join(self.CTX["USED_FEATURES"]), start=C.BRIGHT_BLUE)
            print("mean   :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.means)]))
            print("std dev:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.stds)]))
            # print("mean TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.means)]))
            # print("std  TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.stds)]))
            print("nan pad:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_PAD_VALUES)]))
            print("min    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_MIN_VALUES)]))
            print("max    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_MAX_VALUES)]))
        

        all_lat = x_batches[:, :, LAT_I]
        all_lon = x_batches[:, :, LON_I]

        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)

        MLviz.log("4-x_btch_scaled", x_batches[log_n].transpose(1,0))
        # MLviz.log("5-x_btch_takeoff_scaled", x_batches_takeoff[log_n].transpose(1,0))


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
            t = t[0]
                    
            # compute the bounds of the fragment
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
            
            # shift to always have the last timestep as part of the fragment !!
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
            # build the batch

            x_batch = self.x_test[flight_i][start+shift:end:CTX["DILATION_RATE"]]
            x_batches[n, :pad_lenght] = self.FEATURES_PAD_VALUES
            x_batches[n, pad_lenght:] = x_batch


            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if(self.x_test[flight_i][0,ALT_I] > 2000 or self.x_test[flight_i][0,GEO_I] > 2000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_PAD_VALUES)
                else:
                    takeoff = self.x_test[flight_i][start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[n, :pad_lenght] = self.FEATURES_PAD_VALUES
                x_batches_takeoff[n, pad_lenght:] = takeoff
                

            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = U.getAircraftPosition(CTX, x_batches[n])
                x_batches_map[n] = U.genMap(lat, lon, self.CTX["IMG_SIZE"])

            x_batches[n, pad_lenght:] = U.batchPreProcess(CTX, x_batches[n, pad_lenght:], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[n, pad_lenght:] = U.batchPreProcess(CTX, x_batches_takeoff[n, pad_lenght:])

            # get label
            y_batches[n] = self.y_test[flight_i]
        


            
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

        label = U.getLabel(CTX, icao, callsign)
        if (label == 0): # no label -> skip
            return [], []
        array = U.dfToFeatures(df, None, CTX)
        
        array = fillNaN3D([array], self.FEATURES_PAD_VALUES)[0]
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

            x_batches[t, :pad_lenght] = self.FEATURES_PAD_VALUES
            x_batches[t, pad_lenght:] = x_batch
            
            
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if(array[0,ALT_I] > 2000 or array[0,GEO_I] > 2000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), self.FEATURES_PAD_VALUES)
                else:
                    takeoff = array[start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[t, :pad_lenght] = self.FEATURES_PAD_VALUES
                x_batches_takeoff[t, pad_lenght:] = takeoff
                
            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = U.getAircraftPosition(CTX, x_batches[t])
                x_batches_map[t] = U.genMap(lat, lon, self.CTX["IMG_SIZE"])
            
            x_batches[t, pad_lenght:] = U.batchPreProcess(CTX, x_batches[t, pad_lenght:], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[t, pad_lenght:] = U.batchPreProcess(CTX, x_batches_takeoff[t, pad_lenght:])



        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)

        x_inputs = []
        for i in range(len(x_batches)):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            x_inputs.append(x_input)

        

        return x_inputs, y_batches

