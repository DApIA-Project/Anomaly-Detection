 

from _Utils.Scaler3D import MinMaxScaler3D, StandardScaler3D, MinMaxScaler2D, StandardScaler2D, SigmoidScaler2D, fillNaN3D, sigmoid_inverse
import _Utils.Color as C
from _Utils.Color import prntC

from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader
import D_DataLoader.Utils as U, D_DataLoader.FloodingSolver.Utils as SU

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

        print("Loading dataset :")

        # Read each file
        for f in range(len(data_files)):
            file = data_files[f]
            # set time as index
            df = pd.read_csv(os.path.join(path, file), sep=",",dtype={"callsign":str, "icao24":str})
            df.drop(["icao24", "callsign"], axis=1, inplace=True)
            if ("y_" in df.columns):
                df.drop(["y_"], axis=1, inplace=True)

            df = DataFrame(df)
            array = U.dfToFeatures(df, CTX)
            
            # Add the flight to the dataset
            x.append(array)

            if (f % 20 == (len(data_files)-1) % 20):
                done_20 = int(((f+1)/len(data_files)*20))
                print("\r|"+done_20*"="+(20-done_20)*" "+f"| {(f+1)}/{len(data_files)}", end=" "*20)
        print("\n", flush=True)


        return x
    


    def __init__(self, CTX, path="") -> None:    
        self.CTX = CTX

        if (CTX["EPOCHS"] and path != ""):
            self.x = self.__get_dataset__(path)
        else:
            self.x = []


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

            else:
                self.FEATURES_PAD_VALUES[f] = 0

        self.x = fillNaN3D(self.x, self.FEATURES_PAD_VALUES)

        # Create the scalers
        self.xScaler = StandardScaler3D()
        self.yScaler = SigmoidScaler2D()
       
        
        # Split the dataset into train and test according to the ratio in context
        ratio = self.CTX["TEST_RATIO"]
        split_index = int(len(self.x) * (1 - ratio))
        self.x_train = self.x[:split_index]
        self.x_test = self.x[split_index:]


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

        # Allocate memory for the batches
        x_batches = np.zeros((nb_batches * batch_size, CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        y_batches = np.zeros((nb_batches * batch_size, CTX["FEATURES_OUT"]))

        FEATURES_OUT = CTX["PRED_FEATURES"]
        FEATURES_OUT = [CTX["FEATURE_MAP"][f] for f in FEATURES_OUT]

        for n in range(len(x_batches)):
            flight_i, t = SU.pick_an_interesting_aircraft(CTX, self.x_train) 
   
            
            # compute the bounds of the fragment
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
                
            # shift to always have the last timestep as part of the fragment
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
                
            batch = np.concatenate([
                self.x_train[flight_i][start+shift:end:CTX["DILATION_RATE"]],
                self.x_train[flight_i][[end+CTX["HORIZON"]-1]]
            ], axis=0).copy()

            # print(batch[:, :])

            if (n == 0):
                lats, lons = batch[:, CTX["FEATURE_MAP"]["latitude"]], batch[:, CTX["FEATURE_MAP"]["longitude"]]
                plt.plot(lats, lons)
                plt.scatter(lats[-1], lons[-1], marker="x", color="red")
                plt.scatter(lats[:-1], lons[:-1], marker="x", color="green")
                plt.axis('square')
                plt.savefig("./_Artefact/test.png")
                plt.clf()


            batch = SU.batchPreProcess(CTX, batch, CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])

            if (n == 0):
                lats, lons = batch[:, CTX["FEATURE_MAP"]["latitude"]], batch[:, CTX["FEATURE_MAP"]["longitude"]]
                plt.plot(lats, lons)
                plt.scatter(lats[-1], lons[-1], marker="x", color="red")
                plt.scatter(lats[:-1], lons[:-1], marker="x", color="green")
                plt.axis('square')
                plt.savefig("./_Artefact/test2.png")
                plt.clf()

                plt.plot(lats[-10:], lons[-10:])
                plt.scatter(lats[-1], lons[-1], marker="x", color="red")
                plt.scatter(lats[-10:-1], lons[-10:-1], marker="x", color="green")
                plt.axis('square')
                plt.savefig("./_Artefact/test3.png")
                plt.clf()

            
            x_batches[n, :pad_lenght] = self.FEATURES_PAD_VALUES
            x_batches[n, pad_lenght:] = batch[:-1]

            y_batches[n] = batch[-1, FEATURES_OUT]

        min_lat = np.min(x_batches[:, :, CTX["FEATURE_MAP"]["latitude"]])
        max_lat = np.max(x_batches[:, :, CTX["FEATURE_MAP"]["latitude"]])
        min_lon = np.min(x_batches[:, :, CTX["FEATURE_MAP"]["longitude"]])
        max_lon = np.max(x_batches[:, :, CTX["FEATURE_MAP"]["longitude"]])
        mean_lat = np.mean(x_batches[:, :, CTX["FEATURE_MAP"]["latitude"]])
        mean_lon = np.mean(x_batches[:, :, CTX["FEATURE_MAP"]["longitude"]])
        y_min_lat = np.min(y_batches[:, CTX["PRED_FEATURE_MAP"]["latitude"]])
        y_max_lat = np.max(y_batches[:, CTX["PRED_FEATURE_MAP"]["latitude"]])
        y_min_lon = np.min(y_batches[:, CTX["PRED_FEATURE_MAP"]["longitude"]])
        y_max_lon = np.max(y_batches[:, CTX["PRED_FEATURE_MAP"]["longitude"]])



        # fit the scaler on the first epoch
        if not(self.xScaler.isFitted()):
            self.xScaler.fit(x_batches)
            self.yScaler.fit(y_batches)

            # # rescale scaler to have the same range for lat and lon
            # range = self.yScaler.maxs - self.yScaler.mins
            # range_lat = range[CTX["PRED_FEATURE_MAP"]["latitude"]]
            # center_lon = (max_lon + min_lon) / 2
            # min_lon = center_lon - range_lat / 2
            # max_lon = center_lon + range_lat / 2
            # self.yScaler.mins[CTX["PRED_FEATURE_MAP"]["longitude"]] = min_lon
            # self.yScaler.maxs[CTX["PRED_FEATURE_MAP"]["longitude"]] = max_lon

            # if (CTX["RELATIVE_TRACK"]):
            #     self.yScaler.stds[CTX["PRED_FEATURE_MAP"]["longitude"]] = self.yScaler.stds[CTX["PRED_FEATURE_MAP"]["latitude"]]

            print("DEBUG SCALLERS : ")
            prntC("feature:","|".join(self.CTX["USED_FEATURES"]), start=C.BRIGHT_BLUE)
            print("mean   :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.means)]))
            print("std dev:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.stds)]))
            # print("mean TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.means)]))
            # print("std  TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.stds)]))
            print("nan pad:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_PAD_VALUES)]))
            print("min    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_MIN_VALUES)]))
            print("max    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.FEATURES_MAX_VALUES)]))
        
        # print(np.min(y_batches, axis=0))
        # print(np.max(y_batches, axis=0))

        # print(y_batches)

        x_batches = self.xScaler.transform(x_batches)
        y_batches = self.yScaler.transform(y_batches)


        # Reshape the data into [nb_batches, batch_size, timestep, features]
        x_batches = x_batches.reshape(nb_batches, batch_size, self.CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        y_batches = y_batches.reshape(nb_batches, batch_size, self.CTX["FEATURES_OUT"])

        # y_batches = np.zeros_like(y_batches)

        return x_batches, y_batches


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


        # Allocate memory for the batches
        x_batches = np.zeros((NB_BATCHES, CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        y_batches = np.zeros((NB_BATCHES, CTX["FEATURES_OUT"]))

        FEATURES_OUT = CTX["PRED_FEATURES"]
        FEATURES_OUT = [CTX["FEATURE_MAP"][f] for f in FEATURES_OUT]

        for n in range(len(x_batches)):
            flight_i, t = SU.pick_an_interesting_aircraft(CTX, self.x_test) 
   
            
            # compute the bounds of the fragment
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
                
            # shift to always have the last timestep as part of the fragment !!
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
            # build the batch
                
            batch = np.concatenate([
                self.x_test[flight_i][start+shift:end:CTX["DILATION_RATE"]],
                self.x_test[flight_i][[end+CTX["HORIZON"]-1]]
            ], axis=0)

            batch = SU.batchPreProcess(CTX, batch, CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])

            x_batches[n, :pad_lenght] = self.FEATURES_PAD_VALUES
            x_batches[n, pad_lenght:] = batch[:-1]

            y_batches[n] = batch[-1, FEATURES_OUT]

        x_batches = self.xScaler.transform(x_batches)
        y_batches = self.yScaler.transform(y_batches)
        return x_batches, y_batches


        

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

        df = pd.read_csv(path, sep=",",dtype={"callsign":str, "icao24":str})
        df.drop(["icao24", "callsign"], axis=1, inplace=True)
        if ("y_" in df.columns):
            df.drop(["y_"], axis=1, inplace=True)

        
        df = DataFrame(df)
        array = U.dfToFeatures(df, CTX, __EVAL__=True)
        
        array = fillNaN3D([array], self.FEATURES_PAD_VALUES)[0]

        x_batches = np.zeros((len(array)-CTX["HORIZON"], CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        y_batches = np.zeros((len(array)-CTX["HORIZON"], CTX["FEATURES_OUT"]))

        FEATURES_OUT = CTX["PRED_FEATURES"]
        FEATURES_OUT = [CTX["FEATURE_MAP"][f] for f in FEATURES_OUT]

        i = 0
        ts = []
        for n in range(len(x_batches)):
            
            if not(SU.checkTrajectory(CTX, [array], 0, n)):
                continue

            # compute the bounds of the fragment
            start = max(0, n+1-CTX["HISTORY"])
            end = n+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
                
            # shift to always have the last timestep as part of the fragment !!
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])
            # build the batch
                
            batch = np.concatenate([
                array[start+shift:end:CTX["DILATION_RATE"]],
                array[[end+CTX["HORIZON"]-1]]
            ], axis=0)

            batch = SU.batchPreProcess(CTX, batch, CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], False)
            

            x_batches[i, :pad_lenght] = self.FEATURES_PAD_VALUES
            x_batches[i, pad_lenght:] = batch[:-1]
            ts.append(array[end+CTX["HORIZON"]-1, CTX["FEATURE_MAP"]["timestamp"]])
    
            y_batches[i] = batch[-1, FEATURES_OUT]
            i += 1

        x_batches = x_batches[:i]
        y_batches = y_batches[:i]

        x_batches = self.xScaler.transform(x_batches)
        y_batches = self.yScaler.transform(y_batches)


        return x_batches, y_batches, ts
