
import numpy as np
import os

import _Utils.FeatureGetter as FG
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.Scaler3D import StandardScaler3D, MinMaxScaler2D, fillNaN3D, fillNaN2D
from _Utils.SparceLabelBinarizer import SparceLabelBinarizer
from _Utils.ProgressBar import ProgressBar
import _Utils.Limits as Limits
import _Utils.plotADSB as PLT
from _Utils.ADSB_Streamer import Streamer

import D_DataLoader.Utils as U
import D_DataLoader.AircraftClassification.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================

BAR = ProgressBar()
STREAMER = Streamer()


# |====================================================================================================================
# | DATA LOADER
# |====================================================================================================================

class DataLoader(AbstractDataLoader):

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.PAD = None

        self.streamer:StreamerInterface = StreamerInterface(self)

        self.xScaler = StandardScaler3D()
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler = StandardScaler3D()
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): self.xAirportScaler = MinMaxScaler2D()
        self.yScaler = SparceLabelBinarizer(self.CTX["USED_LABELS"])

        training = (CTX["EPOCHS"] and path != "")
        if (training):

            x, y, self.filenames = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, y)
        else:
            prntC(C.INFO, "Training, deactivated, only evaluation will be launched.")
            prntC(C.WARNING, "Make sure everything is loaded from the disk, especially the PAD values.")



    def __load_dataset__(self, CTX:dict, path:str) -> "tuple[np.ndarray, np.ndarray, list[str]]":
        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.listFlight(path, limit=Limits.INT_MAX)
            prntC(C.INFO, "Dataset loading")
        else:
            # path is a file
            path = path.split("/")
            filenames = [path[-1]]
            path = "/".join(path[:-1])

        SU.resetICAOdb()
        BAR.reset(max=len(filenames))

        x, y = [], []
        for f in range(len(filenames)):
            df = U.read_trajectory(path, filenames[f])

            label = SU.getLabel(CTX, df["icao24", 0])
            if (label == 0):
                continue

            array = U.dfToFeatures(df, CTX)

            x.append(array)
            y.append(label)

            if (is_folder): BAR.update(f+1)

        if (self.PAD is None): self.PAD = U.genPadValues(CTX, x)
        x = fillNaN3D(x, self.PAD)
        y = self.yScaler.transform(y)

        return x, y, filenames


    def __split__(self, x, y):
        split = U.splitDataset([x, y], self.CTX["TEST_RATIO"])
        return split[0][0], split[0][1], split[1][0], split[1][1]


# |====================================================================================================================
# |    SCALERS
# |====================================================================================================================

    def __scalers_transform__(self, CTX, x_batches, x_batches_takeoff, x_batches_airport):
        # fit the scaler on the first epoch
        if not(self.xScaler.isFitted()):
            self.xScaler.fit(x_batches)
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler.fit(x_batches_takeoff)
            if (CTX["ADD_AIRPORT_CONTEXT"]): self.xAirportScaler.fit(x_batches_airport)

        x_batches = self.xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = self.xTakeOffScaler.transform(x_batches_takeoff)
        if (CTX["ADD_AIRPORT_CONTEXT"]): x_batches_airport = self.xAirportScaler.transform(x_batches_airport)
        return x_batches, x_batches_takeoff, x_batches_airport


# |====================================================================================================================
# |     UTILS
# |====================================================================================================================


# |--------------------------------------------------------------------------------------------------------------------
# |     SPLIT DATASET IN BATCHES
# |--------------------------------------------------------------------------------------------------------------------

    def __reshape__(self, CTX:dict,
                    x_batches:np.ndarray,
                    x_batches_takeoff:np.ndarray, x_batches_map:np.ndarray, x_batches_airport:np.ndarray,
                    y_batches:np.ndarray,
                    nb_batches:np.ndarray, batch_size:np.ndarray)\
            ->"tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":

        # Reshape the data into [nb_batches, batch_size, timestep, features]
        x_batches = x_batches.reshape(nb_batches, batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        if CTX["ADD_TAKE_OFF_CONTEXT"]:
            x_batches_takeoff = x_batches_takeoff.reshape(nb_batches, batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        if CTX["ADD_MAP_CONTEXT"]:
            x_batches_map = x_batches_map.reshape(nb_batches, batch_size, CTX["IMG_SIZE"], CTX["IMG_SIZE"], 3)
        if CTX["ADD_AIRPORT_CONTEXT"]:
            x_batches_airport = x_batches_airport.reshape(nb_batches, batch_size, CTX["AIRPORT_CONTEXT_IN"])
        y_batches = y_batches.reshape(nb_batches, batch_size, CTX["FEATURES_OUT"])
        return x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches


# |--------------------------------------------------------------------------------------------------------------------
# |     REORDERING AXIS TO MATCH THE MODELS INPUTS
# |--------------------------------------------------------------------------------------------------------------------

    def __format_return__(self, CTX:dict,
                          x_batches:np.ndarray,
                          x_batches_takeoff:np.ndarray, x_batches_map:np.ndarray, x_batches_airport:np.ndarray,
                          y_batches:np.ndarray)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        x_inputs = []
        for i in range(len(x_batches)):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            if CTX["ADD_AIRPORT_CONTEXT"]: x_input.append(x_batches_airport[i])
            x_inputs.append(x_input)

        return x_inputs, y_batches


# |--------------------------------------------------------------------------------------------------------------------
# |     EXECUTE THE WHOLE BATCH PROCESSING AFTER DATA LOADING
# |--------------------------------------------------------------------------------------------------------------------

    def __post_process_batches__(self, CTX:dict,
                                 x_batches:np.ndarray,
                                 x_batches_takeoff:np.ndarray,
                                 x_batches_map:np.ndarray,
                                 x_batches_airport:np.ndarray,
                                 y_batches:np.ndarray,
                                 nb_batches:np.ndarray, batch_size:np.ndarray)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        x_batches, x_batches_takeoff, x_batches_airport =\
            self.__scalers_transform__(CTX, x_batches, x_batches_takeoff, x_batches_airport)
        x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches =\
            self.__reshape__(CTX,
                             x_batches, x_batches_takeoff, x_batches_map, x_batches_airport,
                             y_batches,
                             nb_batches, batch_size)
        return self.__format_return__(CTX, x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches)


# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================

    def genEpochTrain(self)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        CTX = self.CTX
        NB=self.CTX["NB_TRAIN_SAMPLES"]
        x_batches, y_batches, x_batches_takeoff, x_batches_map, x_batches_airport =\
            SU.allocBatch(self.CTX, CTX["NB_BATCH"] * CTX["BATCH_SIZE"])
        filenames = []

        for nth in range(0, len(x_batches), NB):
            nb = min(NB, len(x_batches) - nth)
            s = slice(nth, nth+nb)

            x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport, filename =\
                SU.genRandomBatch(CTX, self.x_train, self.y_train, self.PAD, nb, filenames=self.filenames)
            x_batches[s] = x_batch
            y_batches[s] = y_batch
            filenames += filename
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff[s] = x_batch_takeoff
            if (CTX["ADD_MAP_CONTEXT"]): x_batches_map[s] = x_batch_map
            if (CTX["ADD_AIRPORT_CONTEXT"]): x_batches_airport[s] = x_batch_airport

        self.__plot_flight__(x_batches[0], y_batches[0], filenames[0])

        return self.__post_process_batches__(CTX,
                    x_batches, x_batches_takeoff, x_batches_map, x_batches_airport,
                    y_batches,
                    CTX["NB_BATCH"], CTX["BATCH_SIZE"])

    def __plot_flight__(self, x, y, filename):
        lat = FG.lat(x)
        lon = FG.lon(x)

        COLORS = ["orange", ["yellow"], "green"]
        box = [min(lat), min(lon), max(lat), max(lon)]
        NAME = "train_example"
        PLT.figure(NAME, box[0], box[1], box[2], box[3])
        PLT.plot(NAME, lon, lat, color="black", linestyle="--")
        PLT.scatter(NAME, lon[:-len(COLORS)], lat[:-len(COLORS)], color="black", marker="x")
        for i in range(len(COLORS)):
            PLT.scatter(NAME, lon[-len(COLORS)+i], lat[-len(COLORS)+i], color=COLORS[i], marker="x")

        PLT.attach_data(NAME, {"filename":filename})


# |====================================================================================================================
# |     GENERATE A TEST SET
# |====================================================================================================================

    def genEpochTest(self)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        CTX = self.CTX
        SIZE =  int(CTX["NB_BATCH"] * CTX["BATCH_SIZE"] * CTX["TEST_RATIO"])

        x_batches, y_batches, x_batches_takeoff, x_batches_map, x_batches_airport =\
            SU.allocBatch(self.CTX, SIZE)

        for nth in range(0, len(x_batches)):
            x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport, filenames =\
                SU.genRandomBatch(CTX, self.x_test, self.y_test, self.PAD, size=1, filenames=self.filenames)
            x_batches[nth] = x_batch
            y_batches[nth] = y_batch
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff[nth] = x_batch_takeoff
            if (CTX["ADD_MAP_CONTEXT"]): x_batches_map[nth] = x_batch_map
            if (CTX["ADD_AIRPORT_CONTEXT"]): x_batches_airport[nth] = x_batch_airport

        batch_size = min(CTX["MAX_BATCH_SIZE"], len(x_batches))
        nb_batches = len(x_batches) // batch_size

        return self.__post_process_batches__(CTX,
                    x_batches, x_batches_takeoff, x_batches_map, x_batches_airport,
                    y_batches,
                    nb_batches, batch_size)


# |====================================================================================================================
# | STREAMING ADS-B MESSAGE FOR EVAL UNDER REAL CONDITIONS
# |====================================================================================================================

class StreamerInterface:
    def __init__(self, dl:DataLoader) -> None:
        self.dl = dl
        self.CTX = dl.CTX

    def stream(self, x:"dict[str, object]"):
        tag = x.get("tag", x['icao24'])

        raw_df = STREAMER.add(x, tag=tag)
        cache = STREAMER.cache("AircraftClassification", tag)

        array = U.dfToFeatures(raw_df[-2:], self.CTX, check_length=False)
        array = fillNaN2D(array, self.dl.PAD)

        # concatenate the new array to the cache
        # remove the first element of the array
        # as it is the last element of the previous array but unformatted
        if (cache is not None):
            cache = np.concatenate([cache, array[1:]], axis=0)
        else:
            cache = array
        STREAMER.cache("AircraftClassification", tag, cache)

        # batch assembly
        x_batch, _, x_batch_takeoff, x_batch_map, x_batch_airport =\
            SU.allocBatch(self.CTX, 1)
        x_batch[0], x_batch_takeoff[0], x_batch_map[0], x_batch_airport[0], valid =\
            SU.genSample(self.CTX, [cache], self.dl.PAD, 0, len(cache)-1)
        x_batches, _ =  self.dl.__post_process_batches__(self.CTX,
                    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport,
                    np.zeros((1, self.CTX["FEATURES_OUT"])),
                    1, 1)
        return x_batches[0], valid

    def predicted(self, x:"dict[str, object]", y_:np.ndarray) -> np.ndarray:
        tag = x.get("tag", x['icao24'])
        preds = STREAMER.cache("AircraftClassification_Pred", tag)

        if (preds is None):
            STREAMER.cache("AircraftClassification_Pred", tag, [y_])
            return np.array([y_])

        preds.append(y_)
        return np.array(preds)



