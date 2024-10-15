
from _Utils.os_wrapper import os

from _Utils.FeatureGetter import FG_spoofing as FG
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.Scaler3D import StandardScaler3D, MinMaxScaler3D, MinMaxScaler2D, fill_nan_3d, fill_nan_2d
from   _Utils.SparceLabelBinarizer import SparceLabelBinarizer
from   _Utils.ProgressBar import ProgressBar
import _Utils.Limits as Limits
from   _Utils.plotADSB import PLT
from   numpy_typing import np, ax, ax
from   _Utils.ADSB_Streamer import streamer, CacheArray2D

import D_DataLoader.Utils as U
import D_DataLoader.AircraftClassification.Utils as SU
from   D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================

BAR = ProgressBar()


# |====================================================================================================================
# | DATA LOADER
# |====================================================================================================================

class DataLoader(AbstractDataLoader):

    CTX:dict
    PAD:np.float64_1d[ax.feature]

    xScaler:StandardScaler3D
    xTakeOffScaler:StandardScaler3D
    xAirportScaler:MinMaxScaler2D
    yScaler:SparceLabelBinarizer

    x_train:np.float64_3d[ax.sample, ax.time, ax.feature]
    y_train:np.float64_2d[ax.sample, ax.label]
    x_test:np.float64_3d[ax.sample, ax.time, ax.feature]
    y_test:np.float64_2d[ax.sample, ax.label]

    win_cache:CacheArray2D
    prediction_cache:CacheArray2D

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.PAD = None

        Scaler = self.getScalerClass(self.CTX["SCALER"])

        self.xScaler:StandardScaler3D = Scaler()
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler:StandardScaler3D = Scaler()
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): self.xAirportScaler = MinMaxScaler2D()
        self.yScaler = SparceLabelBinarizer(self.CTX["USED_LABELS"])

        training = (CTX["EPOCHS"] and path != "")
        if (training):
            x, y, self.filenames = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, y)

        self.win_cache = CacheArray2D()
        self.win_cache.set_feature_size(CTX["FEATURES_IN"])
        self.prediction_cache = CacheArray2D()
        self.prediction_cache.set_feature_size(CTX["LABELS_OUT"])



    def __load_dataset__(self, CTX:dict, path:str) -> """tuple[
            list[np.float64_2d[ax.time, ax.feature]],
            np.float64_2d[ax.sample, ax.label],
            list[str]]""":

        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.list_flights(path, limit=Limits.INT_MAX)
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
            df = U.read_trajectory(filenames[f])
            label = SU.getLabel(CTX, df["icao24"].iloc[0])
            if (label == 0):
                continue

            array = U.df_to_feature_array(CTX, df)

            x.append(array)
            y.append(label)

            if (is_folder): BAR.update(f+1)

        if (self.PAD is None): self.PAD = U.genPadValues(CTX, x)
        x = fill_nan_3d(x, self.PAD)
        y = self.yScaler.transform(y)
        return x, y, filenames




# |====================================================================================================================
# |    SCALERS
# |====================================================================================================================

    # def __scalers_transform__(self, x_batch, x_batch_takeoff, x_batch_airport):
    def __scalers_transform__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                             x_batch_takeoff:np.float64_3d[ax.sample, ax.time, ax.feature],
                             x_batch_airport:np.float64_2d[ax.sample, ax.feature])->"""tuple[
            np.float64_3d[ax.sample, ax.time, ax.feature],
            np.float64_3d[ax.sample, ax.time, ax.feature],
            np.float64_2d[ax.sample, ax.feature]]""":

        # fit the scaler on the first epoch
        if not(self.xScaler.is_fitted()):
            self.xScaler.fit(x_batch)
            if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): self.xTakeOffScaler.fit(x_batch_takeoff)
            if (self.CTX["ADD_AIRPORT_CONTEXT"]): self.xAirportScaler.fit(x_batch_airport)

        x_batch = self.xScaler.transform(x_batch)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff = self.xTakeOffScaler.transform(x_batch_takeoff)
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport = self.xAirportScaler.transform(x_batch_airport)
        return x_batch, x_batch_takeoff, x_batch_airport

    def getScalerClass(self, name:str) -> "type":
        if (name == "standard") : return StandardScaler3D
        if (name == "minmax") : return MinMaxScaler3D
        return None

# |====================================================================================================================
# |     UTILS
# |====================================================================================================================

# |--------------------------------------------------------------------------------------------------------------------
# |     SPLIT IN BATCHES
# |--------------------------------------------------------------------------------------------------------------------

    def __reshape__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                    x_batch_takeoff:np.float64_3d[ax.sample, ax.time, ax.feature],
                    x_batch_map:np.float64_4d[ax.sample, ax.x, ax.y, ax.feature],
                    x_batch_airport:np.float64_2d[ax.sample, ax.feature],
                    y_batch:np.float64_2d[ax.sample, ax.label],
                    nb_batch:int, batch_size:int)->"""tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.ndarray,
            np.float64_3d[ax.batch, ax.sample, ax.feature],
            np.float64_3d[ax.batch, ax.sample, ax.label]]""":


        # Reshape the data into [nb_batch, batch_size, timestep, features]
        CTX = self.CTX
        x_batches = x_batch.reshape(nb_batch, batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        x_batches_takeoff, x_batches_map, x_batches_airport = None, None, None
        if CTX["ADD_TAKE_OFF_CONTEXT"]:
            x_batches_takeoff = x_batch_takeoff.reshape(nb_batch, batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        if CTX["ADD_MAP_CONTEXT"]:
            x_batches_map = x_batch_map.reshape(nb_batch, batch_size, CTX["IMG_SIZE"], CTX["IMG_SIZE"], 3)
        if CTX["ADD_AIRPORT_CONTEXT"]:
            x_batches_airport = x_batch_airport.reshape(nb_batch, batch_size, CTX["AIRPORT_CONTEXT_IN"])
        y_batches = y_batch.reshape(nb_batch, batch_size, CTX["LABELS_OUT"])
        return x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches


# |--------------------------------------------------------------------------------------------------------------------
# |     REORDERING AXIS TO MATCH THE MODELS INPUTS
# |--------------------------------------------------------------------------------------------------------------------

    def __format_return__(self, CTX:dict,
                          x_batches:np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
                          x_batches_takeoff:np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
                          x_batches_map:np.ndarray,
                          x_batches_airport:np.float64_3d[ax.batch, ax.sample, ax.feature],
                          y_batches:np.float64_3d[ax.batch, ax.sample, ax.label])->"""tuple[
                list[
                    tuple[
                        np.float64_3d[ax.sample, ax.time, ax.feature],
                        np.float64_3d[ax.sample, ax.time, ax.feature],
                        np.float64_4d[ax.sample, ax.x, ax.y, ax.feature],
                        np.float64_3d[ax.sample, ax.feature]]],
                np.float64_3d[ax.batch, ax.sample, ax.label]]""":

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

    def __post_process_batch__(self, CTX:dict,
                                 x_batch:np.ndarray,
                                 x_batch_takeoff:np.ndarray,
                                 x_batch_map:np.ndarray,
                                 x_batch_airport:np.ndarray,
                                 y_batches:np.ndarray,
                                 nb_batches:np.ndarray, batch_size:np.ndarray) ->"""tuple[
                list[
                    tuple[
                        np.float64_3d[ax.sample, ax.time, ax.feature],
                        np.float64_3d[ax.sample, ax.time, ax.feature],
                        np.float64_4d[ax.sample, ax.x, ax.y, ax.feature],
                        np.float64_3d[ax.sample, ax.feature]]],
                np.float64_3d[ax.batch, ax.sample, ax.label]]""":

        x_batch, x_batch_takeoff, x_batch_airport =\
            self.__scalers_transform__(x_batch, x_batch_takeoff, x_batch_airport)
        x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches =\
            self.__reshape__(x_batch, x_batch_takeoff, x_batch_map, x_batch_airport,
                             y_batches,
                             nb_batches, batch_size)
        return self.__format_return__(CTX, x_batches, x_batches_takeoff, x_batches_map, x_batches_airport, y_batches)


# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================

    def get_train(self)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        CTX = self.CTX
        x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport =\
            SU.alloc_batch(self.CTX, CTX["NB_BATCH"] * CTX["BATCH_SIZE"])
        filenames = []

        for s in range(0, len(x_batch)):

            x_sample, y_sample, x_sample_takeoff, x_sample_map, x_sample_airport, filename =\
                SU.gen_random_sample(CTX, self.x_train, self.y_train, self.PAD, filenames=self.filenames)
            x_batch[s] = x_sample
            y_batch[s] = y_sample
            filenames += filename
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff[s] = x_sample_takeoff
            if (CTX["ADD_MAP_CONTEXT"]): x_batch_map[s] = x_sample_map
            if (CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport[s] = x_sample_airport

        self.__plot_flight__(x_batch[0], y_batch[0], filenames[0])

        return self.__post_process_batch__(CTX,
                    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport,
                    y_batch,
                    CTX["NB_BATCH"], CTX["BATCH_SIZE"])


    def __plot_flight__(self, x:np.float64_2d[ax.time, ax.feature], y:np.float64_1d[ax.label], filename:str) -> None:
        NAME = "train_example"
        COLORS = ["orange", "yellow", "green"]
        lat = FG.lat(x)
        lon = FG.lon(x)

        box = [min(lat), min(lon), max(lat), max(lon)]
        PLT.figure(NAME, box[0], box[1], box[2], box[3])
        PLT.plot(NAME, lon, lat, color="black", linestyle="--")
        PLT.scatter(NAME, lon[:-len(COLORS)], lat[:-len(COLORS)], color="black", marker="x")
        for i in range(len(COLORS)):
            PLT.scatter(NAME, lon[-len(COLORS)+i], lat[-len(COLORS)+i], color=COLORS[i], marker="x")
        PLT.attach_data(NAME, {"filename":filename})


# |====================================================================================================================
# |     GENERATE A TEST SET
# |====================================================================================================================

    def get_test(self)\
            ->"tuple[list[np.ndarray], np.ndarray]":

        CTX = self.CTX
        SIZE =  int(CTX["NB_BATCH"] * CTX["BATCH_SIZE"] * CTX["TEST_RATIO"])

        x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport =\
            SU.alloc_batch(self.CTX, SIZE)

        for nth in range(0, len(x_batch)):
            x_sample, y_sample, x_sample_takeoff, x_sample_map, x_sample_airport, filenames =\
                SU.gen_random_sample(CTX, self.x_test, self.y_test, self.PAD, filenames=self.filenames)
            x_batch[nth] = x_sample
            y_batch[nth] = y_sample
            if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff[nth] = x_sample_takeoff
            if (CTX["ADD_MAP_CONTEXT"]): x_batch_map[nth] = x_sample_map
            if (CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport[nth] = x_sample_airport

        batch_size = min(CTX["MAX_BATCH_SIZE"], len(x_batch))
        nb_batches = len(x_batch) // batch_size

        return self.__post_process_batch__(CTX,
                    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport,
                    y_batch,
                    nb_batches, batch_size)

    def process_stream_of(self, message:"dict[str, object]") -> """tuple[
             tuple[
                np.float64_3d[ax.sample, ax.time, ax.feature],
                np.float64_3d[ax.sample, ax.time, ax.feature],
                np.float64_4d[ax.sample, ax.x, ax.y, ax.rgb],
                np.float64_3d[ax.sample, ax.feature],
                np.float64_3d[ax.sample, ax.label]
            ],
            list[bool]]""" :

        icao24 = message["icao24"]
        tag = message.get("tag", "0")

        df = streamer.get(icao24, tag)
        if (df is None):
            prntC(C.ERROR, "Cannot get stream of unknown trajectory")

        df = df.until(message["timestamp"])

        if (len(df) <= 1):
            new_msg = U.df_to_feature_array(self.CTX, df.copy(), check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)
        else:
            new_msg = U.df_to_feature_array(self.CTX, df[-2:], check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)[1:]

        win = self.win_cache.extend(icao24, tag, new_msg, [len(df)] * len(new_msg))

        # batch assembly
        x_batch, _, x_batch_takeoff, x_batch_map, x_batch_airport =\
            SU.alloc_batch(self.CTX, 1)
        x_sample, x_sample_takeoff, x_sample_map, x_sample_airport, valid =\
            SU.gen_sample(self.CTX, [win], self.PAD, 0, len(win)-1)

        x_batch[0] = x_sample
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff[0] = x_sample_takeoff
        if (self.CTX["ADD_MAP_CONTEXT"]): x_batch_map[0] = x_sample_map
        if (self.CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport[0] = x_sample_airport

        x_batches, _ =  self.__post_process_batch__(self.CTX,
                    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport,
                    np.zeros((1, self.CTX["LABELS_OUT"])),
                    1, 1)
        return x_batches[0], valid


