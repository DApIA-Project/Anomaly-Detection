

from _Utils.numpy import np, ax
import matplotlib.pyplot as plt
from _Utils.os_wrapper import os
import math

import _Utils.FeatureGetter as FG
import _Utils.Color as C
from   _Utils.Color import prntC
import _Utils.Limits as Limits
from   _Utils.Scaler3D import fill_nan_3d, fill_nan_2d
from   _Utils.ProgressBar import ProgressBar
from   _Utils.ADSB_Streamer import Streamer

import D_DataLoader.Utils as U
import D_DataLoader.ReplaySolver.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================


BAR = ProgressBar()
STREAMER = Streamer()


TEST_SIZE = 60


# |====================================================================================================================
# | DATA LOADER
# |====================================================================================================================

class DataLoader(AbstractDataLoader):

    CTX:dict

    streamer:"StreamerInterface"

    x_train:"list[np.float64_2d[ax.time, ax.feature]]"
    x_test :"list[np.float64_2d[ax.time, ax.feature]]"
    y_train:"list[str]"
    y_test :"list[str]"

    __last_training_flight__:int

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.streamer = StreamerInterface(self)

        training = (CTX["EPOCHS"] and path != "")

        if (training):
            x, files = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, files, size = TEST_SIZE)
            prntC(C.INFO, "Dataset loaded train :", C.BLUE, len(self.x_train), C.RESET,
                                          "test :", C.BLUE, len(self.x_test))

            self.__last_training_flight__ = 0
        else:
            prntC(C.INFO, "Training, deactivated, only evaluation will be launched.")
            prntC(C.WARNING, "Make sure everything is loaded from the disk, especially the PAD values.")
            prntC(C.WARNING, "i'm not sure everything will work as expected : TODO CHECK !")




    def __load_dataset__(self, CTX:dict, path:str) -> "tuple[list[np.float64_2d[ax.time, ax.feature]], list[str]]":
        # path can be a folder or a file
        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.list_flights(path, limit=1000)#Limits.INT_MAX)
            prntC(C.INFO, "Dataset loading")
        else:
            path = path.split("/")
            filenames = [path[-1]]
            path = "/".join(path[:-1])

        BAR.reset(max=len(filenames))

        x = []
        for f in range(len(filenames)):
            df = U.read_trajectory(filenames[f])
            array = U.df_to_feature_array(CTX, df)
            x.append(array)
            filenames[f] = filenames[f].split("/")[-1]
            if (is_folder): BAR.update()


        return x, filenames

# |====================================================================================================================
# |     UTILS
# |====================================================================================================================

    def __reshape__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                          y_batch:np.str_1d[ax.sample],
                          nb_batches:int, batch_size:int) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.str_2d[ax.batch, ax.sample]]""":

        x_batches = x_batch.reshape(nb_batches, batch_size, self.CTX["INPUT_LEN"],self.CTX["FEATURES_IN"])
        y_batches = y_batch.reshape(nb_batches, batch_size)

        return x_batches, y_batches

# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================


    def __next__(self) -> "tuple[int, int, int]":
        # If we have already used all the dataset -> reset !
        first_flight = self.__last_training_flight__
        if (first_flight >= len(self.x_train)):
            self.__last_training_flight__ = 0
            return 0, 0, 0


        # pick the following flights until the batch is full
        size = 0
        limit = self.CTX["MAX_BATCH_SIZE"] * self.CTX["MAX_NB_BATCH"]
        for i in range(first_flight, len(self.x_train)):
            samples = len(self.x_train[i]) - self.CTX["HISTORY"] + 1
            if (size + samples < limit):
                size += len(self.x_train[i]) - self.CTX["HISTORY"] + 1
                self.__last_training_flight__ = i
            else:
                break
        self.__last_training_flight__ += 1
        last_flight = self.__last_training_flight__

        return first_flight, last_flight, size


    def get_train(self) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.str_2d[ax.batch, ax.sample]]""":

        CTX = self.CTX

        first_flight, last_flight, size = self.__next__()
        if (size == 0):
            return np.zeros((0, 0, 0, 0)), np.zeros((0, 0))

        # Allocate the batch and fill it
        x_batches, y_batches = SU.alloc_batch(CTX, size)

        sample_i = 0
        for i in range(first_flight, last_flight):
            for t in range(CTX["HISTORY"] - 1, len(self.x_train[i])):

                sample, valid = SU.gen_sample(CTX, self.x_train, i, t)

                if not(valid):
                    continue

                x_batches[sample_i] = sample
                y_batches[sample_i] = self.y_train[i]
                sample_i += 1

        # compute final shape & clean unused samples (invalid samples)
        batch_size = min(CTX["MAX_BATCH_SIZE"], sample_i)
        nb_batches = math.ceil(sample_i / batch_size)

        size = nb_batches * batch_size

        x_batches, y_batches = x_batches[:size],  y_batches[:size]

        x_batches, y_batches = self.__reshape__(x_batches, y_batches, nb_batches, batch_size)
        return x_batches, y_batches


# |====================================================================================================================
# |     GENERATE A TEST SET
# |====================================================================================================================


    def get_test(self) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.str_2d[ax.batch, ax.sample]]""":

        CTX = self.CTX
        x_batches, y_batches = SU.alloc_batch(CTX, TEST_SIZE)

        for i in range(TEST_SIZE):
            if (i < TEST_SIZE // 2): x, y = self.x_train, self.y_train
            else: x, y = self.x_test, None

            x_sample, (ith, _) = SU.gen_random_sample(CTX, x)
            if (y is not None): y_sample = y[ith]
            else: y_sample = "unknown"

            x_batches[i], y_batches[i] = x_sample, y_sample

        x_batches, y_batches = self.__reshape__(x_batches, y_batches, 1, TEST_SIZE)
        return x_batches, y_batches


# |====================================================================================================================
# | STREAMING ADS-B MESSAGE TO EVALUATE THE MODEL UNDER REAL CONDITIONS
# |====================================================================================================================


class StreamerInterface:

    def __init__(self, dl:DataLoader) -> None:
        self.dl = dl
        self.CTX = dl.CTX

    def stream(self, x:"dict[str, object]") -> """tuple[
            np.float64_3d[ax.sample, ax.time, ax.feature],
            bool]""":

        MAX_LENGTH_NEEDED = self.CTX["HISTORY"]

        tag = x.get("tag", x["icao24"])
        raw_df = STREAMER.add(x, tag=tag)
        cache = STREAMER.cache("ReplaySolver", tag)

        array = U.df_to_feature_array(self.CTX, raw_df[-3:], check_length=False)

        if (cache is not None):
            cache = np.concatenate([cache, array[1:2]], axis=0)
            cache = cache[-MAX_LENGTH_NEEDED:]
        else:
            cache = array
        STREAMER.cache("ReplaySolver", tag, cache)

        x_batch, y_batch = SU.alloc_batch(self.CTX, 1)

        x_batch, valid = SU.gen_sample(self.CTX, [cache], 0, len(cache) - 1)
        if (not valid):
            return np.zeros((0, 0, 0)), False

        x_batches, _ = self.dl.__reshape__(x_batch, y_batch, 1, 1)
        return x_batches[0], valid

    def cache(self, tag:str, cache:object) -> None:
        STREAMER.cache("FloodingSolver", tag, cache)
    def get_cache(self, tag:str) -> object:
        return STREAMER.cache("FloodingSolver", tag)

    def clear(self)-> None:
        STREAMER.clear()
