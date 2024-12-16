

from numpy_typing import np, ax
import matplotlib.pyplot as plt
from _Utils.os_wrapper import os
import math

from _Utils.FeatureGetter import FG_replay as FG
import _Utils.Color as C
from   _Utils.Color import prntC
import _Utils.Limits as Limits
from   _Utils.Scaler3D import fill_nan_3d, fill_nan_2d
from   _Utils.ProgressBar import ProgressBar
from   _Utils.ADSB_Streamer import streamer, CacheArray2D, CacheList

import D_DataLoader.Utils as U
import D_DataLoader.ReplaySolver.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================


BAR = ProgressBar()
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

    __actual_flight__:int

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX

        training = (CTX["EPOCHS"] and path != "")

        if (training):
            x, files = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, files, size = TEST_SIZE)
            self.__actual_flight__ = 0

        # for streaming
        dtype = np.float64
        if (CTX["FEATURES_IN"] == 1 and CTX["USED_FEATURES"][0] == "fingerprint"):
            dtype = np.int8
        self.win_cache = CacheArray2D(dtype)
        self.win_cache.set_feature_size(self.CTX["FEATURES_IN"])
        self.pred_cache = CacheList()


    def __load_dataset__(self, CTX:dict, path:str) -> "tuple[list[np.float64_2d[ax.time, ax.feature]], list[str]]":
        # path can be a folder or a file
        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.list_flights(path, limit=Limits.INT_MAX)
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
        first_flight = self.__actual_flight__
        if (first_flight >= len(self.x_train)):
            self.__actual_flight__ = 0
            return 0, 0, 0

        # pick the following flights until the batch is full
        size = 0
        limit = self.CTX["MAX_BATCH_SIZE"] * self.CTX["MAX_NB_BATCH"]
        while (self.__actual_flight__ < len(self.x_train)):
            samples = len(self.x_train[self.__actual_flight__]) - self.CTX["HISTORY"] + 1
            if (size + samples < limit):
                size += samples
                self.__actual_flight__ += 1
            else:
                break

        last_flight = self.__actual_flight__

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
            n =0
            for t in range(CTX["HISTORY"] - 1, len(self.x_train[i])):

                sample, valid = SU.gen_sample(CTX, self.x_train, i, t)

                if not(valid):
                    continue

                x_batches[sample_i] = sample
                y_batches[sample_i] = self.y_train[i]
                sample_i += 1
                n += 1


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
            if (y is not None):y_sample = y[ith]
            else: y_sample = "unknown"

            x_batches[i], y_batches[i] = x_sample, y_sample

        x_batches, y_batches = self.__reshape__(x_batches, y_batches, 1, TEST_SIZE)
        return x_batches, y_batches

    def process_stream_of(self, message:"dict[str, object]") -> """tuple[
            np.float64_3d[ax.sample, ax.time, ax.feature],
            bool]""":

        icao24 = message["icao24"]
        tag = message.get("tag", "0")

        traj = streamer.get(icao24, tag)
        if (traj is None):
            prntC(C.ERROR, "Cannot get stream of unknown trajectory")

        df = traj.data.until(message["timestamp"])

        new_msg = U.df_to_feature_array(self.CTX, df[-3:], check_length=False)
        if (len(df) >= 3):
            win = self.win_cache.extend(icao24, tag, new_msg[1:2], [len(df)-1])
        else:
            win = self.win_cache.extend(icao24, tag, new_msg[len(df)-1:len(df)], [len(df)-1])

        x_batch, y_batch = SU.alloc_batch(self.CTX, 1)
        x_batch, valid = SU.gen_sample(self.CTX, [win], 0, len(win) - 1)

        if (not valid):
            return np.zeros((0, 0, 0)), False

        x_batches, _ = self.__reshape__(x_batch, y_batch, 1, 1)
        return x_batches[0], valid