

from _Utils.numpy import np, ax
import matplotlib.pyplot as plt
from _Utils.os_wrapper import os
import math

import _Utils.FeatureGetter as FG
import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils import Limits
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

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX

        training = (CTX["EPOCHS"] and path != "")

        if (training):
            x, files = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, files, size = TEST_SIZE)
            prntC(C.INFO, "Dataset loaded train :", C.BLUE, len(self.x_train), C.RESET,
                                          "test :", C.BLUE, len(self.x_test))
        else:
            prntC(C.INFO, "Training, deactivated, only evaluation will be launched.")
            prntC(C.WARNING, "Make sure everything is loaded from the disk, especially the PAD values.")
            prntC(C.WARNING, "i'm not sure everything will work as expected : TODO CHECK !")



    def __load_dataset__(self, CTX:dict, path:str) -> "tuple[list[np.float64_2d[ax.time, ax.feature]], list[str]]":
        # path can be a folder or a file
        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.list_flights(path, limit=300)
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

    def get_train(self) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.str_2d[ax.batch, ax.sample]]""":

        CTX = self.CTX

        size = 0
        for i in range(len(self.x_train)):
            size += len(self.x_train[i]) - CTX["HISTORY"] + 1

        x_batches, y_batches = SU.alloc_batch(CTX, size)

        sample_i = 0
        for i in range(len(self.x_train)):
            for t in range(CTX["HISTORY"] - 1, len(self.x_train[i])):

                sample, valid = SU.gen_sample(CTX, self.x_train, i, t)
                if not(valid):
                    continue

                y = self.y_train[i]

                x_batches[sample_i] = sample[0]
                y_batches[sample_i] = y
                sample_i += 1


        print("sample_i", sample_i, len(x_batches))


        batch_size = min(CTX["MAX_BATCH_SIZE"], sample_i)
        nb_batches = math.ceil(sample_i / batch_size)

        size = nb_batches * batch_size

        x_batches = x_batches[:size]
        y_batches = y_batches[:size]

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
            else: x = self.x_test

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
            np.float64_2d[ax.time, ax.feature],
            bool]""":

        MAX_LENGTH_NEEDED = self.CTX["HISTORY"]

        tag = x.get("tag", x["icao24"])
        raw_df = STREAMER.add(x, tag=tag)
        cache = STREAMER.getCache("ReplaySolver", tag)

        array = U.df_to_feature_array(self.CTX, raw_df[-2:], check_length=False)
        array = fill_nan_2d(array, self.dl.PAD)

        if (cache is not None):
            cache = np.concatenate([cache, array[1:]], axis=0)
            cache = cache[-MAX_LENGTH_NEEDED:]
        else:
            cache = array
        STREAMER.cache("FloodingSolver", tag, cache)

        return cache, SU.check_sample(self.CTX, [cache], 0, len(cache) - 1)


    def clear(self)-> None:
        STREAMER.clear()
