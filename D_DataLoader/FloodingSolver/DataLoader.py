
import pandas as pd

import D_DataLoader.Utils as U
import D_DataLoader.FloodingSolver.Utils as SU
from   D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader

import _Utils.FeatureGetter as FG
import _Utils.Color as C
from   _Utils.Color import prntC
from _Utils import Limits
from   _Utils.Scaler3D import  StandardScaler3D, SigmoidScaler2D, fill_nan_3d, fill_nan_2d
from   _Utils.ProgressBar import ProgressBar
from _Utils.plotADSB import PLT
from   _Utils.ADSB_Streamer import Streamer
from _Utils.numpy import np, ax



# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================

BAR = ProgressBar()
STREAMER = Streamer()



# |====================================================================================================================
# | DATA LOADER
# |====================================================================================================================

# managing the data preprocessing
class DataLoader(AbstractDataLoader):

    x_train:"list[np.float64_2d[ax.time, ax.feature]]"
    x_test :"list[np.float64_2d[ax.time, ax.feature]]"

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.PAD = None

        self.streamer:StreamerInterface = StreamerInterface(self)

        self.xScaler = StandardScaler3D()
        self.yScaler = SigmoidScaler2D()

        training = (CTX["EPOCHS"] and path != "")
        if (training):
            x = self.__get_dataset__(path)
            self.x_train,self.x_test = self.__split__(x)
        else:
            prntC(C.INFO, "Training, deactivated, only evaluation will be launched.")
            prntC(C.WARNING, "Make sure everything is loaded from the disk, especially the PAD values.")



    def __load_dataset__(self, CTX:dict, path:str) -> "list[np.float64_2d[ax.time, ax.feature]]":

        filenames = U.list_flights(path, limit=Limits.INT_MAX) #Limit.INT_MAX
        BAR.reset(max=len(filenames))

        x = []
        for f in range(len(filenames)):
            df = U.read_trajectory(filenames[f])
            array = U.df_to_feature_array(CTX, df)
            x.append(array)
            BAR.update(f+1)

        if (self.PAD is None): self.PAD = U.genPadValues(CTX, x)
        x = fill_nan_3d(x, self.PAD)

        return x



# |====================================================================================================================
# |    SCALERS
# |====================================================================================================================

    def __scalers_transform__(self, CTX:dict,
                                    x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                                    y_batch:np.float64_2d[ax.sample, ax.feature]=None) \
            -> """tuple[np.float64_3d[ax.sample, ax.time, ax.feature], np.float64_2d[ax.sample, ax.feature]]
                | np.float64_3d[ax.sample, ax.time, ax.feature]""":

        if (not(self.xScaler.is_fitted())):
            self.xScaler.fit(x_batch)
        x_batch = self.xScaler.transform(x_batch)

        if (y_batch is not None):
            if (not(self.yScaler.is_fitted())):
                self.yScaler.fit(y_batch)

            y_batch = self.yScaler.transform(y_batch)
            return x_batch, y_batch
        return x_batch



# |====================================================================================================================
# |     UTILS
# |====================================================================================================================

    def __reshape__(self, CTX:dict,
                          x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                          y_batch:np.float64_2d[ax.sample, ax.feature],
                          nb_batches:int, batch_size:int) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float64_3d[ax.batch, ax.sample, ax.feature]]""":

        x_batches = x_batch.reshape(nb_batches, batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"])
        y_batches = y_batch.reshape(nb_batches, batch_size, CTX["FEATURES_OUT"])

        return x_batches, y_batches



# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================

    def genEpochTrain(self) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float64_3d[ax.batch, ax.sample, ax.feature]]""":

        CTX = self.CTX

        # Allocate memory for the batches
        x_batch, y_batch = SU.alloc_batch(CTX, CTX["NB_BATCH"] * CTX["BATCH_SIZE"])

        for n in range(len(x_batch)):
            x_sample, y_sample, origin = SU.gen_random_sample(CTX, self.x_train, self.PAD)

            x_batch[n] = x_sample
            y_batch[n] = y_sample

        self.__plot_flight__(x_sample, y_sample, origin)

        x_batch, y_batch = self.__scalers_transform__(CTX, x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(CTX, x_batch, y_batch, CTX["NB_BATCH"], CTX["BATCH_SIZE"])

        return x_batches, y_batches



    def __plot_flight__(self,
                        x:np.float64_2d[ax.time, ax.feature],
                        y:np.float64_1d[ax.feature],
                        origin:"tuple[float, float, float]") -> None:

        NAME = "train_example"
        lat = FG.lat(x)
        lon = FG.lon(x)
        o_lat, o_lon, o_track = origin

        lat, lon     = U.denormalize_trajectory(self.CTX, lat, lon,
                                                o_lat, o_lon, o_track)
        y_lat, y_lon = U.denormalize_trajectory(self.CTX, [y[0]], [y[1]],
                                                o_lat, o_lon, o_track)

        box = [U.mini(lat, y_lat), U.mini(lon, y_lon), U.maxi(lat, y_lat), U.maxi(lon, y_lon)]
        # add some margin
        size = max(box[2]-box[0], box[3]-box[1])
        box[0] -= size * 0.1
        box[1] -= size * 0.1
        box[2] += size * 0.1
        box[3] += size * 0.1


        PLT.figure (NAME, box[0], box[1], box[2], box[3])
        PLT.title  (NAME, "Flooding Solver - Prediction on a training sample")
        PLT.plot   (NAME, lon, lat, color="tab:blue", linestyle="--")
        PLT.scatter(NAME, lon, lat, color="tab:blue", marker="x")
        PLT.scatter(NAME, y_lon, y_lat, color="tab:green", marker="+")

        PLT.attach_data(NAME+"Origin", (o_lat, o_lon, o_track))



# |====================================================================================================================
# |     GENERATE A TEST SET
# |====================================================================================================================

    def genEpochTest(self) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float64_3d[ax.batch, ax.sample, ax.feature]]""":

        CTX = self.CTX
        SIZE =  int(CTX["NB_BATCH"] * CTX["BATCH_SIZE"] * CTX["TEST_RATIO"])

        x_batch, y_batch = SU.alloc_batch(CTX, SIZE)

        for n in range(SIZE):
            x_sample, y_sample, _ = SU.gen_random_sample(CTX, self.x_test, self.PAD)
            x_batch[n] = x_sample
            y_batch[n] = y_sample

        batch_size = min(CTX["MAX_BATCH_SIZE"], len(x_batch))
        nb_batches = len(x_batch) // batch_size

        x_batch, y_batch = self.__scalers_transform__(CTX, x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(CTX, x_batch, y_batch, nb_batches, batch_size)
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
            np.float64_2d[ax.sample, ax.feature],
            bool,tuple[float, float, float]]""":

        MAX_LENGTH_NEEDED = self.CTX["HISTORY"] + self.CTX["HORIZON"]
        MIN_LENGTH_NEEDED = self.CTX["DILATION_RATE"] + 1 + self.CTX["HORIZON"]

        tag = x.get("tag", x["icao24"])
        raw_df = STREAMER.add(x, tag=tag)
        cache = STREAMER.cache("FloodingSolver", tag)

        array = U.df_to_feature_array(self.CTX, raw_df[-2:], check_length=False)
        array = fill_nan_2d(array, self.dl.PAD)

        if (cache is not None):
            cache = np.concatenate([cache, array[1:]], axis=0)
            cache = cache[-MAX_LENGTH_NEEDED:]
        else:
            cache = array
        STREAMER.cache("FloodingSolver", tag, cache)

        # |--------------------------
        # | Generate a sample
        x_batch, y_batch = SU.alloc_batch(self.CTX, 1)

        # set valid to None, mean that we don't know yet
        valid = None
        if (len(cache) < MIN_LENGTH_NEEDED): valid = False
        x_batch[0], y, valid, origin = SU.gen_sample(
            self.CTX, [cache], self.dl.PAD, 0, len(cache)-1-self.CTX["HORIZON"], valid, training=False)
        y_batch[0] = FG.lat_lon(y)


        x_batch, y_batch = self.dl.__scalers_transform__(self.CTX, x_batch, y_batch)
        x_batches, y_batches = self.dl.__reshape__(self.CTX, x_batch, y_batch, 1, 1)
        return x_batches[0], y_batches[0], valid, origin


    def clear(self)-> None:
        STREAMER.clear()


