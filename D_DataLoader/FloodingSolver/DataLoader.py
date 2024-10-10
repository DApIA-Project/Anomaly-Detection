
import pandas as pd

import D_DataLoader.Utils as U
import D_DataLoader.FloodingSolver.Utils as SU
from   D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader

from  _Utils.FeatureGetter import FG_flooding as FG
import _Utils.Color as C
from   _Utils.Color import prntC
import _Utils.Limits as Limits
from   _Utils.Scaler3D import  StandardScaler3D, SigmoidScaler2D, fill_nan_3d, fill_nan_2d
from   _Utils.ProgressBar import ProgressBar
from   _Utils.plotADSB import PLT
from numpy_typing import np, ax
from   _Utils.ADSB_Streamer import streamer, CacheArray2D




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
    yScaler:SigmoidScaler2D

    x_train:"list[np.float64_2d[ax.time, ax.feature]]"
    x_test :"list[np.float64_2d[ax.time, ax.feature]]"

    win_cache:CacheArray2D
    loss_cache:CacheArray2D

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.PAD = None

        self.xScaler = StandardScaler3D()
        self.yScaler = SigmoidScaler2D()


        training = (CTX["EPOCHS"] and path != "")
        if (training):
            x = self.__get_dataset__(path)
            self.x_train,self.x_test = self.__split__(x)


        # for streaming
        self.win_cache = CacheArray2D()
        self.loss_cache = CacheArray2D()
        self.win_cache.set_feature_size(self.CTX["FEATURES_IN"])
        self.loss_cache.set_feature_size(1)


    def __load_dataset__(self, CTX:dict, path:str) -> "list[np.float64_2d[ax.time, ax.feature]]":

        filenames = U.list_flights(path, limit=Limits.INT_MAX)
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

    def __scalers_transform__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
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

    def __reshape__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                          y_batch:np.float64_2d[ax.sample, ax.feature],
                          nb_batches:int, batch_size:int) -> """tuple[
            np.float64_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float64_3d[ax.batch, ax.sample, ax.feature]]""":

        x_batches = x_batch.reshape(nb_batches, batch_size, self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        y_batches = y_batch.reshape(nb_batches, batch_size, self.CTX["FEATURES_OUT"])

        return x_batches, y_batches



# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================

    def get_train(self) -> """tuple[
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

        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, CTX["NB_BATCH"], CTX["BATCH_SIZE"])

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
        PLT.plot   (NAME, lon, lat, color="tab:blue", linestyle="--")
        PLT.scatter(NAME, lon, lat, color="tab:blue", marker="x")
        # PLT.scatter(NAME, y_lon, y_lat, color="tab:green", marker="+")

        PLT.attach_data(NAME+"Origin", (o_lat, o_lon, o_track))


        # plot
        track = np.zeros((len(lat),))
        lat_n1, lon_n1, _ = U.normalize_trajectory(self.CTX, lat, lon, track, o_lat, o_lon, 0, True, True, False)
        lat_n2, lon_n2, _ = U.normalize_trajectory(self.CTX, lat, lon, track, o_lat, o_lon, o_track, True, True, False)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(lat, lon, color="tab:blue")
        ax[0].scatter(lat, lon, color="tab:blue", marker="x")
        ax[0].scatter([lat[-1]], [lon[-1]], color="tab:red", marker="o")
        ax[0].title.set_text("Trajectory")
        ax[0].axis('square')

        ax[1].plot(lat_n1, lon_n1, color="tab:blue")
        ax[1].scatter(lat_n1, lon_n1, color="tab:blue", marker="x")
        ax[1].scatter([lat_n1[-1]], [lon_n1[-1]], color="tab:red", marker="o")
        ax[1].title.set_text("Normalized lat lon")
        ax[1].axis('square')

        ax[2].plot(lat_n2, lon_n2, color="tab:blue")
        ax[2].scatter(lat_n2, lon_n2, color="tab:blue", marker="x")
        ax[2].scatter([lat_n2[-1]], [lon_n2[-1]], color="tab:red", marker="o")
        ax[2].title.set_text("Normalized lat lon with track")
        ax[2].axis('square')

        fig.tight_layout()
        plt.savefig("./_Artifacts/FloodingSolver/normalisation_example.png")



# |====================================================================================================================
# |     GENERATE A TEST SET
# |====================================================================================================================

    def get_test(self) -> """tuple[
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

        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, nb_batches, batch_size)
        return x_batches, y_batches

    def process_stream_of(self, message:"dict[str, object]") -> """tuple[
            np.float64_3d[ax.sample, ax.time, ax.feature],
            np.float64_2d[ax.sample, ax.feature],
            bool,
            tuple[float, float, float]]""":
        icao24 = message["icao24"]
        tag = message.get("tag", "0")

        df = streamer.get(icao24, tag)
        if (df is None):
            prntC(C.ERROR, "Cannot get stream of unknown trajectory")

        df = df.until(message["timestamp"] + 1)
        if (len(df) <= 1):
            new_msg = U.df_to_feature_array(self.CTX, df.copy(), check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)
        else:
            new_msg = U.df_to_feature_array(self.CTX, df[-2:], check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)[1:]

        win = self.win_cache.extend(icao24, tag, new_msg, [len(df)] * len(new_msg))
        
        x_batch, y_batch = SU.alloc_batch(self.CTX, 1)
        x_batch[0], y, valid, origin = SU.gen_sample(
            self.CTX, [win], self.PAD, 0, len(win)-1-self.CTX["HORIZON"], training=False)
        y_batch[0] = FG.lat_lon(y)


        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, 1, 1)
        return x_batches[0], y_batches[0], valid, origin
