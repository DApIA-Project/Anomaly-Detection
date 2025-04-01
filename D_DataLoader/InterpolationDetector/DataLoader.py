
from numpy_typing import np, ax


import D_DataLoader.Utils as U
import D_DataLoader.InterpolationDetector.Utils as SU
from   D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader

from   _Utils.FeatureGetter import FG_interp as FG
import _Utils.Color as C
from   _Utils.Color import prntC
import _Utils.Limits as Limits
from   _Utils.Scaler3D import  StandardScaler3D, StandardScaler2D, fill_nan_3d, fill_nan_2d
from   _Utils.ProgressBar import ProgressBar
# from   _Utils.plotADSB import PLT
import matplotlib.pyplot as plt
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

    x_train:"list[np.float64_2d[ax.time, ax.feature]]"
    y_train:"list[np.float64_1d[ax.feature]]"
    x_test :"list[np.float64_2d[ax.time, ax.feature]]"
    y_test :"list[np.float64_1d[ax.feature]]"

    win_cache:CacheArray2D
    preds_cache:CacheArray2D

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX:dict, path:str="") -> None:
        self.CTX = CTX
        self.PAD = None

        self.xScaler = StandardScaler3D()

        training = (CTX["EPOCHS"] and path != "")
        if (training):
            x, y = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(x, y)


        # for streaming
        self.win_cache = CacheArray2D()
        self.preds_cache = CacheArray2D()
        self.win_cache.set_feature_size(self.CTX["FEATURES_IN"])
        self.preds_cache.set_feature_size(1)


    def __load_dataset__(self, CTX:dict, path:str) -> "list[np.float64_2d[ax.time, ax.feature]]":

        filenames = []
        filenames += U.list_flights(path+"base/", limit=1000)#Limits.INT_MAX)
        filenames += U.list_flights(path+"interp_0-0.00015/", limit=1000)#Limits.INT_MAX)
        # filenames += U.list_flights(path+"interp_0.0001/", limit=1000)#Limits.INT_MAX)
        # filenames += U.list_flights(path+"interp_3e-05/", limit=1000)#Limits.INT_MAX)
        BAR.reset(max=len(filenames))
        
        np.random.shuffle(filenames)

        x = []
        y = []
        for f in range(len(filenames)):
            df = U.read_trajectory(filenames[f])
            array = U.df_to_feature_array(CTX, df)
            x.append(array)
            y.append("interp" in filenames[f])
            BAR.update(f+1)

        if (self.PAD is None): self.PAD = U.genPadValues(CTX, x)
        x = fill_nan_3d(x, self.PAD)

        return x, y



# |====================================================================================================================
# |    SCALERS
# |====================================================================================================================

    def __scalers_transform__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature]) \
            -> """np.float64_3d[ax.sample, ax.time, ax.feature]""":

        if (not(self.xScaler.is_fitted())):
            self.xScaler.fit(x_batch)
            
        x_batch = self.xScaler.transform(x_batch)
        
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
        y_batches = y_batch.reshape(nb_batches, batch_size, 1)

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
            x_sample, y_sample = SU.gen_random_sample(CTX, self.x_train, self.y_train, self.PAD)

            x_batch[n] = x_sample
            y_batch[n] = y_sample


        x_batch = self.__scalers_transform__(x_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, CTX["NB_BATCH"], CTX["BATCH_SIZE"])
        
        self.__plot_flight__(x_batches[0, 0:4], y_batches[0, 0:4])


        return x_batches, y_batches



    def __plot_flight__(self,
                        x:np.float64_3d[ax.sample, ax.time, ax.feature],
                        y:np.float64_2d[ax.sample, ax.feature]) -> None:

        NAME = "train_example"
        
        nb_view = 1 # default view is lat, lon,
        views = ["vertically normalized"]
        if ("random_angle_latitude"  in self.CTX["USED_FEATURES"] and \
            "random_angle_longitude" in self.CTX["USED_FEATURES"]):
            nb_view += 1
            views.insert(0, "default")
            
        
        
        fig, ax = plt.subplots(len(x), nb_view, figsize=(4*nb_view, 4*len(x)))
        if  (nb_view == 1):
            ax = [[a] for a in ax]
        
        for i in range(len(x)):
            for v in range(nb_view):
                view = views[v]
                if (view == "vertically normalized"):
                    lat = FG.lat(x[i])
                    lon = FG.lon(x[i])
                    
                if (view == "default"):
                    lat = FG.get(x[i], "random_angle_latitude")
                    lon = FG.get(x[i], "random_angle_longitude")
                
                ax[i][v].plot(lon, lat, color="tab:blue", linestyle="--")
                ax[i][v].scatter(lon, lat, color="tab:blue", marker="x")
                    
                if (i == 0):
                    ax[i][v].set_title(view)
                if (v == 0):
                    ax[i][v].set_ylabel("Interpolated" if y[i] > 0.5 else "Normal")
                    
                ax[i][v].set_aspect('equal', adjustable='datalim')
                    
        plt.tight_layout()
        plt.savefig("./_Artifacts/InterpolationDetector/train_ex.png")
        plt.close(fig)

        # PLT.figure (NAME, box[0], box[1], box[2], box[3], display_map=[[False]])
        # PLT.plot   (NAME, lon, lat, color="tab:blue", linestyle="--")
        # PLT.scatter(NAME, lon, lat, color="tab:blue", marker="x")
        # PLT.title(NAME, "interpolated traj" if y[0] > 0.5 else "normal trajectory")
        # PLT.show(NAME, "./_Artifacts/InterpolationDetector/train_ex.png")




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
            x_sample, y_sample = SU.gen_random_sample(CTX, self.x_test, self.y_test, self.PAD)
            x_batch[n] = x_sample
            y_batch[n] = y_sample

        batch_size = min(CTX["MAX_BATCH_SIZE"], len(x_batch))
        nb_batches = len(x_batch) // batch_size

        x_batch = self.__scalers_transform__(x_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, nb_batches, batch_size)
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

        if (len(df) <= 1):
            new_msg = U.df_to_feature_array(self.CTX, df.copy(), check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)
        else:
            idx = max(-len(df), -3)
            new_msg = U.df_to_feature_array(self.CTX, df[idx:], check_length=False)
            new_msg = fill_nan_2d(new_msg, self.PAD)[-1:]

        win = self.win_cache.extend(icao24, tag, new_msg, [len(df)-1] * len(new_msg))
        
        x_batch, y_batch = SU.alloc_batch(self.CTX, 1)

        t = len(win)-1
        if (t >= 0):
            while (t < len(win)-1 and FG.timestamp(win[t]) + self.CTX["HORIZON"] < message["timestamp"]):
                t += 1
                
        x_batch[0], _, valid = SU.gen_sample(
            self.CTX, [win], [None], self.PAD, 0, t, training=False)

        x_batch = self.__scalers_transform__(x_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, 1, 1)
        
        
        return x_batches[0], valid
