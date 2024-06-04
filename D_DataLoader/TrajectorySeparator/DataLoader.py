import pandas as pd
from _Utils.numpy import np, ax

import _Utils.Color as C
from _Utils.Color import prntC
import _Utils.FeatureGetter as FG
import _Utils.Limits as Limits
from _Utils.ADSB_Streamer import Streamer
from _Utils.ProgressBar import ProgressBar

import D_DataLoader.Utils as U
import D_DataLoader.TrajectorySeparator.Utils as SU
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
# |     INITIALISATION : LOADING RAW DATASET FROM DISK (NOT USED BECAUSE THERE IS NO TRAINING)
# |====================================================================================================================

    def __init__(self, CTX:dict) -> None:
        self.CTX = CTX
        self.streamer:StreamerInterface = StreamerInterface(self)



    def __load_dataset__(self, CTX:dict, path:str) -> "tuple[np.ndarray, np.ndarray, pd.DataFrame]":
        filenames = U.list_flights(path, limit=Limits.INT_MAX)

        # merge all files into one dataframe (unsplit flights)
        df = pd.DataFrame()
        icao_codes = set()
        for f in range(len(filenames)):
            messages = U.read_trajectory(filenames[f])
            # check if the icao24 is already in the set
            icao = messages.iloc[0]['icao24']
            if (icao in icao_codes):
                if ("_" in icao): icao = icao.split("_")[0]
                icao = icao + "_" + str(len(icao_codes))

            messages['y'] = icao
            icao_codes.add(icao)
            df = pd.concat([df, messages], ignore_index=True)

        df = df.sort_values(by=['timestamp'])
        df = df.reset_index(drop=True)
        y = df['y'].to_numpy()
        df.drop(columns=['y'], inplace=True)
        x = U.df_to_feature_array(CTX, df, check_length=False)

        return x, y, df


    def genEpochTrain(self) -> None:
        prntC(C.WARNING, "No training needed for TrajectorySeparator")




# |====================================================================================================================
# | STREAMING ADS-B MESSAGE FOR EVAL UNDER REAL CONDITIONS
# |====================================================================================================================


class StreamerInterface:
    def __init__(self, dl:DataLoader) -> None:
        self.dl = dl
        self.CTX = dl.CTX

    def stream(self, x:"dict[str, object]") -> "list[np.ndarray]":
        # in this problem, tag corresponds to the icao24 + a number to differentiate multiple aircrafts
        # with the same icao24
        # tag is icao if the message hasn't been associated to any aircraft yet

        tag = x.get("tag", x['icao24'])
        icao = x.get("icao24", tag)

        if (tag != icao):
            STREAMER.add(x, tag=tag)
        else:
            prntC(C.WARNING, "No tag provided for message : ", x['icao24'])


    def clear(self)-> None:
        STREAMER.clear()


    def get_flights_with_icao(self, icao:str, timestamp:int) -> """tuple[
        list[np.float64_2d[ax.time, ax.feature]],
        list[str]]""":

        tags = STREAMER.get_tags_for_icao(icao)
        if (len(tags) == 0):
            return [], []

        COL = self.CTX["USED_FEATURES"]
        x, tags = zip(*[[STREAMER.get(tag).getColumns(COL), tag] for tag in tags])

        # remove trajectory witch already has a message at this timestamp
        x_tag = [(x[i], tags[i]) for i in range(len(x))
                        if len(x[i]) <= 1 or int(x[i][-1, -1]) < timestamp]

        if (len(x_tag) > 0):
            x, tags = zip(*x_tag)
            x, tags = list(x), list(tags)

        return list(x), list(tags)




