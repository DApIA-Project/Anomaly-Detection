
import _Utils.Color as C
from _Utils.Color import prntC
import _Utils.FeatureGetter as FG

import D_DataLoader.Utils as U
import D_DataLoader.TrajectorySeparator.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader

import os
import pandas as pd


class DataLoader(AbstractDataLoader):

    def __init__(self, CTX) -> None:    
        self.CTX = CTX 


    def __load_dataset__(self, CTX, path):
        filenames = os.listdir(path)
        filenames = [f for f in filenames if f.endswith(".csv")]

        df = pd.DataFrame()
        for f in range(len(filenames)):
            messages = U.read_trajectory(path, filenames[f])

            df = pd.concat([df, messages], ignore_index=True)

        df = df.sort_values(by=['timestamp'])
        df = df.reset_index(drop=True)


        x, y = [], []
        base_icao = df['icao24'].iloc[0]
        if ("_" in base_icao): base_icao = base_icao.split("_")[0]
        icaos_tab = {}
        icaos = df['icao24'].unique()
        for icao in icaos:
            icaos_tab[icao] = base_icao + "_" + str(len(icaos_tab))

        for i in range(len(df)):
            y.append(icaos_tab[df['icao24'].iloc[i]])

        df.drop(columns=['icao24'], inplace=True)
            
        x = U.dfToFeatures(df, CTX, __EVAL__=True)
        
        return x, y, df

            








    def genEpochTrain(self):
        prntC(C.WARNING, "No training needed for TrajectorySeparator")

    def genEval(self, path):
        x, y, df = self.__load_dataset__(self.CTX, path)
        return x, y, df