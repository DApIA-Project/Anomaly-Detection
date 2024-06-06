

from _Utils.numpy import np, ax
import matplotlib.pyplot as plt
import os

import _Utils.FeatureGetter as FG
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.Scaler3D import StandardScaler3D, fill_nan_3d, StandardScaler2D,MinMaxScaler2D
from _Utils.SparceLabelBinarizer import SparceLabelBinarizer
from _Utils.ProgressBar import ProgressBar
from _Utils.Limits import *
from _Utils.DebugGui import GUI

import D_DataLoader.Utils as U
import D_DataLoader.ReplaySolver.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


###################################################
# GLOBAL VARIABLES
###################################################

BAR = ProgressBar()


###################################################
# DATA LOADER
###################################################

class DataLoader(AbstractDataLoader):

    ###################################################
    # LOADING DATASET FROM DISK
    ###################################################

    def __init__(self, CTX, path="") -> None:
        self.CTX = CTX
        self.PAD = None

        training = (CTX["EPOCHS"] and path != "")

        if (training):
            self.x, self.files = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(self.x, self.files)
        else:
            prntC(C.INFO, "Training, deactivated, only evaluation will be launched.")
            prntC(C.WARNING, "Make sure everything is loaded from the disk, especially the PAD values.")
            prntC(C.WARNING, "i'm not sure everything will work as expected : TODO CHECK !")



    def __load_dataset__(self, CTX, path):
        is_folder = os.path.isdir(path)
        if (is_folder):
            filenames = U.list_flights(path, limit=INT_MAX) #INT_MAX
            prntC(C.INFO, "Dataset loading")
        else: # allow to load a single file (can be useful for eval)
            path = path.split("/")
            filenames = [path[-1]]
            path = "/".join(path[:-1])

        BAR.reset(max=len(filenames))

        x = []
        for f in range(len(filenames)):
            df = U.read_trajectory(path, filenames[f])
            array = U.df_to_feature_array(CTX, df)

            x.append(array)

            if (is_folder): BAR.update(f+1)

        if (self.PAD is None): self.PAD = U.genPadValues(CTX, x)
        x = fill_nan_3d(x, self.PAD)

        prntC()
        return x, filenames



    ###################################################
    # SCALERS
    ###################################################

    def __scalers_transform__(self, CTX, x_batches):
        return x_batches


    ###################################################
    # UTILS
    ###################################################

    def __reshape__(self, CTX, x_batches, y_batches, x_alters=None):
        if (x_alters is not None):
            return [x_batches], [y_batches], [x_alters]
        return [x_batches], [y_batches]


    ###################################################
    # GENERATING TRAINING SET
    ###################################################

    def genEpochTrain(self, ):

        CTX = self.CTX

        if (CTX["NB_BATCH"] == 1 and CTX["BATCH_SIZE"] == None):
            x_batches = self.x_train.copy()
            y_batches = self.y_train.copy()

            x_batches =\
                self.__scalers_transform__(CTX, x_batches)
            return self.__reshape__(CTX, x_batches, y_batches)

        prntC(C.ERROR, "Not implemented yet : impossible to generate multiple batches for now")
        exit(1)


    ###################################################
    # GENERATING TEST SET
    ###################################################

    def genEpochTest(self):

        CTX = self.CTX
        NB_TEST = 60

        if (CTX["NB_BATCH"] == 1 and CTX["BATCH_SIZE"] == None):
            x_batches = []
            y_batches = []

            for _ in range(NB_TEST):
                x, y, known = SU.getRandomFlight(CTX, self.x_train, self.y_train, self.x_test, self.y_test)

                if known:
                    x_batches.insert(0, x)
                    y_batches.insert(0, y)
                else:
                    x_batches.append(x)
                    y_batches.append(y)

            x_batches =\
                self.__scalers_transform__(CTX, x_batches)
            return self.__reshape__(CTX, x_batches, y_batches)


        prntC(C.ERROR, "Not implemented yet : impossible to generate multiple batches for now")
        exit(1)


    ###################################################
    # GENERATING EVAL SET
    ###################################################

    def genEval(self) -> "tuple[np.ndarray, np.ndarray, list[list[str]]]":
        CTX = self.CTX
        if (CTX["NB_BATCH"] == 1 and CTX["BATCH_SIZE"] == None):
            x_batches = []
            x_alters = []
            y_batches = []

            for i in range(200):
                alter = (i%2 == 0)
                if alter:
                    f = np.random.randint(len(self.x_train))
                    x, type = SU.alter(self.x_train[f], CTX)
                    x_batches.insert(0, x)
                    x_alters.insert(0, type)
                    y_batches.insert(0, self.files[f])

                else:
                    f = np.random.randint(len(self.x_test))
                    x_batches.append(self.x_test[f])
                    x_alters.append("None")
                    y_batches.append("Unknown-flight")

            x_batches =\
                self.__scalers_transform__(CTX, x_batches)
            x_batches, y_batches, x_alters =\
                self.__reshape__(CTX, x_batches, y_batches, x_alters)
            return x_batches, y_batches, x_alters

        prntC(C.ERROR, "Not implemented yet : impossible to generate multiple batches for now")
        exit(1)
