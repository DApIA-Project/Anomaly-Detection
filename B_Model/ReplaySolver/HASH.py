from numpy_typing import np, ax
from _Utils.os_wrapper import os

from B_Model.AbstractModel import Model as AbstactModel

from _Utils.ProgressBar import ProgressBar
import B_Model.ReplaySolver.Utils.hashing as hashing


import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.hashtable import FileMultiHashTable


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================


PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
BAR = ProgressBar()


class Model(AbstactModel):

    name = "HASH"

    hashtable:"dict[int, list[str]]"

# |====================================================================================================================
# |    INITIALIZATION
# |====================================================================================================================

    def __init__(self, CTX:dict) -> None:
        # Public attributes
        self.CTX = CTX
        hashing.init_hash(CTX["HISTORY"])
        self.hashtable = None

        if (CTX["EPOCHS"] > 0):
            self.hashtable = FileMultiHashTable(f"./_Artifacts/{PBM_NAME}/hashtable", dtype=str, content_size=45)
            self.hashtable.clear()



# |====================================================================================================================
# |    PREDICTION
# |====================================================================================================================


    def predict(self, x:np.float64_3d[ax.sample, ax.time, ax.feature]) -> "list[list[str]]":

        if (x.shape[1] != self.CTX["HISTORY"]):
            raise ValueError("The input shape is not correct !")

        y_ = []
        for b in range(len(x)):
            fps = hashing.sub_fingerprint(x[b, :, 0])
            hashes = hashing.hash(fps)
            matches = hashing.match(hashes, self.hashtable)
            unique_matches = list(set(matches))
            y_.append(unique_matches)

        return y_


    def compute_loss(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], y:np.str_1d[ax.sample]) -> """
            tuple[float, list[list[str]]]""":

        y_ = self.predict(x)

        accuracy, size = 0, 0
        for s in range(len(y_)):
            if (y[s] == np.nan):
                continue

            # if (y_[s][0] == "skip"):
            #     continue

            nb_correct = 0
            for pred in y_[s]:
                if (pred == y[s]):
                    nb_correct += 1
                    break

            if (len(y_[s]) > 0):
                accuracy += nb_correct
            size += 1

        return accuracy / size, y_



    def training_step(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], y:np.str_1d[ax.sample]) -> None:
        for b in range(len(x)):
            if (y[b] == 'nan'):
                continue

            fps = hashing.sub_fingerprint(x[b, :, 0])
            hashes = hashing.hash(fps)

            for h in range(len(hashes)):
                self.hashtable[hashes[h]] = y[b]

    def nb_parameters(self):
        return 0
    

    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """


    def get_variables(self):
        """
        Return the variables of the model
        """
        return None


    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        # self.hashtable = variables
        self.hashtable = FileMultiHashTable(variables, dtype=str, content_size=45)

