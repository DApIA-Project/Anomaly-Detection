from _Utils.numpy import np, ax
from _Utils.os_wrapper import os

from B_Model.AbstractModel import Model as AbstactModel

from _Utils.ProgressBar import ProgressBar
from B_Model.ReplaySolver.Utils import hashing


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
        self.hashtable = {}



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
            y_.append(matches)

        return y_



    def compute_loss(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], y:np.str_1d[ax.sample]) -> """
            tuple[float, list[list[str]]]""":

        y_ = self.predict(x)

        accuracy, size = 0, 0
        for s in range(len(y_)):
            if (len(y_[s]) == 0):
                y_[s].append("unknown")

            if (y[s] == np.nan):
                continue

            nb_correct = 0
            for pred in y_[s]:
                if (pred == y[s]):
                    nb_correct += 1

            if (len(y_[s]) > 0):
                accuracy += nb_correct / len(y_[s])
            size += 1

        return accuracy / size, y_



    def training_step(self, x:np.float64_3d[ax.sample, ax.time, ax.feature], y:np.str_1d[ax.sample]) -> None:
        for b in range(len(x)):
            if (y[b] == np.nan):
                continue

            fps = hashing.sub_fingerprint(x[b, :, 0])
            hashes = hashing.hash(fps)

            for h in range(len(hashes)):
                matches = self.hashtable.get(hashes[h], None)
                if (matches is None):
                    self.hashtable[hashes[h]] = set([y[b]])
                else:
                    self.hashtable[hashes[h]].add(y[b])




    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """


    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.hashtable


    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        self.hashtable = variables
