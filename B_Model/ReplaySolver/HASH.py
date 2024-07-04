from _Utils.numpy import np, ax
from _Utils.os_wrapper import os

from B_Model.AbstractModel import Model as AbstactModel

from _Utils.ProgressBar import ProgressBar
import B_Model.ReplaySolver.Utils.hashing as hashing


import _Utils.Color as C
from   _Utils.Color import prntC


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================


PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
BAR = ProgressBar()

class FileHashTable:

    FILE_COUNT = 2**16
    PATH = "./_Artifacts/HASH/hashtable/"
    MAX_HASH = 2**self.CTX["HISTORY"]

    def __init__(self, CTX:dict) -> None:
        self.CTX = CTX

    def set_path(self, path:str) -> None:
        self.PATH = path

    def __read_assoc_file__(self, l) -> "dict[int, list[str]]":
        path = f"{self.PATH}assoc.txt"
        # assoc file is build as follow:
        # nth line = filename (max 128 char)
        self.assoc_file = open(path, "r")
        self.assoc_file.seek(l * 128)
        filename = self.assoc_file.read(128).strip()
        self.assoc_file.close()

    def __write_assoc_file__(self, filename:str) -> None:
        filename = filename.ljust(128, " ")
        self.assoc_file = open(f"{self.PATH}assoc.txt", "a")
        self.assoc_file.write(filename)
        self.assoc_file.close()




    def get(self, key:int) -> "list[str]":
        if (key >= self.MAX_HASH or key < 0):
            prntC(C.WARNING, f"Invalid key {key} !")
            return []

        # file_id | line_id
        # 16 bits  | 16 bits
        file_id = key // self.FILE_COUNT
        bit_id = key % self.FILE_COUNT

        if (not os.path.exists(f"{self.PATH}{file_id}.txt")):
            return []

        file = open(f"{self.PATH}{file_id}.txt", "rb")
        # move cursor
        file.seek(bit_id * 4)
        line = int.from_bytes(file.read(4), "big")
        file.close()
        return self.__read_assoc_file__(line)


    def set(self, key:int, value:"list[str]") -> None:
        if (key >= self.MAX_HASH or key < 0):
            prntC(C.WARNING, f"Invalid key {key} !")
            return

        # file_id | line_id
        # 16 bits  | 16 bits
        file_id = key // self.FILE_COUNT
        bit_id = key % self.FILE_COUNT

        if (not os.path.exists(f"{self.PATH}{file_id}.txt")):
            pass

        file = open(f"{self.PATH}{file_id}.txt", "rb+")
        # move cursor
        file.seek(bit_id * 4)
        line = int.from_bytes(file.read(4), "big")
        file.close()
        return self.__write_assoc_file__(line)




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
