VERSION = "0.8.2"
HASH_TABLE_VERSION = "0.6.3"

from ._Utils_os_wrapper import os
from numpy_typing import np, ax
HERE = os.path.abspath(os.path.dirname(__file__))
from ._Utils_ProgressBar import ProgressBar
from ._Utils_ADSB_Streamer import cast_msg
from . import _Utils_Limits as Limits
import time
from . import _Utils_Color as C
from ._Utils_Color import prntC
from ._Utils_module import buildCTX
from ._Utils_ADSB_Streamer import streamer
import shutil


bar = ProgressBar()

def show_progress(block_num, block_size, total_size):
    if (bar.get_value() == 0):
        print()
        bar.reset(0, total_size)
    bar.update((block_num+1) * block_size)


def download_hash_table():
    URL = "http://51.77.221.41/hashtable.zip"
    import urllib.request
    import zipfile

    if (os.path.exists(HERE+"/hashtable.zip")):
        os.remove(HERE+"/hashtable.zip")

    update_version = False

    if not(os.path.exists(HERE+"/version")):
        if (os.path.exists(HERE+"/ReplaySolver/hashtable")):
            shutil.rmtree(HERE+"/ReplaySolver/hashtable/")
        update_version = True
    else:
        last_version = open(HERE+"/version", "r").read()

        if (last_version != HASH_TABLE_VERSION):
            if (os.path.exists(HERE+"/ReplaySolver/hashtable")):
                shutil.rmtree(HERE+"/ReplaySolver/hashtable/")

            update_version = True

    if (os.path.exists(HERE+"/ReplaySolver/hashtable")):
        return

    os.makedirs(HERE+"/ReplaySolver", exist_ok=True)
    prntC(C.INFO, "DOWNLOADING HASH TABLE ", end="...", flush=True)
    # create dir
    urllib.request.urlretrieve(URL, HERE+"/hashtable.zip", show_progress)
    prntC(C.INFO, "DOWNLOADING HASH TABLE", C.CYAN, " [DONE]")

    prntC(C.INFO, "UNZIPPING HASH TABLE ", end="...", flush=True)
    # Unzipping the file
    with zipfile.ZipFile(HERE+"/hashtable.zip", 'r') as zip_ref:
        zip_ref.extractall(HERE+"/ReplaySolver")
    os.remove(HERE+"/hashtable.zip")
    prntC("\r",C.INFO, "UNZIPPING HASH TABLE", C.CYAN, " [DONE]")

    if (update_version):
        file = open(HERE+"/version", "w")
        file.write(HASH_TABLE_VERSION)
        file.close()


download_hash_table()


def nan_to_null(value):
    if (np.isnan(value)):
        return None
    return value




# |====================================================================================================================
# | TRAJECTORY SEPARATOR
# |====================================================================================================================

# prntC(C.INFO, "LOADING TRAJECTORY SEPARATOR MODEL ", end="...", flush=True)

# from .E_Trainer_TrajectorySeparator_Trainer import Trainer as TrajectorySeparator
# from .B_Model_TrajectorySeparator import Model as MODEL_SEPARETOR
# from . import C_Constants_TrajectorySeparator as CTX_SEPARETOR
# from . import C_Constants_TrajectorySeparator_DefaultCTX as DefaultCTX_SEPARETOR

# CTX_TS = buildCTX(CTX_SEPARETOR, DefaultCTX_SEPARETOR)
# CTX_TS["LIB"] = True
# trajectorySeparator = TrajectorySeparator(CTX_TS, MODEL_SEPARETOR)
# trajectorySeparator.load(HERE+"/TrajectorySeparator")

# prntC("\r",C.INFO, "LOADING TRAJECTORY SEPARATOR MODEL", C.CYAN, " [DONE]")

# |====================================================================================================================
# | FLOODING DETECTION
# |====================================================================================================================

prntC(C.INFO, "LOADING FLOODING MODEL ", end="...", flush=True)

from .E_Trainer_FloodingSolver_Trainer import Trainer as FloodingSolver
from .B_Model_FloodingSolver import Model as MODEL_FLOODING
from . import C_Constants_FloodingSolver as CTX_FLOODING
from . import C_Constants_FloodingSolver_DefaultCTX as DefaultCTX_FLOODING

CTX_FS = buildCTX(CTX_FLOODING, DefaultCTX_FLOODING)
CTX_FS["LIB"] = True
floodingSolver = FloodingSolver(CTX_FS, MODEL_FLOODING)
floodingSolver.load(HERE+"/FloodingSolver")

prntC("\r",C.INFO, "LOADING FLOODING MODEL", C.CYAN, " [DONE]")

flooding_icao: "dict[str, int]" = {}

# |====================================================================================================================
# | REPLAY DETECTION
# |====================================================================================================================

prntC(C.INFO, "LOADING REPLAY MODEL ", end="...", flush=True)

from .E_Trainer_ReplaySolver_Trainer import Trainer as ReplaySolver
from .B_Model_ReplaySolver import Model as MODEL_REPLAY
from . import C_Constants_ReplaySolver as CTX_REPLAY
from . import C_Constants_ReplaySolver_DefaultCTX as DefaultCTX_REPLAY

CTX_RS = buildCTX(CTX_REPLAY, DefaultCTX_REPLAY)
CTX_RS["LIB"] = True
replaySolver = ReplaySolver(CTX_RS, MODEL_REPLAY)
replaySolver.load(HERE+"/ReplaySolver/hashtable")

prntC("\r",C.INFO, "LOADING REPLAY MODEL", C.CYAN, " [DONE]")

# |====================================================================================================================
# | SPOOFING DETECTION
# |====================================================================================================================

# prntC(C.INFO, "LOADING SPOOFING MODEL ", end="...", flush=True)

# from .E_Trainer_AircraftClassification_Trainer import Trainer as AircraftClassification
# from .B_Model_AircraftClassification import Model as MODEL_SPOOFING
# from . import C_Constants_AircraftClassification as CTX_SPOOFING
# from . import C_Constants_AircraftClassification_DefaultCTX as DefaultCTX_SPOOFING
from .D_DataLoader_AircraftClassification_Utils import getLabel


# CTX_AC = buildCTX(CTX_SPOOFING, DefaultCTX_SPOOFING)
# CTX_AC["LIB"] = True
# aircraftClassification = AircraftClassification(CTX_AC, MODEL_SPOOFING)
# aircraftClassification.load(HERE+"/AircraftClassification")

# prntC("\r",C.INFO, "LOADING SPOOFING MODEL", C.CYAN, " [DONE]")

# |====================================================================================================================
# | INTERPOLATION DETECTION
# |====================================================================================================================

# prntC(C.INFO, "LOADING INTERPOLATION MODEL ", end="...", flush=True)

# from .E_Trainer_InterpolationDetector_Trainer import Trainer as InterpolationDetector
# from .B_Model_InterpolationDetector import Model as MODEL_INTERP
# from . import C_Constants_InterpolationDetector as CTX_INTERP
# from . import C_Constants_InterpolationDetector_DefaultCTX as DefaultCTX_INTERP

# CTX_ID = buildCTX(CTX_INTERP, DefaultCTX_INTERP)
# CTX_ID["LIB"] = True
# interpolationDetector = InterpolationDetector(CTX_ID, MODEL_INTERP)
# interpolationDetector.load(HERE+"/InterpolationDetector")

# prntC("\r",C.INFO, "LOADING INTERPOLATION MODEL", C.CYAN, " [DONE]")



# |====================================================================================================================
# | ANOMALY DETECTION
# |====================================================================================================================


class AnomalyType:
    VALID = 0
    SPOOFING = 1
    FLOODING = 2
    REPLAY = 3,
    INTERP = 4,
    __INVALID__ = 1000



# def hash_message(message: "dict[str, str]") -> "int":
#     # message["icao24"] -> hex to int
#     return (hash(message["icao24"]) \
#                 + hash(message["timestamp"]) \
#                 + hash(message["latitude"]) \
#                 + hash(message["longitude"]) \
#                 + hash(message["altitude"])) % (Limits.INT_MAX)


# hash_table:"dict[int, list]" = {}
# def get_message_predictions(message: "dict[str, str]") -> "dict[str, str]":
#     h = hash_message(message)
#     data = hash_table.get(h, None)
#     if (data == None):
#         return None, h
#     return data[0], h

# def add_message_predictions(hash:int, message: "dict[str, str]") -> None:
#     global hash_table
#     if (hash == 0):
#         return
#     expiration = time.time() + 30 * 60 # 30 minutes
#     hash_table[hash] = [message, expiration]

# last_clean = 0
# def clean_hash_table() -> None:
#     global hash_table, last_clean
#     if (time.time() - last_clean < 1 * 60):
#         return

#     hash_table = {k:v for k, v in hash_table.items() if v[1] > time.time()}
#     last_clean = time.time()


def compress_messages(messages: "list[dict[str, str]]", compress=True, keep_debug=False) -> "list[dict[str, str]]":
    res = []

    if (compress):
        for i in range(len(messages)):
            res.append({
                "tag": messages[i].get("tag", ""),
                "anomaly": messages[i].get("anomaly", AnomalyType.VALID),
            })

        if keep_debug:
            for i in range(len(messages)):
                for k in messages[i].keys():
                    if (k.startswith("debug")):
                        res[i][k] = messages[i][k]
    else:
        res = messages
        if not keep_debug:
            for i in range(len(messages)):
                for k in list(messages[i].keys()):
                    if (k.startswith("debug")):
                        del res[i][k]

    return res

def message_subset(messages: "list[dict[str, str]]") -> "tuple[list[dict[str, str]], list[int]]":
    indices = [i for i in range(len(messages)) if messages[i]["anomaly"] == AnomalyType.VALID]
    sub = [messages[i] for i in indices]
    return sub, indices


def clear_cache(flight_icao:str, tag="0") -> None:
    traj = streamer.get(flight_icao, tag)
    if (traj is None): return
    prntC(C.INFO, "Cleaning ", C.CYAN, flight_icao, "...", flush=True)
    streamer.__remove_trajectory__(traj)


def predict(messages: "list[dict[str, str]]", compress:bool=True, debug:bool=False) -> "list[dict[str, str]]":

    # stream message
    splits = []
    split_caches = np.zeros(len(messages), dtype=bool)
    for i in range(len(messages)):
        messages[i]["tag"] = messages[i].get("tag", "0")
        messages[i]["anomaly"] = AnomalyType.VALID
        messages[i] = cast_msg(messages[i])
        split_caches[i] = streamer.add(messages[i])

        if (split_caches[i]):
            splits.append(i)

    # remove consecutive splits with reversed for
    for i in range(len(splits)-1, 0, -1):
        if (splits[i] == splits[i-1]+1):
            splits.pop(i)

    if (len(splits) == 0):
        return __predict__(messages, compress=compress, debug=debug)

    splits.append(len(messages))
    res = []
    if (splits[0] != 0):
        res += __predict__(messages[:splits[0]], compress=compress, debug=debug)

    for i in range(1, len(splits)):
        for t in range(splits[i-1], splits[i]):
            if (split_caches[t]):
                streamer.split_cache(messages[t])

        res += __predict__(messages[splits[i-1]:splits[i]], compress=compress, debug=debug)
    return res




def __predict__(messages: "list[dict[str, str]]", compress:bool=True, debug:bool=False) -> "list[dict[str, str]]":


    # check validity of messages
    for i in range(len(messages)):
        if (messages[i].get("icao24", None) == None or messages[i].get("timestamp", None) == None or \
            messages[i].get("latitude", None) == None or messages[i].get("longitude", None) == None):
            messages[i]["alteration"] =AnomalyType.__INVALID__


    # separate trajectories with duplicated icao24
    # sub_msg, indices = message_subset(messages)
    # sub_icaos = trajectorySeparator.predict(sub_msg)
    # for i in range(len(sub_msg)):
    #     messages[indices[i]]["tag"] = sub_icaos[i]


    # check for replay anomalies
    sub_msg, indices = message_subset(messages)
    matches = replaySolver.predict(sub_msg)
    for i in range(len(indices)):
        if (matches[i] != "unknown"):
            messages[indices[i]]["anomaly"] = AnomalyType.REPLAY
            
    # # check for interpolation anomalies
    # sub_msg, indices = message_subset(messages)
    # _, _, anomaly = interpolationDetector.predict(sub_msg)
    # for i in range(len(indices)):
    #     if (anomaly[i]):
    #         messages[indices[i]]["anomaly"] = AnomalyType.INTERP

    
    # check for flooding anomalies
    sub_msg, indices = message_subset(messages)
    y_, loss, anomaly = floodingSolver.predict(sub_msg)
    for i in range(len(indices)):
        # if (loss[i] > CTX_FS["THRESHOLD"]):
        #     messages[indices[i]]["anomaly"] = AnomalyType.FLOODING
        messages[indices[i]]["debug_flooding_loss"] = loss[i]
        messages[indices[i]]["debug_lat_lon"] = [nan_to_null(y_[i][0]), nan_to_null(y_[i][1])]

        if (anomaly[i]):
            messages[indices[i]]["anomaly"] = AnomalyType.FLOODING


    # check for spoofing
    # filter messages having unknown icao24
    # true_labels = get_true_aircraft_type(messages)
    # for i in range(len(messages)):
    #     if (true_labels[i] == 0 and messages[i]["anomaly"] == AnomalyType.VALID):
    #         messages[i]["anomaly"] = AnomalyType.__INVALID__

    # sub_msg, indices = message_subset(messages)
    # _, label_proba = aircraftClassification.predict(sub_msg)
    # spoofing = is_spoofing(true_labels[indices], label_proba)
    # for i in range(len(indices)):
    #     if (spoofing[i]):
    #         messages[indices[i]]["anomaly"] = AnomalyType.SPOOFING

    #     messages[indices[i]]["debug_spoofing_proba"] = label_proba[i].tolist()


    # # save messages predictions in case of a future request
    # for i in range(len(messages)):
    #     add_message_predictions(hashes[i], response[i])

    for i in range(len(messages)):
        if (messages[i]["anomaly"] == AnomalyType.__INVALID__):
            messages[i]["anomaly"] = AnomalyType.VALID


    return compress_messages(messages, compress=compress, keep_debug=debug)




# |====================================================================================================================
# | UTILS
# |====================================================================================================================


def get_base_icaos(messages: "list[dict[str, str]]") -> "list[str]":
    icaos = [messages[i]["icao24"] for i in range(len(messages))]
    return [icaos[i].split("_")[0] if ("_" in icaos[i]) else icaos[i] for i in range(len(icaos))]

# def get_true_aircraft_type(messages: "list[dict[str, str]]") -> np.int32_1d[ax.sample]:
#     icaos = get_base_icaos(messages)
#     return np.array([getLabel(CTX_AC, icaos[i]) for i in range(len(icaos))], dtype=np.int32)


# def get_pred_aircraft_type(proba: "np.ndarray") -> "list[int]":
#     argmax = np.argmax(proba, axis=1)
#     confidence = np.nan_to_num([proba[i][argmax[i]] for i in range(len(argmax))])
#     return [0 if confidence[i] <= 0.5 else CTX_AC["USED_LABELS"][argmax[i]] for i in range(len(argmax))]

# def is_spoofing(true_labels: "np.int32_1d[ax.sample]", predictions: np.float64_2d[ax.feature, ax.label])\
#         -> "list[bool]":
#     if (len(true_labels) == 0): return []
#     pred_labels = get_pred_aircraft_type(predictions)
#     return [pred_labels[i] != 0 and true_labels[i] != 0 and pred_labels[i] != true_labels[i]
#         for i in range(len(true_labels))]

