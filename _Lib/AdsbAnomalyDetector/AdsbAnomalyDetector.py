from ._Utils_os_wrapper import os
from numpy_typing import np, ax
HERE = os.path.abspath(os.path.dirname(__file__))
from ._Utils_module import module_to_dict
from ._Utils_ADSB_Streamer import cast_msg
from . import _Utils_Limits as Limits
import time
from . import _Utils_Color as C
from ._Utils_Color import prntC
from ._Utils_module import buildCTX
from ._Utils_ADSB_Streamer import streamer






# |====================================================================================================================
# | TRAJECTORY SEPARATOR
# |====================================================================================================================

# prntC(C.INFO, "LOADING TRAJECTORY SEPARATOR MODEL ", end="...", flush=True)

# from .E_Trainer_TrajectorySeparator_Trainer import Trainer as TrajectorySeparator
# from .B_Model_TrajectorySeparator import Model as MODEL_SEPARETOR
# from . import C_Constants_TrajectorySeparator as CTX_SEPARETOR
# from . import C_Constants_TrajectorySeparator_DefaultCTX as DefaultCTX_SEPARETOR

# CTX_TS = buildCTX(CTX_SEPARETOR, DefaultCTX_SEPARETOR)
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
floodingSolver = FloodingSolver(CTX_FS, MODEL_FLOODING)
floodingSolver.load(HERE+"/FloodingSolver")

prntC("\r",C.INFO, "LOADING FLOODING MODEL", C.CYAN, " [DONE]")

# |====================================================================================================================
# | REPLAY DETECTION
# |====================================================================================================================

prntC(C.INFO, "LOADING REPLAY MODEL ", end="...", flush=True)

from .E_Trainer_ReplaySolver_Trainer import Trainer as ReplaySolver
from .B_Model_ReplaySolver import Model as MODEL_REPLAY
from . import C_Constants_ReplaySolver as CTX_REPLAY
from . import C_Constants_ReplaySolver_DefaultCTX as DefaultCTX_REPLAY

CTX_RS = buildCTX(CTX_REPLAY, DefaultCTX_REPLAY)
replaySolver = ReplaySolver(CTX_RS, MODEL_REPLAY)
replaySolver.load(HERE+"/ReplaySolver/hashtable")

prntC("\r",C.INFO, "LOADING REPLAY MODEL", C.CYAN, " [DONE]")

# |====================================================================================================================
# | SPOOFING DETECTION
# |====================================================================================================================

prntC(C.INFO, "LOADING SPOOFING MODEL ", end="...", flush=True)

from .E_Trainer_AircraftClassification_Trainer import Trainer as AircraftClassification
from .B_Model_AircraftClassification import Model as MODEL_SPOOFING
from . import C_Constants_AircraftClassification as CTX_SPOOFING
from . import C_Constants_AircraftClassification_DefaultCTX as DefaultCTX_SPOOFING
from .D_DataLoader_AircraftClassification_Utils import getLabel


CTX_AC = buildCTX(CTX_SPOOFING, DefaultCTX_SPOOFING)
aircraftClassification = AircraftClassification(CTX_AC, MODEL_SPOOFING)
aircraftClassification.load(HERE+"/AircraftClassification")

prntC("\r",C.INFO, "LOADING SPOOFING MODEL", C.CYAN, " [DONE]")


def hash_message(message: "dict[str, str]") -> "int":
    # message["icao24"] -> hex to int
    return (hash(message["icao24"]) \
                + hash(message["timestamp"]) \
                + hash(message["latitude"]) \
                + hash(message["longitude"]) \
                + hash(message["altitude"])) % (Limits.INT_MAX)


hash_table:"dict[int, list]" = {}
def get_message_predictions(message: "dict[str, str]") -> "dict[str, str]":
    h = hash_message(message)
    data = hash_table.get(h, None)
    if (data == None):
        return None, h
    return data[0], h

def add_message_predictions(hash:int, message: "dict[str, str]") -> None:
    global hash_table
    if (hash == 0):
        return
    expiration = time.time() + 30 * 60 # 30 minutes
    hash_table[hash] = [message, expiration]

def compress_messages(messages: "list[dict[str, str]]"):
    res = []
    for i in range(len(messages)):
        res.append({
            "tag": messages[i].get("tag", ""),
            "anomaly": messages[i].get("anomaly", UNSET),
        })
    return res

last_clean = 0
def clean_hash_table() -> None:
    global hash_table, last_clean
    if (time.time() - last_clean < 1 * 60):
        return

    hash_table = {k:v for k, v in hash_table.items() if v[1] > time.time()}
    last_clean = time.time()


def message_subset(messages: "list[dict[str, str]]") -> "tuple[list[dict[str, str]], list[int]]":
    indices = [i for i in range(len(messages)) if messages[i]["anomaly"] == UNSET]
    sub = [messages[i] for i in indices]
    return sub, indices


UNSET = 0
SPOOFING = 1
FLOODING = 2
REPLAY = 3
INVALID = 1000


def predict(messages: "list[dict[str, str]]", compress=True) -> "list[dict[str, str]]":

    # stream messages
    for i in range(len(messages)):
        messages[i]["tag"] = messages[i].get("tag", "")
        messages[i]["anomaly"] = UNSET

        streamer.add(messages[i])


    # check validity of messages
    for i in range(len(messages)):
        if (messages[i].get("icao24", None) == None or messages[i].get("timestamp", None) == None or \
            messages[i].get("latitude", None) == None or messages[i].get("longitude", None) == None):
            messages[i]["alteration"] = INVALID


    # # separate trajectories with duplicated icao24
    # sub_msg, indices = message_subset(messages)
    # sub_icaos = trajectorySeparator.predict(sub_msg)
    # for i in range(len(sub_msg)):
    #     messages[indices[i]]["tag"] = sub_icaos[i]


    # check for replay anomalies
    sub_msg, indices = message_subset(messages)
    matches = replaySolver.predict(sub_msg)
    for i in range(len(indices)):
        if (matches[i] != "unknown"):
            print("REPLAY")
            messages[indices[i]]["anomaly"] = REPLAY


    # # check for flooding anomalies
    # sub_msg, indices = message_subset(messages)
    # _, loss = floodingSolver.predict(sub_msg)
    # for i in range(len(indices)):
    #     if (loss[i] > CTX_FS["THRESHOLD"]):
    #         messages[indices[i]]["anomaly"] = FLOODING


    # # check for spoofing
    # # filter messages having unknown icao24
    true_labels = get_true_aircraft_type(messages)
    for i in range(len(messages)):
        if (true_labels[i] == 0 and messages[i]["anomaly"] == UNSET):
            messages[i]["anomaly"] = INVALID

    sub_msg, indices = message_subset(messages)
    _, label_proba = aircraftClassification.predict(sub_msg)
    spoofing = is_spoofing(true_labels[indices], label_proba)
    for i in range(len(indices)):
        if (spoofing[i]):
            print("SPOOFING")
            messages[indices[i]]["anomaly"] = SPOOFING


    # save messages predictions in case of a future request
    # for i in range(len(messages)):
    #     add_message_predictions(hashes[i], response[i])

    for i in range(len(messages)):
        if (messages[i]["anomaly"] == INVALID):
            messages[i]["anomaly"] = UNSET
    if compress:
        messages = compress_messages(messages)

    return messages



# |====================================================================================================================
# | UTILS
# |====================================================================================================================


def get_base_icaos(messages: "list[dict[str, str]]") -> "list[str]":
    icaos = [messages[i]["icao24"] for i in range(len(messages))]
    return [icaos[i].split("_")[0] if ("_" in icaos[i]) else icaos[i] for i in range(len(icaos))]

def get_true_aircraft_type(messages: "list[dict[str, str]]") -> np.int32_1d[ax.sample]:
    icaos = get_base_icaos(messages)
    return np.array([getLabel(CTX_AC, icaos[i]) for i in range(len(icaos))], dtype=np.int32)


def get_pred_aircraft_type(proba: "np.ndarray") -> "list[int]":
    argmax = np.argmax(proba, axis=1)
    confidence = np.nan_to_num([proba[i][argmax[i]] for i in range(len(argmax))])
    return [0 if confidence[i] <= 0.5 else CTX_AC["USED_LABELS"][argmax[i]] for i in range(len(argmax))]

def is_spoofing(true_labels: "np.int32_1d[ax.sample]", predictions: np.float64_2d[ax.feature, ax.label])\
        -> "list[bool]":
    if (len(true_labels) == 0): return []
    pred_labels = get_pred_aircraft_type(predictions)
    return [pred_labels[i] != 0 and true_labels[i] != 0 and pred_labels[i] != true_labels[i]
        for i in range(len(true_labels))]