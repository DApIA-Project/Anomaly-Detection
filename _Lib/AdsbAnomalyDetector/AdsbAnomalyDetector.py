from ._Utils_os_wrapper import os
from ._Utils_numpy import np, ax
HERE = os.path.abspath(os.path.dirname(__file__))
from ._Utils_module import module_to_dict
from ._Utils_ADSB_Streamer import cast_msg
from . import _Utils_Limits as Limits
import time
from . import _Utils_Color as C
from ._Utils_Color import prntC
from ._Utils_module import buildCTX


# |====================================================================================================================
# | SPOOFING DETECTION
# |====================================================================================================================

prntC(C.INFO, "LOADING SPOOFING MODEL ", end="...", flush=True)

from .E_Trainer_AircraftClassification_Trainer import Trainer as AircraftClassification
from .B_Model_AircraftClassification import Model as MODEL_SPOOFING
from . import C_Constants_AircraftClassification as CTX_SPOOFING
from . import C_Constants_AircraftClassification_DefaultCTX as DefaultCTX_SPOOFING

CTX_AC = buildCTX(CTX_SPOOFING, DefaultCTX_SPOOFING)
aircraftClassification = AircraftClassification(CTX_AC, MODEL_SPOOFING)
aircraftClassification.load(HERE+"/AircraftClassification")

prntC("\r",C.INFO, "LOADING SPOOFING MODEL", C.CYAN, " [DONE]")


# |====================================================================================================================
# | TRAJECTORY SEPARATOR
# |====================================================================================================================

prntC(C.INFO, "LOADING TRAJECTORY SEPARATOR MODEL ", end="...", flush=True)

from .E_Trainer_TrajectorySeparator_Trainer import Trainer as TrajectorySeparator
from .B_Model_TrajectorySeparator import Model as MODEL_SEPARETOR
from . import C_Constants_TrajectorySeparator as CTX_SEPARETOR
from . import C_Constants_TrajectorySeparator_DefaultCTX as DefaultCTX_SEPARETOR
from .D_DataLoader_AircraftClassification_Utils import getLabel

CTX_TS = buildCTX(CTX_SEPARETOR, DefaultCTX_SEPARETOR)
trajectorySeparator = TrajectorySeparator(CTX_TS, MODEL_SEPARETOR)
trajectorySeparator.load(HERE+"/TrajectorySeparator")

prntC("\r",C.INFO, "LOADING TRAJECTORY SEPARATOR MODEL", C.CYAN, " [DONE]")

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


# def get(lst, bool_arr) -> list:
#     return [lst[i] for i in range(len(lst)) if bool_arr[i]]

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


last_clean = 0
def clean_hash_table() -> None:
    global hash_table, last_clean
    if (time.time() - last_clean < 1 * 60):
        return

    hash_table = {k:v for k, v in hash_table.items() if v[1] > time.time()}
    last_clean = time.time()


def message_subset(messages: "list[dict[str, str]]", indices: "list[bool]") -> "list[dict[str, str]]":
    return [messages[i] for i in range(len(messages)) if indices[i]]

def predict(messages: "list[dict[str, str]]", compress=True) -> "list[dict[str, str]]":
    # print("i")
    clean_hash_table()

    # load messages predictions if they has already been computed
    message_filter = np.ones(len(messages), dtype=bool)

    response = [{
        "tag": messages[i].get("tag", messages[i]["icao24"]),
        "spoofing": False,
        "replay": False,
        "flooding": False
    } for i in range(len(messages))]

    if not compress:
        for i in range(len(messages)):
            for key in messages[i]:
                response[i][key] = messages[i][key]


    # check validity of messages
    for i in range(len(messages)):
        if (messages[i].get("icao24", None) == None or messages[i].get("timestamp", None) == None or \
            messages[i].get("latitude", None) == None or messages[i].get("longitude", None) == None):
            message_filter[i] = False


    # check if the message has already been processed
    # + cast other messages to the right type
    hashes = np.zeros(len(messages), dtype=np.uint32)
    for i in range(len(messages)):
        if (not message_filter[i]):
            continue

        y_, hashes[i] = get_message_predictions(messages[i])
        if (y_ != None):
            message_filter[i] = False
            response[i] = y_
        else:
            messages[i] = {col:cast_msg(col,  messages[i].get(col, np.nan)) for col in messages[i]}



    # separate trajectories with duplicated icao24
    sub_msg = message_subset(messages, message_filter)
    # sub_icaos = trajectorySeparator.predict(sub_msg)
    # sub_i = 0
    # for i in range(len(messages)):
    #     if (message_filter[i]):
    #         response[i]["tag"] = sub_icaos[sub_i]

    #         sub_i += 1


    # print("a")


    # check for replay anomalies
    # matches = replaySolver.predict(sub_msg)
    # sub_i = 0
    # for i in range(len(messages)):
    #     if (message_filter[i]):
    #         response[i]["replay"] = (matches[sub_i] != "none" and matches[sub_i] != "unknown")

    #         message_filter[i] = not(response[i]["replay"])
    #         sub_i += 1

    # print("b")

    # check for flooding anomalies
    sub_msg = message_subset(messages, message_filter)
    y_, loss = floodingSolver.predict(sub_msg)
    sub_i = 0
    for i in range(len(messages)):
        if (message_filter[i]):
            print(loss[sub_i])
            response[i]["flooding"] = (loss[sub_i] > CTX_FS["THRESHOLD"])
            response[i]["flooding_lat"] = y_[sub_i][0]
            response[i]["flooding_lon"] = y_[sub_i][1]

            message_filter[i] = not(response[i]["flooding"])
            sub_i += 1

    # print("c")

    # check for spoofing
    # filter messages having unknown icao24
    # true_labels = get_true_aircraft_type(messages)
    # for i in range(len(messages)):
    #     if (true_labels[i] == 0):
    #         message_filter[i] = False

    # sub_msg = message_subset(messages, message_filter)
    # _, label_proba = aircraftClassification.predict(sub_msg)
    # spoofing = is_spoofing(true_labels[message_filter], label_proba)
    # sub_i = 0
    # for i in range(len(messages)):
    #     if (message_filter[i]):
    #         response[i]["spoofing"] = spoofing[sub_i]
    #         sub_i += 1

    # print("d")


    # save messages predictions in case of a future request
    for i in range(len(messages)):
        add_message_predictions(hashes[i], response[i])

    # print("e")

    return response



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