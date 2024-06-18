from ._Utils_os_wrapper import os
from ._Utils_numpy import np, ax
from . import _Utils_geographic_maths as geo
HERE = os.path.abspath(os.path.dirname(__file__))
from ._Utils_module import module_to_dict
from .E_Trainer_TrajectorySeparator_Trainer import Trainer as TrajectorySeparator
from .E_Trainer_AircraftClassification_Trainer import Trainer as AircraftClassification
from .E_Trainer_FloodingSolver_Trainer import Trainer as FloodingSolver
from .E_Trainer_ReplaySolver_Trainer import Trainer as ReplaySolver
from . import _Utils_FeatureGetter as FG
from ._Utils_ADSB_Streamer import cast_msg
import time

def getCTX(CTX, default_CTX=None):
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]
    return CTX



from .B_Model_AircraftClassification_CNN2 import Model as CNN2
from . import C_Constants_AircraftClassification_CNN as CNN2_CTX
from . import C_Constants_AircraftClassification_DefaultCTX as CNN2_DefaultCTX
CTX_AC = getCTX(CNN2_CTX, CNN2_DefaultCTX)
aircraftClassification = AircraftClassification(CTX_AC, CNN2)
aircraftClassification.load(HERE+"/AircraftClassification")


from .B_Model_TrajectorySeparator_GeoModel import Model as GEO
from . import C_Constants_TrajectorySeparator_Model as GEO_CTX
from . import C_Constants_TrajectorySeparator_DefaultCTX as GEO_DefaultCTX
from .D_DataLoader_AircraftClassification_Utils import getLabel
CTX_TS = getCTX(GEO_CTX, GEO_DefaultCTX)
trajectorySeparator = TrajectorySeparator(CTX_TS, GEO)
trajectorySeparator.load(HERE+"/TrajectorySeparator")

from .B_Model_FloodingSolver_LSTM import Model as LSTM
from . import C_Constants_FloodingSolver_LSTM as LSTM_CTX
from . import C_Constants_FloodingSolver_DefaultCTX as LSTM_DefaultCTX
CTX_FS = getCTX(LSTM_CTX, LSTM_DefaultCTX)
floodingSolver = FloodingSolver(CTX_FS, LSTM)
floodingSolver.load(HERE+"/FloodingSolver")

from .B_Model_ReplaySolver_HASH import Model as HASH
from . import C_Constants_ReplaySolver_HASH as HASH_CTX
from . import C_Constants_ReplaySolver_DefaultCTX as HASH_DefaultCTX
CTX_RS = getCTX(HASH_CTX, HASH_DefaultCTX)
replaySolver = ReplaySolver(CTX_RS, HASH)
replaySolver.load(HERE+"/ReplaySolver")



def get(lst, bool_arr) -> list:
    return [lst[i] for i in range(len(lst)) if bool_arr[i]]


def hash_message(message: "dict[str, str]") -> "int":
    # message["icao24"] -> hex to int
    return int(
                int(message["icao24"], 16) + \
                int(message["timestamp"]) +  \
                float(message["latitude"]) * 1000 + \
                float(message["longitude"]) * 1000 + \
                float(message["altitude"]) * 1000
            ) % 2147483647


hash_table:"dict[str, list]" = {}
def get_message_predictions(message: "dict[str, str]") -> "dict[str, str]":
    h = hash_message(message)
    data = hash_table.get(h, None)
    if (data == None):
        return None, h
    return data[0], h

def add_message_predictions(hash:int, message: "dict[str, str]") -> None:
    global hash_table
    expiration = time.time() + 30 * 60
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

def predict(messages: "list[dict[str, str]]") -> "list[dict[str, str]]":
    clean_hash_table()

    message_filter = np.ones(len(messages), dtype=bool)
    hashes = np.zeros(len(messages), dtype=np.int32)
    for i in range(len(messages)):
        y_, hashes[i] = get_message_predictions(messages[i])
        if (y_ != None):
            message_filter[i] = False
            messages[i] = y_
        else:
            messages[i] = {col:cast_msg(col,  messages[i].get(col, np.nan)) for col in messages[i]}


    FG.init(CTX_TS)
    sub_msg = message_subset(messages, message_filter)
    sub_icaos = trajectorySeparator.predict(sub_msg)
    sub_i = 0
    for i in range(len(messages)):
        if (message_filter[i]):
            messages[i]["icao24"] = sub_icaos[sub_i]
            sub_i += 1

    FG.init(CTX_RS)
    matches = replaySolver.predict(sub_msg)
    sub_i = 0
    for i in range(len(messages)):
        if (message_filter[i]):
            messages[i]["replay"] = (matches[sub_i] != "none" and matches[sub_i] != "unknown")
            message_filter[i] = not(messages[i]["replay"])
        else:
            messages[i]["replay"] = False

    FG.init(CTX_FS)
    sub_msg = message_subset(messages, message_filter)
    y_, y = floodingSolver.predict(sub_msg)
    sub_i = 0
    for i in range(len(messages)):
        if (message_filter[i]):
            d = geo.distance(y_[sub_i][0], y_[sub_i][1], y[sub_i][0], y[sub_i][1])
            messages[i]["flooding"] = (d > CTX_FS["THRESHOLD"])
            message_filter[i] = not(messages[i]["flooding"])
        else:
            messages[i]["flooding"] = False


    FG.init(CTX_AC)
    sub_msg = message_subset(messages, message_filter)
    _, label_proba = aircraftClassification.predict(sub_msg)
    spoofing = is_spoofing(sub_msg, label_proba)
    sub_i = 0
    for i in range(len(messages)):
        if (message_filter[i]):
            messages[i]["spoofing"] = spoofing[sub_i]
            sub_i += 1
        else:
            messages[i]["spoofing"] = False



    for i in range(len(messages)):
        add_message_predictions(hashes[i], messages[i])

    return messages



# |====================================================================================================================
# | UTILS
# |====================================================================================================================


def get_base_icaos(messages: "list[dict[str, str]]") -> "list[str]":
    icaos = [messages[i]["icao24"] for i in range(len(messages))]
    return [icaos[i].split("_")[0] if ("_" in icaos[i]) else icaos[i] for i in range(len(icaos))]

def get_true_aircraft_type(messages: "list[dict[str, str]]") -> "list[int]":
    icaos = get_base_icaos(messages)
    return [getLabel(CTX_AC, icaos[i]) for i in range(len(icaos))]


def get_pred_aircraft_type(proba: "np.ndarray") -> "list[int]":
    argmax = np.argmax(proba, axis=1)
    confidence = np.nan_to_num([proba[i][argmax[i]] for i in range(len(argmax))])
    return [0 if confidence[i] <= 0.5 else CTX_AC["USED_LABELS"][argmax[i]] for i in range(len(argmax))]

def is_spoofing(messages: "list[dict[str, str]]", predictions: "np.ndarray") -> "list[bool]":
    true_labels = get_true_aircraft_type(messages)
    pred_labels = get_pred_aircraft_type(predictions)

    return [pred_labels[i] != 0
        and pred_labels[i] != true_labels[i]
        for i in range(len(true_labels))]