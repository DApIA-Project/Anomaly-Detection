import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from . import CTX
from . import DefaultCTX as default_CTX

import pandas as pd
import numpy as np

from .module import module_to_dict
from .save import load

from .model import Model
from .DataLoader import DataLoader
from .Scaler3D import StandardScaler3D, fillNaN3D
from .SparceLabelBinarizer import SparceLabelBinarizer
from . import Utils_1 as SU
from . import Utils as U
from .Trainer import reshape

import time

from .DataFrame import DataFrame

# ------------------------------- BACKEND -------------------------------


__HERE__ = os.path.abspath(os.path.dirname(__file__))

# Convert CTX to dict
CTX = module_to_dict(CTX)
if (default_CTX != None):
    default_CTX = module_to_dict(default_CTX)
    for param in default_CTX:
        if (param not in CTX):
            CTX[param] = default_CTX[param]

CTX["EPOCHS"] = 0


model = Model(CTX)
xScaler = StandardScaler3D()
xScalerTakeoff = StandardScaler3D()

dataloader = DataLoader(CTX)

w = load(__HERE__ + "/w")
xs = load(__HERE__ + "/xs")
xts = load(__HERE__ + "/xts")


model.setVariables(w)
xScaler.setVariables(xs)
xScalerTakeoff.setVariables(xts)

def cast_msg(col, msg):
    if (msg == np.nan or msg == None or msg == ""):
        return np.nan
    elif (col == "icao24" or col == "callsign"):
        return msg
    elif (col == "onground" or col == "alert" or col == "spi"):
        return float(msg == "True")
    elif (col == "timestamp"):
        return int(msg)
    else:
        return float(msg)



class sortedMap:
    def __init__(self):
        self._keys = []
        self._values = []
    
    def add(self, key, value):
        left = 0
        right = len(self._keys)
        mid = (left + right)//2

        while (left < right):
            if (self._keys[mid] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2
        self._keys.insert(mid, key)
        self._values.insert(mid, value)


    def get(self, key):
        left = 0
        right = len(self._keys)
        mid = (left + right)//2
        while (left < right):
            if (self.key[mid] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2
        if (self._keys[mid] == key):
            return self._values[mid]
        return None
        
        
    def __str__(self):
        str_rp = "{"
        for i in range(len(self._keys)):
            str_rp += f"{self._keys[i]}:{self._values[i]}"
            if (i < len(self._keys)-1):
                str_rp += ", "
        str_rp += "}"
        return str_rp

    
    def clear(self):
        del self.map
        self.map = []  

    def keys(self):
        return self._keys
    def values(self):
        return self._values

__FEATURES__ = ["timestamp", "latitude", "longitude","groundspeed","track","vertical_rate","onground","alert","spi","squawk","altitude","geoaltitude"]
TIMESTAMP_I = __FEATURES__.index("timestamp")

class TrajectoryBuffer:
    def __init__(self):
        # store message per icao, sorted by timestamp
        self.trajectories:dict[str, DataFrame] = {}
        self.last_update = {}
        self.ERASE_TIME = 60*15 # 15 minutes

        self.chache_predictions = {}

    def add_message(self, message:dict):
        icao24 = message["icao24"]
        if (icao24 not in self.trajectories):
            df = DataFrame(len(__FEATURES__))
            df.setColums(__FEATURES__)

            self.trajectories[icao24] = df

        if (icao24 not in self.chache_predictions):
            self.chache_predictions[icao24] = {}

        
        for k in __FEATURES__:
            if (k not in message):
                print(f"Warning : unknown column {k} in message of aircraft {icao24}")

        msg = [cast_msg(col, message.get(col, np.nan)) for col in __FEATURES__]



        if (len(self.trajectories[icao24]) > 0):
            last_timestamp = self.trajectories[icao24]["timestamp"][-1]
            diff = int(msg[TIMESTAMP_I] - last_timestamp)

            if (diff > 1):
                if (CTX["INPUT_PADDING"] == "nan"):
                    pad = np.full((diff-1, len(__FEATURES__)), np.nan)
                    pad[:, TIMESTAMP_I] = np.arange(last_timestamp+1, msg[TIMESTAMP_I])
                    self.trajectories[icao24].__append__(pad)

                elif (CTX["INPUT_PADDING"] == "last"):
                    pad = np.full((diff-1, len(__FEATURES__)), self.trajectories[icao24][-1])
                    pad[:, TIMESTAMP_I] = np.arange(last_timestamp+1, msg[TIMESTAMP_I])
                    self.trajectories[icao24].__append__(pad)
        
        if not(self.trajectories[icao24].add(msg)):
            print(f"Warning : duplicate message for aircraft {icao24} at timestamp {msg[TIMESTAMP_I]}")
        self.last_update[icao24] = time.time()
        self.update()

    def getDataForPredict(self, icao24s:"list[dict[str, str]]"):


        LON_I = CTX["FEATURE_MAP"]["longitude"]
        LAT_I = CTX["FEATURE_MAP"]["latitude"]
        ALT_I = CTX["FEATURE_MAP"]["altitude"]
        GEO_I = CTX["FEATURE_MAP"]["geoaltitude"]

        x_batches = np.zeros((len(icao24s), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = np.zeros((len(icao24s), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = np.zeros((len(icao24s), CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float32)

        for i in range(len(icao24s)):
            icao24 = icao24s[i]["icao24"]
            df = self.trajectories[icao24]
            end_timestamp = int(icao24s[i]["timestamp"])

            df = df.subset(end_timestamp)
            if (df == None):
                raise Exception(f"ANOMALY : aircraft {icao24} has no data for timestamp {end_timestamp}")
            
            array = U.dfToFeatures(df, CTX, __LIB__=True)
            array = fillNaN3D([array], dataloader.FEATURES_PAD_VALUES)[0]

            # preprocess
            t = len(df)-1
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

            x_batch = array[start+shift:end:CTX["DILATION_RATE"]]

            x_batches[i, :pad_lenght] = dataloader.FEATURES_PAD_VALUES
            x_batches[i, pad_lenght:] = x_batch

            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                # compute the bounds of the fragment
                start = 0
                end = length
                shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

                # build the batch
                if(array[0,ALT_I] > 2000 or array[0,GEO_I] > 2000):
                    takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), dataloader.FEATURES_PAD_VALUES)
                else:
                    takeoff = array[start+shift:end:CTX["DILATION_RATE"]]
                
                # add padding and add to the batch
                x_batches_takeoff[i, :pad_lenght] = dataloader.FEATURES_PAD_VALUES
                x_batches_takeoff[i, pad_lenght:] = takeoff
                
            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = SU.getAircraftPosition(CTX, x_batches[i])
                x_batches_map[i] = SU.genMap(lat, lon, CTX["IMG_SIZE"])
            
            x_batches[i, pad_lenght:] = SU.batchPreProcess(CTX, x_batches[i, pad_lenght:], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[i, pad_lenght:] = SU.batchPreProcess(CTX, x_batches_takeoff[i, pad_lenght:])

        x_batches = xScaler.transform(x_batches)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = xScalerTakeoff.transform(x_batches_takeoff)

        x_inputs = []
        for i in range(len(x_batches)):
            x_input = [x_batches[i]]
            if CTX["ADD_TAKE_OFF_CONTEXT"]: x_input.append(x_batches_takeoff[i])
            if CTX["ADD_MAP_CONTEXT"]: x_input.append(x_batches_map[i])
            x_inputs.append(x_input)

        self.update()
        return x_inputs



    def update(self):
        # remove aircrafts that have not been updated for a while (15 minutes)
        t = time.time()
        for icao in list(self.trajectories.keys()):
            if (t - self.last_update[icao] > self.ERASE_TIME):
                del self.trajectories[icao]
                del self.last_update[icao]
                del self.chache_predictions[icao]

    def clear(self):
        del self.trajectories
        del self.last_update
        del self.chache_predictions
        self.trajectories:dict[str, pd.DataFrame] = {}
        self.last_update = {}
        self.chache_predictions = {}


buffer = TrajectoryBuffer()

    















# ------------------------------- LIBRARY -------------------------------

class CONTEXT:
    LABEL_NAMES = CTX["LABEL_NAMES"]

LABEL_NAMES = CTX["LABEL_NAMES"]



def predictAircraftType(messages: "list[dict[str, str]]"):
    """
    Make predicitons of aircraft type based on ADS-B/FLARM features.

    Each feature is an array of shape [?, HISTORY].

    return : probability array of shape [?, FEATURES_OUT]
    """

    # sort message by timestamp
    messages.sort(key=lambda x: int(x["timestamp"]))
    exist = [False for _ in range(len(messages))]
    for i in range(len(messages)):
        if (messages[i]["icao24"] in buffer.chache_predictions):
            if (messages[i]["timestamp"] in buffer.chache_predictions[messages[i]["icao24"]]):
                exist[i] = True
    
    messages_exists = [messages[i] for i in range(len(messages)) if exist[i]]
    messages = [messages[i] for i in range(len(messages)) if not exist[i]]



    # save messages in buffer
    for message in messages:
        buffer.add_message(message)

    if (len(messages) > 0):
        # get the data for prediction
        x_batch = buffer.getDataForPredict(messages)

        # do the prediction
        proba = model.predict(reshape(x_batch))

        # cache the predictions
        for i in range(len(messages)):
            buffer.chache_predictions[messages[i]["icao24"]][messages[i]["timestamp"]] = proba[i].numpy()

    
    # format the result as a dict of icao24 -> proba array
    res = {}
    for i in range(len(messages)):
        if (messages[i]["icao24"] not in res):
            res[messages[i]["icao24"]] = {}
        res[messages[i]["icao24"]][messages[i]["timestamp"]] = buffer.chache_predictions[messages[i]["icao24"]][messages[i]["timestamp"]]
    
    for i in range(len(messages_exists)):
        if (messages_exists[i]["icao24"] not in res):
            res[messages_exists[i]["icao24"]] = {}
        res[messages_exists[i]["icao24"]][messages_exists[i]["timestamp"]] = buffer.chache_predictions[messages_exists[i]["icao24"]][messages_exists[i]["timestamp"]]

    return res

def probabilityToLabel(proba):
    """
    Take an array of probabilities and give the label
    corresponding to the highest class.

    proba : probabilities, array of shape : [?, FEATURES_OUT]

    return : label id, array of int, shape : [?]
    """
    i = np.argmax(proba, axis=1)
    l = [CTX["USED_LABELS"][j] for j in i]
    return l

def labelToName(label) -> "list[str]" :
    """
    Give the label name (easily readable for humman) 
    according to the label id.

    label : label id, array of int, shape : [?]

    return : classes names, array of string, shape : [?]
    """
    # if it's iterable
    if (isinstance(label, (list, tuple, np.ndarray))):
        return [CTX["LABEL_NAMES"][l] for l in label]
    else:
        return CTX["LABEL_NAMES"][label]



labels_file = __HERE__+"/labels.csv"
__icao_db__ = pd.read_csv(labels_file, sep=",", header=None, dtype={"icao24":str})
__icao_db__.columns = ["icao24", "label"]
__icao_db__ = __icao_db__.fillna("NULL")
for label in CTX["MERGE_LABELS"]:
    __icao_db__["label"] = __icao_db__["label"].replace(CTX["MERGE_LABELS"][label], label)
__icao_db__ = __icao_db__.set_index("icao24").to_dict()["label"]

def getTruthLabelFromIcao(icao24):
    """
    Get the label of an aircraft from its icao24.

    icao24 : icao24 of the aircraft, string

    return : label id, int
    """
    if (icao24 in __icao_db__):
        return __icao_db__[icao24]
    else:
        return 0
    
def __clearBuffer__():
    buffer.clear()