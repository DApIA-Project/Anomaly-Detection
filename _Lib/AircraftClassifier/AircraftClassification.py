
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
from . import Utils as U
from .Trainer import reshape

import time

import os

# ------------------------------- BACKEND -------------------------------


__HERE__ = os.path.abspath(os.path.dirname(__file__))

# Convert CTX to dict
CTX = module_to_dict(CTX)
if (default_CTX != None):
    default_CTX = module_to_dict(default_CTX)
    for param in default_CTX:
        if (param not in CTX):
            CTX[param] = default_CTX[param]


model = Model(CTX)
xScaler = StandardScaler3D()
xScalerTakeoff = StandardScaler3D()

dataloader = DataLoader(CTX)

print("Loading weights, please wait...")


w = load(__HERE__ + "/w")
xs = load(__HERE__ + "/xs")
xts = load(__HERE__ + "/xts")


model.setVariables(w)
xScaler.setVariables(xs)
xScalerTakeoff.setVariables(xts)



class TrajectoryBuffer:
    def __init__(self):
        # store message per icao, sorted by timestamp
        self.trajectories:dict[str, pd.DataFrame] = {}
        self.last_update = {}
        self.ERASE_TIME = 60*15 # 15 minutes

    def add_message(self, message:dict):
        icao24 = message["icao24"]
        if (icao24 not in self.trajectories):
            # dataframe with timestamp as index, message attributes as columns, sorted by timestamp
            # icao24 is str
            df = pd.DataFrame(
                columns=["timestamp","icao24", "latitude", "longitude","groundspeed","track","vertical_rate","callsign","onground","alert","spi","squawk","altitude","geoaltitude"], 
                dtype=np.float32)
            df["icao24"] = df["icao24"].astype(str)
            df["callsign"] = df["callsign"].astype(str)
            # df.set_index("timestamp", inplace=True)
            # df.sort_index(inplace=True)
            self.trajectories[icao24] = df

        # only keep the columns of the dataframe
        # warning : if the message does not contain all the columns, it will be ignored
        for k in self.trajectories[icao24].columns:
            if (k not in message):
                print(f"Warning : missing column {k} in message of aircraft {icao24}")

        message = {k: message.get(k, np.nan) for k in self.trajectories[icao24].columns}
        # self.trajectories[icao24] = self.trajectories[icao24].append(message, ignore_index=True)
        # cat
        self.trajectories[icao24] = pd.concat([self.trajectories[icao24], pd.DataFrame(message, index=[0])], ignore_index=True)
        self.last_update[icao24] = time.time()
        
        self.update()

    def getDataForPredict(self, icao24s:"list[str]"):
        dfs = [self.trajectories[icao24] for icao24 in icao24s]

        LON_I = CTX["FEATURE_MAP"]["longitude"]
        LAT_I = CTX["FEATURE_MAP"]["latitude"]
        ALT_I = CTX["FEATURE_MAP"]["altitude"]
        GEO_I = CTX["FEATURE_MAP"]["geoaltitude"]

        x_batches = np.zeros((len(dfs), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batches_takeoff = np.zeros((len(dfs), CTX["INPUT_LEN"], CTX["FEATURES_IN"]))
        if (CTX["ADD_MAP_CONTEXT"]): x_batches_map = np.zeros((len(dfs), CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float32)

        for d in range(len(dfs)):
            # df to array
            df = dfs[d]
            array = U.dfToFeatures(df.copy(), None, CTX, __LIB__=True)
            # print(array)
            # print(array.shape)
            array = fillNaN3D([array], dataloader.FEATURES_PAD_VALUES)[0]

            # preprocess
            t = len(df)-1
            start = max(0, t+1-CTX["HISTORY"])
            end = t+1
            length = end - start
            pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
            shift = U.compute_shift(start, end, CTX["DILATION_RATE"])

            x_batch = array[start+shift:end:CTX["DILATION_RATE"]]

            x_batches[d, :pad_lenght] = dataloader.FEATURES_PAD_VALUES
            x_batches[d, pad_lenght:] = x_batch

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
                x_batches_takeoff[d, :pad_lenght] = dataloader.FEATURES_PAD_VALUES
                x_batches_takeoff[d, pad_lenght:] = takeoff
                
            if CTX["ADD_MAP_CONTEXT"]:
                lat, lon = U.getAircraftPosition(CTX, x_batches[d])
                x_batches_map[d] = U.genMap(lat, lon, CTX["IMG_SIZE"])
            
            x_batches[d, pad_lenght:] = U.batchPreProcess(CTX, x_batches[d, pad_lenght:], CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
            if CTX["ADD_TAKE_OFF_CONTEXT"]:
                x_batches_takeoff[d, pad_lenght:] = U.batchPreProcess(CTX, x_batches_takeoff[d, pad_lenght:])

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


buffer = TrajectoryBuffer()

    


# def __preprocess__(timestamp,latitude,longitude,groundspeed,track,vertical_rate,onground,alert,spi,squawk,altitude,geoaltitude):

#     df = pd.DataFrame(
#         {
#             "timestamp": timestamp,
#             "latitude": latitude,
#             "longitude": longitude,
#             "groundspeed": groundspeed,
#             "track": track,
#             "vertical_rate": vertical_rate,
#             "onground": onground,
#             "alert": alert,
#             "spi": spi,
#             "squawk": squawk,
#             "altitude": altitude,
#             "geoaltitude": geoaltitude,
#         }
#     )


#     array = DataLoader.dfToFeatures(df, None, CTX)
#     array = array[CTX["DILATION_RATE"]-1::CTX["DILATION_RATE"]]
#     return array













# ------------------------------- LIBRARY -------------------------------

class CONTEXT:
    LABEL_NAMES = CTX["LABEL_NAMES"]

LABEL_NAMES = CTX["LABEL_NAMES"]





def predictAircraftType(messages: "list[dict[str, float]]"):
    """
    Make predicitons of aircraft type based on ADS-B/FLARM features.

    Each feature is an array of shape [?, HISTORY].

    return : probability array of shape [?, FEATURES_OUT]
    """

    # save messages in buffer
    for message in messages:
        buffer.add_message(message)

    # get the list of aircrafts to predict
    unique_icao = list(set([m["icao24"] for m in messages]))    

    # get the data for prediction
    x_batch = buffer.getDataForPredict(unique_icao)

    # do the prediction
    proba = model.predict(reshape(x_batch))

    # format the result as a dict of icao24 -> proba array
    res = {}
    for i in range(len(unique_icao)):
        res[unique_icao[i]] = proba[i].numpy()
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