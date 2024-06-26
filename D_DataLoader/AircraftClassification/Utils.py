import numpy  as np
import pandas as pd
import math
from _Utils.os_wrapper import os
from PIL import Image

import _Utils.Color         as C
import _Utils.FeatureGetter as FG
from   _Utils.Color import prntC
from _Utils.numpy import np, ax

import D_DataLoader.Utils   as U

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

# |====================================================================================================================
# | LABEL MANAGEMENT
# |====================================================================================================================

__icao_db__ = None
def getLabel(CTX, icao):
    """
    Give the label of an aircraft based on his icao imatriculation
    """
    if ("_" in icao):
        icao = icao.split("_")[0]


    global __icao_db__
    if __icao_db__ is None:
        labels_file = os.path.join("./A_Dataset/AircraftClassification/labels.csv")
        __icao_db__ = pd.read_csv(labels_file, sep=",", header=None, dtype={"icao24":str})
        __icao_db__.columns = ["icao24", "label"]
        __icao_db__ = __icao_db__.fillna("NULL")


        # merge labels asked as similar
        for label in CTX["MERGE_LABELS"]:
            __icao_db__["label"] = __icao_db__["label"].replace(CTX["MERGE_LABELS"][label], label)

        # to dict
        __icao_db__ = __icao_db__.set_index("icao24").to_dict()["label"]


    if (icao in __icao_db__):
        return __icao_db__[icao]

    prntC(C.WARNING, icao, "not found in labels.csv")

    return 0


def resetICAOdb():
    global __icao_db__
    __icao_db__ = None





# |====================================================================================================================
# | OSM MAP TILES GENERATION
# |====================================================================================================================

# |--------------------------------------------------------------------------------------------------------------------
# | UTILS function to compute map projections
# |--------------------------------------------------------------------------------------------------------------------

def deg2num_int(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


# load image as numpy array
path = "A_Dataset/AircraftClassification/map.png"
img = Image.open(path)
MAP =  np.array(img, dtype=np.float64) / 255.0
def genMap(lat:float, lon:float, size:int) -> np.ndarray:
    """Generate an image of the map with the flight at the center"""

    if (lat == 0 and lon == 0):
        return np.zeros((size, size, 3), dtype=np.float64)


    #######################################################
    # Convert lat, lon to px
    # thoses param are constants used to generate the map
    zoom = 13
    min_lat, min_lon, max_lat, max_lon = 43.01581, 0.62561,  44.17449, 2.26344
    # conversion
    xmin, _ = deg2num_int(min_lat, min_lon, zoom)
    _, ymin = deg2num_int(max_lat, max_lon, zoom)
    #######################################################

    x_center, y_center = deg2num(lat, lon, zoom)

    x_center = (x_center-xmin)*255
    y_center = (y_center-ymin)*255


    x_min = int(x_center - (size / 2.0))
    x_max = int(x_center + (size / 2.0))
    y_min = int(y_center - (size / 2.0))
    y_max = int(y_center + (size / 2.0))

    if (x_min <= 0):
        x_max = size
        x_min = 0

    elif (x_max >= MAP.shape[1]):
        x_max = MAP.shape[1] -1
        x_min = MAP.shape[1] - size -1

    if (y_min <= 0):
        y_max = size
        y_min = 0

    elif (y_max >= MAP.shape[0]):
        y_max = MAP.shape[0] -1
        y_min = MAP.shape[0] - size -1

    img = MAP[
        y_min:y_max,
        x_min:x_max, :]

    return img


# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================


def in_bbox(CTX:"dict[str, object]", lat:float, lon:float) -> bool:
    return  lat >= CTX["BOUNDING_BOX"][0][0] \
        and lat <= CTX["BOUNDING_BOX"][1][0] \
        and lon >= CTX["BOUNDING_BOX"][0][1] \
        and lon <= CTX["BOUNDING_BOX"][1][1] \


def check_sample(CTX:"dict[str, object]", x:"np.ndarray", i:int, t:int) -> bool:
    lats = FG.lat(x[i])
    lons = FG.lon(x[i])

    lat = lats[t]
    lon = lons[t]

    if (t < CTX["HISTORY"]//4):
        return False

    if (lat == 0 and lon == 0):
        return False

    if (lats[t-1] == lats[t] and lons[t-1] == lons[t]):
        return False

    if (not in_bbox(CTX, lat, lon)):
        return False

    return True

# |====================================================================================================================
# | BATCH GENERATION
# |====================================================================================================================

def alloc_sample(CTX:dict)\
        -> """tuple[
                  np.float64_2d[ax.time, ax.feature],
                  np.float64_2d[ax.time, ax.feature],
                  np.float64_3d[ax.x, ax.y, ax.rgb],
                  np.float64_1d[ax.feature]]""":

    x_sample = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    x_sample_takeoff, x_sample_map, x_sample_airport = None, None, None
    if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_sample_takeoff = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    if (CTX["ADD_MAP_CONTEXT"]): x_sample_map = np.zeros((CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float64)
    if (CTX["ADD_AIRPORT_CONTEXT"]): x_sample_airport = np.zeros((CTX["AIRPORT_CONTEXT_IN"]))
    return x_sample, x_sample_takeoff, x_sample_map, x_sample_airport

def alloc_batch(CTX:dict, size:int)\
        -> """tuple[
                  np.float64_3d[ax.sample, ax.time, ax.feature],
                  np.float64_2d[ax.sample, ax.feature],
                  np.float64_3d[ax.sample, ax.time, ax.feature],
                  np.float64_4d[ax.sample, ax.x, ax.y, ax.rgb],
                  np.float64_2d[ax.sample, ax.feature]]""":

    x_batch = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    y_batch = np.zeros((size, CTX["LABELS_OUT"]))
    x_batch_takeoff, x_batch_map, x_batch_airport = None, None, None
    if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    if (CTX["ADD_MAP_CONTEXT"]): x_batch_map = np.zeros((size, CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float64)
    if (CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport = np.zeros((size, CTX["AIRPORT_CONTEXT_IN"]))
    return x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport


def gen_random_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]",
                                y:np.float64_2d[ax.sample, ax.label],
                                PAD:np.float64_1d[ax.feature], filenames:"list[str]"=[]) -> """tuple[
        np.float64_2d[ax.time, ax.feature],
        np.float64_2d[ax.label],
        np.float64_2d[ax.time, ax.feature] | None,
        np.float64_3d[ax.x, ax.y, ax.rgb] | None,
        np.float64_1d[ax.feature] | None,
        str]""":

    i, t = pick_random_loc(CTX, x, y, filenames)
    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, _ = gen_sample(CTX, x, PAD, i, t, valid=True)
    return x_batch, y[i], x_batch_takeoff, x_batch_map, x_batch_airport, filenames[i]


def pick_random_loc(CTX:"dict[str, object]",
                    x:"list[np.float64_2d[ax.time, ax.feature]]",
                    y:np.float64_2d[ax.sample, ax.label],
                    filenames:"list[str]"=[]) -> "tuple[int, int]":

    ON_TAKE_OFF = 5.0/100.0#%
    # pick a label
    label = np.random.randint(0, CTX["LABELS_OUT"])

    # pick a flight
    i = -1
    while i == -1 or y[i, label] != 1:
        i = np.random.randint(0, len(x))

    # pick a timestamp
    t, tries = None, 0
    while t == None or not(check_sample(CTX, x, i, t)):

        if (np.random.uniform(0, 1) < ON_TAKE_OFF):
            t = np.random.randint(CTX["HISTORY"]//4, CTX["HISTORY"]-1)
        else: t = np.random.randint(CTX["HISTORY"]-1, len(x[i]))

        tries += 1
        if (tries > 1000):
            prntC(C.WARNING, "Failed to a clean window aircraft in", filenames[i])
            return pick_random_loc(CTX, x, y, filenames)

    return i, t


def gen_sample(CTX:"dict[str, object]",
                x:"list[np.float64_2d[ax.time, ax.feature]]",
                PAD:np.float64_1d[ax.feature],
                i:int, t:int,
                valid:bool=None) -> """tuple[
          np.float64_2d[ax.time, ax.feature],
          np.float64_2d[ax.time, ax.feature] | None,
          np.float64_3d[ax.x, ax.y, ax.rgb] | None,
          np.float64_1d[ax.feature] | None,
          bool]""":

    if (valid is None): valid = check_sample(CTX, x, i, t)
    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport = alloc_sample(CTX)
    if (not valid):
        return x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, valid


    # Trajectory
    start, end, length, pad_lenght, shift = U.window_slice(CTX, t)
    x_batch[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
    x_batch[:pad_lenght] = PAD

    #TODO remove this check
    last_message = U.get_aircraft_last_message(CTX, x_batch)
    lat, lon = FG.lat(last_message), FG.lon(last_message)
    if (lat == FG.lat(PAD) or lon == FG.lon(PAD)):
        prntC(C.ERROR, "ERROR: lat or lon is 0")
        prntC(list(range(start, end, CTX["DILATION_RATE"])))
        prntC(FG.lat(x[i][start:end]))
        prntC(i, t, start, end, length, pad_lenght, shift)
        prntC(C.ERROR, "ERROR: lat or lon is 0")


    # Take-Off
    if CTX["ADD_TAKE_OFF_CONTEXT"]:
        start, end, length, pad_lenght, shift = U.window_slice(CTX, length-1)

        # if onground => valid takeoff
        if(FG.baroAlt(x[i][0]) > 2000 or FG.geoAlt(x[i][0]) > 2000):
            x_batch_takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), PAD)
        else:
            x_batch_takeoff[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
            x_batch_takeoff[:pad_lenght] = PAD

    # Map
    if CTX["ADD_MAP_CONTEXT"]:
        x_batch_map = genMap(lat, lon, CTX["IMG_SIZE"])

    # Airport Distance
    if (CTX["ADD_AIRPORT_CONTEXT"]):
        dists = U.toulouse_airport_distance(lat, lon)
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            # reverse the trajectory to get the first position (not the last as default)
            to_lat, to_lon = U.get_aircraft_position(CTX, x_batch_takeoff[::-1])
            airport = U.toulouse_airport_distance(to_lat, to_lon)
            dists = np.concatenate([dists, airport])
        x_batch_airport = dists

    x_batch = U.batch_preprocess(CTX, x_batch, PAD,
                                 CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
    if CTX["ADD_TAKE_OFF_CONTEXT"]:
        x_batch_takeoff = U.batch_preprocess(CTX, x_batch_takeoff, PAD, relative_position=False)

    return x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, valid

