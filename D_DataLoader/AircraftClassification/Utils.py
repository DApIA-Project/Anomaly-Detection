import numpy as np
import pandas as pd
import math
import os
from PIL import Image

import _Utils.Color as C
from _Utils.Color import prntC
import _Utils.FeatureGetter as FG

import D_DataLoader.Utils as U

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

###################################################
# LABEL MANAGEMENT
###################################################

__icao_db__ = None
def getLabel(CTX, df):
    """
    Give the label of an aircraft based on his icao imatriculation
    """
    icao = FG.df_icao(df)
    callsign = FG.df_callsign(df)

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

    if (12 in CTX["MERGE_LABELS"] and callsign.startswith("SAMU")):
        return 12

    if (icao in __icao_db__):
        return __icao_db__[icao]

    prntC(C.WARNING, icao, "not found in labels.csv")

    return 0


def resetICAOdb():
    global __icao_db__
    __icao_db__ = None

###################################################
# TRAJECTORY PROCESSING
###################################################

def getAircraftLastMessage(CTX, flight):
    # get the aircraft last non zero latitudes and longitudes
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    i = len(lat)-1
    while (i >= 0 and (lat[i] == 0 and lon[i] == 0)):
        i -= 1
    if (i == -1):
        return None
    return flight[i]

def getAircraftPosition(CTX, flight):
    # get the aircraft last non zero latitudes and longitudes
    pos = getAircraftLastMessage(CTX, flight)
    return FG.lat(pos), FG.lon(pos)

def batchPreProcess(CTX, flight, PAD, relative_position=False, relative_track=False, random_track=False):
    pos = getAircraftLastMessage(CTX, flight)
    nan_value = np.logical_and(FG.lat(flight) == FG.lat(PAD), FG.lon(flight) == FG.lon(PAD))
    lat, lon, track = U.normalize_trajectory(CTX,
                                             FG.lat(flight), FG.lon(flight), FG.track(flight),
                                             FG.lat(pos), FG.lon(pos), FG.track(pos),
                                             relative_position, relative_track, random_track)
    # only apply normalization on non zero lat/lon
    flight[~nan_value, FG.lat()] = lat[~nan_value]
    flight[~nan_value, FG.lon()] = lon[~nan_value]
    flight[~nan_value, FG.track()] = track[~nan_value]

    # fill nan lat/lon with the first non zero lat lon
    first_non_zero_ts = np.argmax(~nan_value)     # (argmax return the first max value)
    start_lat, start_lon = lat[first_non_zero_ts], lon[first_non_zero_ts]
    flight[nan_value, FG.lat()] = start_lat
    flight[nan_value, FG.lon()] = start_lon

    # if there is timestamp in the features, we normalize it
    if (FG.has("timestamp")):
        flight[:, FG.timestamp()] = flight[:, FG.timestamp()] - flight[-1, FG.timestamp()]
        flight[nan_value, FG.timestamp()] = flight[0, FG.timestamp()]

    return flight

###################################################
# OSM MAP TILES GENERATION
###################################################


"""
UTILITARY FUNCTION FOR MAP PROJECTION
"""
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
MAP =  np.array(img, dtype=np.float32) / 255.0
def genMap(lat, lon, size):
    """Generate an image of the map with the flight at the center"""

    if (lat == 0 and lon == 0):
        return np.zeros((size, size, 3), dtype=np.float32)


    #######################################################
    # Convert lat, lon to px
    # thoses param are constants used to generate the map
    zoom = 13
    min_lat, min_lon, max_lat, max_lon = 43.01581, 0.62561,  44.17449, 2.26344
    # conversion
    xmin, ymax = deg2num_int(min_lat, min_lon, zoom)
    xmax, ymin = deg2num_int(max_lat, max_lon, zoom)
    #######################################################

    x_center, y_center = deg2num(lat, lon, zoom)

    x_center = (x_center-xmin)*255
    y_center = (y_center-ymin)*255


    x_min = int(x_center - (size / 2.0))
    x_max = int(x_center + (size / 2.0))
    y_min = int(y_center - (size / 2.0))
    y_max = int(y_center + (size / 2.0))

    d_x_min = x_min
    d_x_max = x_max
    d_y_min = y_min
    d_y_max = y_max

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

    if (img.shape[0] != size or img.shape[1] != size):
        prntC(C.ERROR, "map size is not correct")


    return img


###################################################
# CHECKING CLEANESS FOR TRAINING DATA
###################################################

def inBB(lat, lon, CTX):
    return  lat >= CTX["BOUNDING_BOX"][0][0] \
        and lat <= CTX["BOUNDING_BOX"][1][0] \
        and lon >= CTX["BOUNDING_BOX"][0][1] \
        and lon <= CTX["BOUNDING_BOX"][1][1] \


def check_batch(CTX, x, i, t):
    lats = FG.lat(x[i])
    lons = FG.lon(x[i])


    lat = lats[t]
    lon = lons[t]


    if (lat == 0 and lon == 0):
        return False

    if (t>0 and lats[t-1] == lats[t] and lons[t-1] == lons[t]):
        return False

    if (not inBB(lat, lon, CTX)):
        return False

    return True

###################################################
# BATCH GENERATION
###################################################
def allocSample(CTX):
    x_batch = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    x_batch_takeoff, x_batch_map, x_batch_airport = None, None, None
    if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    if (CTX["ADD_MAP_CONTEXT"]): x_batch_map = np.zeros((CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float32)
    if (CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport = np.zeros((CTX["AIRPORT_CONTEXT_IN"]))
    return x_batch, x_batch_takeoff, x_batch_map, x_batch_airport

def allocBatch(CTX, size):
    x_batch = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    y_batch = np.zeros((size, CTX["FEATURES_OUT"]))
    x_batch_takeoff, x_batch_map, x_batch_airport = None, None, None
    if (CTX["ADD_TAKE_OFF_CONTEXT"]): x_batch_takeoff = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    if (CTX["ADD_MAP_CONTEXT"]): x_batch_map = np.zeros((size, CTX["IMG_SIZE"], CTX["IMG_SIZE"],3), dtype=np.float32)
    if (CTX["ADD_AIRPORT_CONTEXT"]): x_batch_airport = np.zeros((size, CTX["AIRPORT_CONTEXT_IN"]))
    return x_batch, y_batch, x_batch_takeoff, x_batch_map, x_batch_airport


def genRandomBatch(CTX, x, y, PAD, size, filenames=[]):
    i, ts = getRandomLoc(CTX, x, y, size, filenames)
    batches = ([], [], [], [], [])
    for t in ts:
        x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, valid = genSample(CTX, x, PAD, i, t, valid=True)
        batches[0].append(x_batch)
        batches[1].append(y[i])
        batches[2].append(x_batch_takeoff)
        batches[3].append(x_batch_map)
        batches[4].append(x_batch_airport)
    filenames = [filenames[i]] * size
    return tuple((np.array(b) for b in batches)) + (filenames,)


def getRandomLoc(CTX, x, y, size=1, filenames=[]):
    ON_TAKE_OFF = 5.0/100.0#%
    # pick a label
    label = np.random.randint(0, CTX["FEATURES_OUT"])

    # pick a flight
    i = -1
    while i == -1 or y[i, label] != 1:
        i = np.random.randint(0, len(x))

    # pick a timestamp
    t, tries = None, 0
    while t == None or not(check_batch(CTX, x, i, t)):

        if (np.random.uniform(0, 1) < ON_TAKE_OFF):
            t = np.random.randint(CTX["HISTORY"]//4, CTX["HISTORY"]-1)
        else: t = np.random.randint(CTX["HISTORY"]-1, len(x[i])-(size-1))

        tries += 1
        if (tries > 1000):
            prntC(C.WARNING, "Failed to a clean window aircraft in", filenames[i])
            return getRandomLoc(CTX, x, y, label, size, filenames)

    return i, np.arange(t, t+size)


def genSample(CTX, x, PAD, i, t, valid=None):
    if (valid is None): valid = check_batch(CTX, x, i, t)
    x_batch, x_batch_takeoff, x_batch_map, x_batch_airport = allocSample(CTX)
    if (not valid):
        return x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, valid


    # Trajectory
    start, end, length, pad_lenght, shift = U.windowBounds(CTX, t)
    x_batch[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
    x_batch[:pad_lenght] = PAD
    lat, lon = getAircraftPosition(CTX, x_batch)
    if (lat == FG.lat(PAD) or lon == FG.lon(PAD)):
        prntC(C.ERROR, "ERROR: lat or lon is 0")
        prntC(list(range(start, end, CTX["DILATION_RATE"])))
        prntC(FG.lat(x[i][start:end]))
        prntC(i, t, start, end, length, pad_lenght, shift)
        prntC(C.ERROR, "ERROR: lat or lon is 0")


    # Take-Off
    if CTX["ADD_TAKE_OFF_CONTEXT"]:
        start, end, length, pad_lenght, shift = U.windowBounds(CTX, length-1)

        # if onground => valid takeoff
        if(FG.baroAlt(x[i][0]) > 2000 or FG.geoAlt(x[i][0]) > 2000):
            pass
            # takeoff = np.full((len(x_batch), CTX["FEATURES_IN"]), pad)
        x_batch_takeoff[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
        x_batch_takeoff[:pad_lenght] = PAD

    # Map
    if CTX["ADD_MAP_CONTEXT"]:
        x_batch_map = genMap(lat, lon, CTX["IMG_SIZE"])

    # Airport Distance
    if (CTX["ADD_AIRPORT_CONTEXT"]):
        dists = U.toulouse_airportDistance(lat, lon)[0]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            # reverse the trajectory to get the first position (not the last as default)
            to_lat, to_lon = getAircraftPosition(CTX, x_batch_takeoff[::-1])
            airport = U.toulouse_airportDistance(to_lat, to_lon)
            dists = np.concatenate([dists, airport[0]])
        x_batch_airport = dists

    x_batch = batchPreProcess(CTX, x_batch, PAD, CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
    if CTX["ADD_TAKE_OFF_CONTEXT"]:
        x_batch_takeoff = batchPreProcess(CTX, x_batch_takeoff, PAD, relative_position=False)

    return x_batch, x_batch_takeoff, x_batch_map, x_batch_airport, valid




###################################################
# STATISTICS
###################################################






def debugTraining():
    pass
        # print("toff_nb ratio : ", stat_takeoff_count, "/", len(x_batches))

        # # plot some trajectories
        # c = math.sqrt(16)
        # fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        # for i in range(9):
        #     lat = x_batches[i, :, LAT_I]
        #     lon = x_batches[i, :, LON_I]
        #     x, y = i//3, i%3
        #     if (lon[0] == 0):
        #         print("ERROR lon 0 = 0")
        #     axs[x, y].plot(lon, lat)
        #     axs[x, y].scatter(lon[-1], lat[-1], color="green")
        #     axs[x, y].scatter(lon[0], lat[0], color="red")
        #     for i in range(3):
        #         axs[x, y].scatter(lon[i+1], lat[i+1], color="orange")

        #     for i in range(3):
        #         axs[x, y].scatter(lon[-(i+2)], lat[-(i+2)], color="blue")

        # fig.savefig('_Artifacts/trajectory.png')
        # plt.close(fig)

        # MIN_VALUES, MAX_VALUES = U.analysis(self.x)

        #     print("DEBUG SCALLERS : ")
        #     prntC("feature:","|".join(self.CTX["USED_FEATURES"]), start=C.BRIGHT_BLUE)
        #     print("mean   :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.means)]))
        #     print("std dev:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xScaler.stds)]))
        #     print("mean TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.means)]))
        #     print("std  TO:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.xTakeOffScaler.stds)]))
        #     print("nan pad:","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(self.PAD)]))
        #     print("min    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(MIN_VALUES)]))
        #     print("max    :","|".join([str(round(v, 1)).ljust(len(self.CTX["USED_FEATURES"][i])) for i, v in enumerate(MAX_VALUES)]))

        #     if (CTX["ADD_AIRPORT_CONTEXT"]):
        #         print("DEBUG TO SCALLERS : ")
        #         prntC("feature:","|".join([str(i).ljust(5) for i in range(len(self.xAirportScaler.mins))]), start=C.BRIGHT_BLUE)
        #         print("min    :","|".join([str(round(v, 1)).ljust(5) for i, v in enumerate(self.xAirportScaler.mins)]))
        #         print("max    :","|".join([str(round(v, 1)).ljust(5) for i, v in enumerate(self.xAirportScaler.maxs)]))