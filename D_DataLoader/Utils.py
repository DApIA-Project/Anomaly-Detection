import pandas as pd
import os
from typing import overload

import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.DataFrame import DataFrame
import _Utils.FeatureGetter as FG
from _Utils.Limits import INT_MAX
from _Utils.numpy import np, ax

from D_DataLoader.Airports import TOULOUSE

# |====================================================================================================================
# | OVERLOADS
# |====================================================================================================================

@overload
def x_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":...
@overload
def x_rotation(x:np.float64_1d, y:np.float64_1d, z:np.float64_1d, a:float)\
    -> "tuple[np.float64_1d, np.float64_1d, np.float64_1d]":...
@overload
def y_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":...
@overload
def y_rotation(x:np.float64_1d, y:np.float64_1d, z:np.float64_1d, a:float)\
    -> "tuple[np.float64_1d, np.float64_1d, np.float64_1d]":...
@overload
def z_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":...
@overload
def z_rotation(x:np.float64_1d, y:np.float64_1d, z:np.float64_1d, a:float)\
    -> "tuple[np.float64_1d, np.float64_1d, np.float64_1d]":...
@overload
def spherical_to_cartesian(lat:float, lon:float) -> "tuple[float, float, float]":...
@overload
def spherical_to_cartesian(lat:np.float64_1d, lon:np.float64_1d)\
    -> "tuple[np.float64_1d, np.float64_1d, np.float64_1d]":...
@overload
def cartesian_to_spherical(x:float, y:float, z:float) -> "tuple[float, float]":...
@overload
def cartesian_to_spherical(x:np.float64_1d, y:np.float64_1d, z:np.float64_1d)\
    -> "tuple[np.float64_1d, np.float64_1d]":...


# |====================================================================================================================
# | UTILS
# |====================================================================================================================


def mini(*args):
    m = min(args[0])
    for a in args[1:]:
        m = min(m, min(a))
    return m
def maxi(*args):
    m = max(args[0])
    for a in args[1:]:
        m = max(m, max(a))
    return m

# |====================================================================================================================
# | LOADING FLIGHTS FROM DISK UTILS
# |====================================================================================================================

def list_flights(path:str, limit:int=INT_MAX) -> "list[str]":
    filenames = os.listdir(path)
    filenames = [os.path.join(path, f) for f in filenames if f.endswith(".csv")]
    filenames.sort()
    return filenames[:limit]


def read_trajectory(path:str, file:str=None) -> pd.DataFrame:
    """
    Read a trajectory from a csv or other file
    """
    if (file != None):
        path = os.path.join(path, file)
    return pd.read_csv(path, sep=",",dtype={"callsign":str, "icao24":str})


# |====================================================================================================================
# | FEW MATH UTILS FOR SPHERICAL CALCULATIONS
# |====================================================================================================================

def x_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x, y * np.cos(-a) - z * np.sin(-a), y * np.sin(-a) + z * np.cos(-a)

def y_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x * np.cos(-a) + z * np.sin(-a), y, -x * np.sin(-a) + z * np.cos(-a)

def z_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z

def spherical_to_cartesian(lat:float, lon:float) -> "tuple[float, float, float]":
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))
    return x, y, z

def cartesian_to_spherical(x:float, y:float, z:float) -> "tuple[float, float]":
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def angle_diff(a:float, b:float) -> float:
    a = a % 360
    b = b % 360

    # compute relative angle
    diff = b - a

    if (diff > 180):
        diff -= 360
    elif (diff < -180):
        diff += 360
    return diff


# |====================================================================================================================
# | TRAJECTORY PREPROCESSING
# |====================================================================================================================

# |--------------------------------------------------------------------------------------------------------------------
# | WINDOW SLICING
# |--------------------------------------------------------------------------------------------------------------------

def compute_shift(start:int, end:int, dilatation:int) -> int:
    """
    compute needed shift to have the last timesteps at the end of the array
    """

    d = end - start
    shift = (d - (d // dilatation) * dilatation - 1) % dilatation
    return shift

def window_slice(CTX:dict, t:int) -> "tuple[int, int, int, int, int]":
    """
    Compute the bounds of the window that should end at t (included)
    """

    start = max(0, t+1-CTX["HISTORY"])
    end = t+1
    length = end - start
    pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
    shift = compute_shift(start, end, CTX["DILATION_RATE"])

    return start, end, length, pad_lenght, shift


# |--------------------------------------------------------------------------------------------------------------------
# | GET LAST MESSAGE FROM TRAJECTORY WITH NANs
# |--------------------------------------------------------------------------------------------------------------------

def get_aircraft_last_message(CTX:dict, flight:np.float64_2d[ax.time, ax.feature]) -> np.float64_1d:
    # get the aircraft last non zero latitudes and longitudes
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    i = len(lat)-1
    while (i >= 0 and (lat[i] == 0 and lon[i] == 0)):
        i -= 1
    if (i == -1):
        return None
    return flight[i]

def get_aircraft_position(CTX:dict, flight:np.float64_2d[ax.time, ax.feature]) -> "tuple[float, float]":
    # get the aircraft last non zero latitudes and longitudes
    pos = get_aircraft_last_message(CTX, flight)
    return FG.lat(pos), FG.lon(pos)


# |--------------------------------------------------------------------------------------------------------------------
# | Convert a CSV dataframe into a numerical array with the right features
# |--------------------------------------------------------------------------------------------------------------------

def df_to_feature_array(CTX:dict, df:DataFrame, check_length:bool=True) -> np.float64_2d[ax.time, ax.feature]:
    """
    Convert a complete ADS-B trajectory dataframe into a numpy array
    with the right features and preprocessing
    """
    if isinstance(df, pd.DataFrame):
        df = DataFrame(df)
    df = pad(df, CTX)

    # if no padding check there is no nan in latitude
    if (CTX["INPUT_PADDING"] == "valid"):
        if (np.isnan(df["latitude"]).any()):
            prntC(C.WARNING, "[df_to_feature_array]:", "NaN in latitude")
            return []
        if (np.isnan(df["longitude"]).any()):
            prntC(C.WARNING, "[df_to_feature_array]:", "NaN in longitude")
            return []

    # add sec (60), min (60), hour (24) and day_of_week (7) features
    timestamp = df["timestamp"]
    df.add_column("day", (timestamp//86400 + 4) % 7)
    df.add_column("hour", (timestamp//3600 + 1) % 24)
    df.add_column("min", (timestamp//60) % 60)
    df.add_column("sec", timestamp % 60)


    # cap altitude to min = 0
    df.setColumValue("altitude", slice(0, len(df)), np.clip(df["altitude"], 0, None))
    df.setColumValue("geoaltitude", slice(0, len(df)), np.clip(df["geoaltitude"], 0, None))

    # add relative track
    track = df["track"]
    relative_track = track.copy()
    for i in range(1, len(relative_track)):
        relative_track[i] = angle_diff(track[i-1], track[i])
    relative_track[0] = 0
    df.add_column("relative_track", relative_track)
    df.setColumValue("timestamp", slice(0, len(df)), df["timestamp"]) # 01/01/2015


    if ("toulouse_0" in CTX["USED_FEATURES"]):
        dists = toulouse_airportDistance(df["latitude"], df["longitude"])

        for i in range(len(TOULOUSE)):
            df.add_column("toulouse_"+str(i), dists[:, i])



    # remove too short flights
    if (check_length and len(df) < CTX["HISTORY"]):
        prntC(C.WARNING, "[df_to_feature_array]: flight too short")
        return []

    # Cast booleans into numeric
    for col in df.columns:
        if (df[col].dtype == bool):
            df[col] = df[col].astype(int)


    # Remove useless columns
    df = df.getColumns(CTX["USED_FEATURES"])


    array = df.astype(np.float64)

    if (len(array) == 0): return None
    return array

TOULOUSE_LATS = np.array([TOULOUSE[i]['lat'] for i in range(len(TOULOUSE))], dtype=np.float64)
TOULOUSE_LONS = np.array([TOULOUSE[i]['long'] for i in range(len(TOULOUSE))], dtype=np.float64)

def toulouse_airportDistance(lats:"list[float]", lons:"list[float]"):
    """
    Compute the distance to the nearest airport
    """
    dtype_number = False
    if (isinstance(lats, int) or isinstance(lats, float)):
        lats = [lats]
        lons = [lons]
        dtype_number = True

    dists = np.zeros((len(lats), len(TOULOUSE)), dtype=np.float64)
    for i in range(len(lats)):
        dists[i] = latlondistance(lats[i], lons[i], TOULOUSE_LATS, TOULOUSE_LONS)


    # cap distance to 50km max
    dists = dists / 1000
    dists = np.clip(dists, 0, 50)
    for i in range(len(dists)):
        for j in range(len(dists[i])):
            if (lats[i] == 0 and lons[i] == 0):
                dists[i][j] = 0

    if (dtype_number):
        return dists[0]
    return dists

def pad(df:DataFrame, CTX):
    """
    Pad a dataframe with the right padding method
    """
    df.add_column("pad", np.zeros(len(df), dtype=np.float64))
    if (CTX["INPUT_PADDING"] == "valid"): return df

    start = df["timestamp"][0]
    total_length = df["timestamp"][-1] - df["timestamp"][0] + 1

    pad_df = np.full((int(total_length), len(df.columns)), np.nan, dtype=np.float64)
    pad_df[:, -1] = np.ones(int(total_length), dtype=np.float64)
    for i in range(len(df)):
        t = df["timestamp"][i]
        pad_df[int(t - start)] = df[i]
    pad_df[:, 0] = np.arange(start, df["timestamp"][-1]+1)

    if (CTX["INPUT_PADDING"] == "last"):
        # replace nan with last value
        for l in range(1, len(pad_df)):
            for c in range(len(pad_df[l])):
                if (np.isnan(pad_df[l][c])):
                    pad_df[l][c] = pad_df[l-1][c]

    df.from_numpy(pad_df)
    return df


def analysis(CTX:dict, dataframe:"list[np.float64_2d[ax.time, ax.feature]]") -> """tuple[
        np.float64_1d[ax.feature],
        np.float64_1d[ax.feature]]""":

    """ dataframe : (sample, timestep, feature)"""
    mins = np.full(CTX["FEATURES_IN"], np.nan)
    maxs = np.full(CTX["FEATURES_IN"], np.nan)

    for i in range(len(dataframe)):
        mins = np.nanmin([mins, np.nanmin(dataframe[i], axis=0)], axis=0)
        maxs = np.nanmax([maxs, np.nanmax(dataframe[i], axis=0)], axis=0)

    return mins, maxs

def genPadValues(CTX:dict, flights:"list[np.float64_2d[ax.time, ax.feature]]") -> np.float64_1d:
    minValues = analysis(CTX, flights)[0]
    padValues = minValues

    for f in range(len(CTX["USED_FEATURES"])):
        feature = CTX["USED_FEATURES"][f]

        if (feature == "latitude"
                or feature == "longitude"):

            padValues[f] = 0

        elif (feature == "altitude"
                or feature == "geoaltitude"
                or feature == "vertical_rate"
                or feature == "groundspeed"
                or feature == "track"
                or feature == "relative_track"
                or feature == "timestamp"):

            padValues[f] = 0

        elif (feature.startswith("toulouse")):
            padValues[f] = 0

        else: # default
            padValues[f] = 0
    return padValues

def splitDataset(data, ratio):
    """
    Split data into train, test and validation set
    """
    train = []
    test = []
    for i in range(len(data)):
        split_index = int(len(data[i]) * (1 - ratio))
        train.append(data[i][:split_index])
        test.append(data[i][split_index:])
    return train, test

# |====================================================================================================================
# | TRAJECTORY PRE PROCESS : SPHERICAL NORMALIZATION
# |====================================================================================================================

def normalize_trajectory(CTX:"dict[str, object]",
                         lat:np.float64_1d[ax.time], lon:np.float64_1d[ax.time], track:np.float64_1d[ax.time],
                         Olat:float, Olon:float, Otrack:float,
                         relative_position:bool, relative_track:bool, random_track:bool)\
        -> "tuple[np.float64_1d[ax.time], np.float64_1d[ax.time], np.float64_1d[ax.time]]":

    LAT, LON, ROT = 0, 0, 0
    if relative_position:
        LAT = -Olat
        LON = -Olon
    else:
        LAT = -CTX["BOX_CENTER"][0]
        LON = -CTX["BOX_CENTER"][1]
    if relative_track:
        ROT = -Otrack
    if random_track:
        ROT = np.random.randint(0, 360)

    x, y, z = spherical_to_cartesian(lat, lon)
    # Normalize longitude with Z rotation
    x, y, z = z_rotation(x, y, z, np.radians(LON))
    # Normalize latitude with Y rotation
    x, y, z = y_rotation(x, y, z, np.radians(LAT))
    # Rotate the fragment with the random angle along X axis
    if (ROT != 0):
        x, y, z = x_rotation(x, y, z, np.radians(ROT))

    lat, lon = cartesian_to_spherical(x, y, z)
    track = np.remainder(track + ROT, 360)

    return lat, lon, track


def denormalize_trajectory(CTX:dict, lat:"np.float64_1d[ax.time]", lon:"np.float64_1d[ax.time]",
                              Olat:float, Olon:float, Otrack:float,
                              relative_position:bool=None, relative_track:bool=None)\
        -> "tuple[np.float64_1d[ax.time], np.float64_1d[ax.time]]":

    if (relative_position is None):
        relative_position = CTX["RELATIVE_POSITION"]
    if (relative_track is None):
        relative_track = CTX["RELATIVE_TRACK"] or CTX["RANDOM_TRACK"]

    ROT = 0
    if relative_position:
        LAT = -Olat
        LON = -Olon
    else:
        LAT = -CTX["BOX_CENTER"][0]
        LON = -CTX["BOX_CENTER"][1]
    if relative_track:
        ROT = -Otrack

    x, y, z = spherical_to_cartesian(lat, lon)
    # UN- the fragment with the random angle along X axis
    x, y, z = x_rotation(x, y, z, np.radians(-ROT))
    # UN-Normalize latitude with Y rotation
    x, y, z = y_rotation(x, y, z, np.radians(-LAT))
    # UN-Normalize longitude with Z rotation
    x, y, z = z_rotation(x, y, z, np.radians(-LON))
    lat, lon = cartesian_to_spherical(x, y, z)

    return lat, lon


def batch_preprocess(CTX:dict, flight:"np.float64_2d[ax.time, ax.feature]",
                          PAD:"np.float64_1d[ax.feature]",
                          relative_position:bool=None, relative_track:bool=None, random_track:bool=None,
                          post_flight:"np.float64_2d[ax.time, ax.feature]"=None)\
        -> """np.float64_2d[ax.time, ax.feature]
            | tuple[np.float64_2d[ax.time, ax.feature], np.float64_2d[ax.time, ax.feature]]""":

    if (relative_position is None):
        relative_position = CTX["RELATIVE_POSITION"]
    if (relative_track is None):
        relative_track = CTX["RELATIVE_TRACK"]
    if (random_track is None):
        random_track = CTX["RANDOM_TRACK"]

    # calculate normalized trajectory
    pos = get_aircraft_last_message(CTX, flight)
    x = flight
    if (post_flight is not None):
        x = np.concatenate([flight, post_flight], axis=0)

    nan_value = np.logical_and(FG.lat(x) == FG.lat(PAD), FG.lon(x) == FG.lon(PAD))
    lat, lon, track = normalize_trajectory(CTX,
                                             FG.lat(x), FG.lon(x), FG.track(x),
                                             FG.lat(pos), FG.lon(pos), FG.track(pos),
                                             relative_position, relative_track, random_track)

    # only apply normalization on non zero lat/lon
    x[~nan_value, FG.lat()] = lat[~nan_value]
    x[~nan_value, FG.lon()] = lon[~nan_value]
    x[~nan_value, FG.track()] = track[~nan_value]

    # fill nan lat/lon with the first non zero lat lon
    first_non_zero_ts = np.argmax(~nan_value)
    start_lat, start_lon = lat[first_non_zero_ts], lon[first_non_zero_ts]
    x[nan_value, FG.lat()] = start_lat
    x[nan_value, FG.lon()] = start_lon

    # if there is timestamp in the features, we normalize it
    if (FG.has("timestamp")):
        x[:, FG.timestamp()] = FG.timestamp(pos) - FG.timestamp(x)

    if (post_flight is not None):
        return x[:len(flight)], x[len(flight):]
    return x

