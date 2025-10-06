import pandas as pd
from _Utils.os_wrapper import os
from typing import overload
from numpy_typing import np, ax

import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.DataFrame import DataFrame
import _Utils.Limits as Limits
import _Utils.geographic_maths as GEO
from   _Utils.Scaler3D import *
from   _Utils.FeatureGetter import FeatureGetter

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

def list_flights(path:str, limit:int=Limits.INT_MAX) -> "list[str]":
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
    df = pd.read_csv(path, sep=",",dtype={"callsign":str, "icao24":str})
    if not("tag" in df.columns):
        df["tag"] = "0"
    return df


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
# | SCALERS
# |--------------------------------------------------------------------------------------------------------------------

<<<<<<< HEAD
def getScaler(name:str, dims=2) -> "type":
    if (dims == 2):
        if (name == "standard") : return StandardScaler2D
        if (name == "minmax") : return MinMaxScaler2D
        if (name == "dummy") : return DummyScaler2D
    
    if (dims == 3):
        if (name == "standard") : return StandardScaler3D
        if (name == "minmax") : return MinMaxScaler3D
        if (name == "dummy") : return DummyScaler3D
=======
def getScaler(name:str, dims=2, params = None) -> "type":
    smin, smax = 0, 1
    if (params is not None):
        dims = params.get("dims", dims)
        smin = params.get("min", 0)
        smax = params.get("max", 1)
        
    if (dims == 2):
        if (name == "standard") : return StandardScaler2D()
        if (name == "minmax") : return MinMaxScaler2D(smin, smax)
        if (name == "dummy") : return DummyScaler2D()
    
    if (dims == 3):
        if (name == "standard") : return StandardScaler3D()
        if (name == "minmax") : return MinMaxScaler3D(smin, smax)
        if (name == "dummy") : return DummyScaler3D()
>>>>>>> master
    return None


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

def equals(a, b, d=1e-6) -> bool:
    """
    Check if two values are equal with a tolerance
    """
    return abs(a - b) < d

def get_aircraft_last_message_t(CTX:dict, flight:np.float64_2d[ax.time, ax.feature], last_t = -1) -> np.float64_1d:
    # get the aircraft last non zero latitudes and longitudes
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    if (last_t == -1): last_t = len(lat)-1
    i = last_t
    while (i >= 1 and ((equals(lat[i], 0) and equals(lon[i], 0)) or (equals(lat[i], lat[i-1]) and equals(lon[i], lon[i-1])))):
        i -= 1
    return i

def get_aircraft_last_message(CTX:dict, flight:np.float64_2d[ax.time, ax.feature]) -> np.float64_1d:
    i = get_aircraft_last_message_t(CTX, flight)
    if (i == -1):
        return None
    return flight[i]

def get_aircraft_position(CTX:dict, flight:np.float64_2d[ax.time, ax.feature]) -> "tuple[float, float]":
    # get the aircraft last non zero latitudes and longitudes
    pos = get_aircraft_last_message(CTX, flight)
    return CTX["FG"].lat(pos), CTX["FG"].lon(pos)


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

    df = __pad__(CTX, df)

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
<<<<<<< HEAD
=======
    
    if ("timestamp_diff" in CTX["FEATURE_MAP"]):
        time = df["timestamp"]
        time_diff = np.diff(time, prepend=time[0] - 1)
        df.add_column("timestamp_diff", time_diff)
        
>>>>>>> master


    df["altitude"] = np.clip(df["altitude"], 0, None)
    df["geoaltitude"] = np.clip(df["geoaltitude"], 0, None)

    # SECONDARY FEATURES
    if ("track_diff" in CTX["FEATURE_MAP"]):
        track = df["track"]
        relative_track = track.copy()
        for i in range(1, len(relative_track)):
            relative_track[i] = angle_diff(track[i-1], track[i])
        relative_track[0] = 0
        df.add_column("track_diff", relative_track)
        
    if ("bearing" in CTX["FEATURE_MAP"]):
        df.add_column("bearing", compute_bearing(df))
        
    if ("distance" in CTX["FEATURE_MAP"]):
        df.add_column("distance", compute_distance(df))
        
<<<<<<< HEAD
=======
    # print(np.dstack((df["bearing"], df["track"])))

>>>>>>> master
    if ("bearing_diff" in CTX["FEATURE_MAP"]):
        if ("bearing" in CTX["FEATURE_MAP"]):
            bearing = df["bearing"]
        else:
            bearing = compute_bearing(df)
        
        
        df.add_column("bearing_diff", compute_bearing_diff(bearing))
        
<<<<<<< HEAD
=======
    if ("track_drift" in CTX["FEATURE_MAP"]):
        if ("bearing" in CTX["FEATURE_MAP"]):
            bearing = df["bearing"]
        else:
            bearing = compute_bearing(df)
        track = df["track"]
        track_drift = np.zeros(len(track))
        for i in range(len(track)):
            track_drift[i] = angle_diff(track[i], bearing[i])
        df.add_column("track_drift", track_drift)
        
>>>>>>> master
    if ("distance_diff" in CTX["FEATURE_MAP"]):
        if ("distance" in CTX["FEATURE_MAP"]):
            distance = df["distance"]
        else:
            distance = compute_distance(df)
        df.add_column("distance_diff", compute_distance_diff(distance))

    if ("random_angle_latitude" in CTX["FEATURE_MAP"]):
        df.add_column("random_angle_latitude", np.zeros(len(df), dtype=np.float64))
    if ("random_angle_longitude" in CTX["FEATURE_MAP"]):
        df.add_column("random_angle_longitude", np.zeros(len(df), dtype=np.float64))
    if ("random_angle_track" in CTX["FEATURE_MAP"]):
        df.add_column("random_angle_track", np.zeros(len(df), dtype=np.float64))
        
        



    if ("toulouse_0" in CTX["FEATURE_MAP"]):
        dists = toulouse_airport_distance(df["latitude"], df["longitude"])

        for i in range(len(TOULOUSE)):
            df.add_column("toulouse_"+str(i), dists[:, i])

    if ("fingerprint" in CTX["FEATURE_MAP"]):
        fgp, _ = fingerprint(df["latitude"], df["longitude"])
        df.add_column("fingerprint", fgp)
        # if the only feature is fingerprint, input data can be reduced to int8 instead of float64
        if (len(CTX["USED_FEATURES"]) == 1):
            df.cast(np.int8)

    if ("pred_distance" in CTX["FEATURE_MAP"]):
        df.add_column("pred_distance", np.zeros(len(df), dtype=np.float64))

    # remove too short flights
    if (check_length and len(df) < CTX["HISTORY"]):
        prntC(C.WARNING, "[df_to_feature_array]: flight too short")
        return []

    # filter selected features
    array = df.get_columns(CTX["USED_FEATURES"])


<<<<<<< HEAD
    if (len(array) == 0): return None
=======
    if (len(array) == 0): return []
>>>>>>> master
    return array


def __pad__(CTX:dict, df:DataFrame) -> DataFrame:
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


TOULOUSE_LATS = np.array([TOULOUSE[i]['lat'] for i in range(len(TOULOUSE))], dtype=np.float64)
TOULOUSE_LONS = np.array([TOULOUSE[i]['long'] for i in range(len(TOULOUSE))], dtype=np.float64)
def toulouse_airport_distance(lats:"list[float]", lons:"list[float]") -> "np.float64_2d[ax.sample, ax.feature]":
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
        dists[i] = GEO.np.distance(lats[i], lons[i], TOULOUSE_LATS, TOULOUSE_LONS)
  

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


def compute_bearing(df:DataFrame):
    lat = df["latitude"]
    lon = df["longitude"]
    bearing = np.zeros(len(lat))
    for i in range(1, len(bearing)):
        bearing[i] = GEO.bearing(lat[i-1], lon[i-1], lat[i], lon[i])
    return bearing


def compute_distance(df:DataFrame):
    lat = df["latitude"]
    lon = df["longitude"]
    distance = np.zeros(len(lat))
    for i in range(1, len(distance)):
        distance[i] = GEO.distance(lat[i-1], lon[i-1], lat[i], lon[i])
    return distance


def compute_bearing_diff(bearing:np.float64_1d[ax.time]):
    bearing_diff = np.zeros(len(bearing))
    for i in range(2, len(bearing)):
        bearing_diff[i] = GEO.bearing_diff(bearing[i-1], bearing[i])
    return bearing_diff


def compute_distance_diff(distance:np.float64_1d[ax.time]):
    distance_diff = np.zeros(len(distance))
    for i in range(2, len(distance)):
        distance_diff[i] = distance[i] - distance[i-1]
    return distance_diff


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


<<<<<<< HEAD
def eval_curvature(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, start:int, end:int) -> float:
    FG:FeatureGetter = CTX["FG"]
    start = max(0, start)
    end = end
    
    lat = FG.lat(x[i][start:end])
    lon = FG.lon(x[i][start:end])
=======
def eval_curvature(lat:"np.float64_1d[ax.time]", lon:"np.float64_1d[ax.time]") -> float:
    
>>>>>>> master
    a_lat, a_lon = lat[0], lon[0]
    b_lat, b_lon = lat[-1], lon[-1]
    m1_lat, m1_lon = lat[len(lat)//3], lon[len(lat)//3]
    m2_lat, m2_lon = lat[2*len(lat)//3], lon[2*len(lat)//3]

<<<<<<< HEAD

=======
>>>>>>> master
    b_a_m1 = GEO.bearing(a_lat, a_lon, m1_lat, m1_lon)
    b_m1_m2 = GEO.bearing(m1_lat, m1_lon, m2_lat, m2_lon)
    b_m2b = GEO.bearing(m2_lat, m2_lon, b_lat, b_lon)

    d1 = abs(angle_diff(b_a_m1, b_m1_m2))
    d2 = abs(angle_diff(b_m1_m2, b_m2b))

    return d1+d2

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
                or feature == "track_diff"
                or feature == "timestamp"):

            padValues[f] = 0

        elif (feature.startswith("toulouse")):
            padValues[f] = 0

        else: # default
            padValues[f] = 0
    return padValues

def splitDataset(data, ratio:float=None, size:int=None):
    """
    Split data into train, test and validation set
    """
    if (ratio is None and size is None):
        raise ValueError("splitDataset: ratio or size must be specified")
    train = []
    test = []
    for i in range(len(data)):
        if (ratio is not None):
            split_index = int(len(data[i]) * (1 - ratio))
            train.append(data[i][:split_index])
            test .append(data[i][split_index:])
        else:
            train.append(data[i][:-size])
            test .append(data[i][-size:])

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
    if (track is not None):
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
                          rotate:float = None,
                          post_flight:"np.float64_2d[ax.time, ax.feature]"=None)\
        -> """np.float64_2d[ax.time, ax.feature]
            | tuple[np.float64_2d[ax.time, ax.feature], np.float64_2d[ax.time, ax.feature]]""":

    if (relative_position is None):
        relative_position = CTX["RELATIVE_POSITION"]
    if (relative_track is None):
        relative_track = CTX["RELATIVE_TRACK"]
    if (random_track is None):
        random_track = CTX["RANDOM_TRACK"]
    
    # compute the origin of the transformation
    pos = get_aircraft_last_message(CTX, flight)
    FG:FeatureGetter = CTX["FG"]
    
    # include post flight in the transform
    x = flight
    if (post_flight is not None):
        x = np.concatenate([flight, post_flight], axis=0)
        
    # if rotate parameter set, apply the desired rotation
    if (not(relative_track) and not(random_track) and rotate is not None):
        rotation = rotate
        relative_track = True
    else:
        rotation = FG.track(pos)
    

    nan_value = np.logical_and(FG.lat(x) == FG.lat(PAD), FG.lon(x) == FG.lon(PAD))
    lat, lon, track = normalize_trajectory(CTX,
                                             FG.lat(x), FG.lon(x), FG.track(x),
                                             FG.lat(pos), FG.lon(pos), rotation,
                                             relative_position, relative_track, random_track)

    # only apply normalization on non zero lat/lon
    x[~nan_value, FG.lat()] = lat[~nan_value]
    x[~nan_value, FG.lon()] = lon[~nan_value]
    if (FG.track() is not None):
        x[~nan_value, FG.track()] = track[~nan_value]

    # fill nan lat/lon with the first non zero lat lon
    first_non_zero_ts = np.argmax(~nan_value)
    start_lat, start_lon = lat[first_non_zero_ts], lon[first_non_zero_ts]
    x[nan_value, FG.lat()] = start_lat
    x[nan_value, FG.lon()] = start_lon

    # if there is timestamp in the features, we normalize it
    if (FG.has("timestamp")):
        x[~nan_value, FG.timestamp()] -= FG.timestamp(pos)
        x[nan_value, FG.timestamp()] = 0
        x[:, FG.timestamp()] *= -1

    if (post_flight is not None):
        return x[:len(flight)], x[len(flight):]
    return x





def fingerprint(lat:np.float64_1d, lon:np.float64_1d) -> np.int8_1d:
    """
    Convert a trajectory into a fingerprint
    """

    if (len(lat) < 3):
        return np.zeros(len(lat), dtype=np.int8), np.zeros(len(lat), dtype=np.float64)

    rot = np.zeros(len(lat), dtype=np.float64)
    for i in range(0, len(lat)-1):

        d = GEO.distance(lat[i], lon[i], lat[i+1], lon[i+1])
        if (d < 0.000001):
            rot[i] = np.nan
            continue

        bx, by, bz = spherical_to_cartesian(lat[i+1], lon[i+1])
        ax, ay, az = spherical_to_cartesian(lat[i], lon[i])

        bx, by, bz = z_rotation(bx, by, bz, np.radians(-lon[i]))
        ax, ay, az = z_rotation(ax, ay, az, np.radians(-lon[i]))

        bx, by, bz = y_rotation(bx, by, bz, np.radians(-lat[i]))
        ax, ay, az = y_rotation(ax, ay, az, np.radians(-lat[i]))

        rot[i] = np.degrees(np.arctan2(bz, by))

    rot[-1] = rot[-2]


    rot_speed = np.zeros(len(lat), dtype=np.float64)
    for i in range(1, len(lat)-1):
        rot_speed[i] = angle_diff(rot[i-1], rot[i])

    fingerprint = np.zeros(len(lat), dtype=np.int8)

    # remplace nan
    fingerprint[np.isnan(rot_speed)] = 0
    for i in range(len(rot_speed)):
        if (abs(rot_speed[i]) < 0.1):
            fingerprint[i] = 0
        elif (rot_speed[i] > 0):
            fingerprint[i] = 1
        elif (rot_speed[i] < 0):
            fingerprint[i] = -1

    return fingerprint, rot_speed


<<<<<<< HEAD
=======
def convert_distance_bearing_traj_to_lat_lon(distances:np.float64_1d[ax.time], bearings:np.float64_1d[ax.time]) -> "np.float64_2d[ax.time, ax.feature]":
    new_sample = [(0, 0)]
    for t in range(1, len(distances)):
        new_sample.append(GEO.predict(new_sample[-1][0], new_sample[-1][1], bearings[t], distances[t]))
    return np.array(new_sample, dtype=np.float64)


>>>>>>> master

# def rotate_df(df:DataFrame) -> DataFrame:
#     df = df.copy()
#     angle = np.random.uniform(-np.pi, np.pi)
#     lat, lon = df["latitude"], df["longitude"]
#     lat, lon, _ = z_rotation(lat, lon, None, angle)
#     df["latitude"] = lat
#     df["longitude"] = lon
#     return df

# def scale_df(df:DataFrame) -> DataFrame:
#     df = df.copy()
#     slat, slon = np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)
#     clat, clon = np.mean(df["latitude"]), np.mean(df["longitude"])
#     lat = (df["latitude"] - clat) * slat + clat
#     lon = (df["longitude"] - clon) * slon + clon
#     df["latitude"] = lat

#     df["longitude"] = lon
#     return df



# def plot_fingerprint(fingerprint:np.int8_2d, sub_rot) -> None:
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Rectangle
#     COLORS = ["tab:green", "tab:blue", "tab:red"]

#     # fig size
#     ratio = len(fingerprint[0]) / len(fingerprint)

#     plt.figure(figsize=(10, 10/ratio))

#     for line in range(len(fingerprint)):
#         for i in range(len(fingerprint[line])):
#             plt.gca().add_patch(Rectangle((i, line), 1, 1, fill=True, color=COLORS[fingerprint[line][i]]))
#             # text

#             plt.text(i+0.5, line+0.5, str(round(sub_rot[line][i], 1)), ha='center', va='center', color="black", fontsize=8)

#     plt.xlim(0, len(fingerprint[0]))
#     plt.ylim(0, len(fingerprint))
#     plt.show()

