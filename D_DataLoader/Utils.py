import numpy as np
import pandas as pd
import os
import math

import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.DataFrame import DataFrame
from _Utils.Limits import *

from D_DataLoader.Airports import TOULOUSE

###################################################
# MATHS
###################################################

def Xrotation(x, y, z, t):
    return x, y * np.cos(-t) - z * np.sin(-t), y * np.sin(-t) + z * np.cos(-t)

def Yrotation(x, y, z, t):
    return x * np.cos(-t) + z * np.sin(-t), y, -x * np.sin(-t) + z * np.cos(-t)

def Zrotation(x, y, z, t):
    return x * np.cos(t) - y * np.sin(t), x * np.sin(t) + y * np.cos(t), z

def spherical_to_cartesian(lat, lon):
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))
    return x, y, z

def cartesian_to_spherical(x, y, z):
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def latlondistance(lat1, lon1, lat2, lon2):
    """Return the distance in meters between two points"""
    R = 6371e3 # metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(lon2-lon1)

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = R * c
    return d

def angle_diff(a, b):
    a = a % 360
    b = b % 360

    # compute relative angle
    diff = b - a

    if (diff > 180):
        diff -= 360
    elif (diff < -180):
        diff += 360
    return diff


###################################################
# TRAJECTORY PREPROCESSING
###################################################

def compute_shift(start, end, dilatation):
    """
    compute needed shift to have the last timesteps at the end of the array
    """

    d = end - start
    shift = (d - (d // dilatation) * dilatation - 1) % dilatation
    return shift

def windowBounds(CTX, t):
    """
    Compute the bounds of the window that should end at t (included)

    Returns:
    --------

    start: int
    end: int
        The [start:end] slice of the window

    length: int
        The length of the window

    pad_lenght: int
        The number of padding to add to the window

    shift: int
        The shift to apply to the window to have the last timestep at the end even with dilatation
    """
    start = max(0, t+1-CTX["HISTORY"])
    end = t+1
    length = end - start
    pad_lenght = (CTX["HISTORY"] - length)//CTX["DILATION_RATE"]
    shift = compute_shift(start, end, CTX["DILATION_RATE"])

    return start, end, length, pad_lenght, shift


def listFlight(path, limit=INT_MAX):
    filenames = os.listdir(path)
    filenames = [f for f in filenames if f.endswith(".csv")]
    filenames.sort()
    return filenames[:limit]


def read_trajectory(path, file=None) -> pd.DataFrame:
    """
    Read a trajectory from a csv or other file
    """
    if (file != None):
        path = os.path.join(path, file)
    return pd.read_csv(path, sep=",",dtype={"callsign":str, "icao24":str})


def dfToFeatures(df:DataFrame, CTX, check_length=True):
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
            prntC(C.WARNING, "[dfToFeatures]:", "NaN in latitude")
            return []
        if (np.isnan(df["longitude"]).any()):
            prntC(C.WARNING, "[dfToFeatures]:", "NaN in longitude")
            return []

    # add sec (60), min (60), hour (24) and day_of_week (7) features
    timestamp = df["timestamp"]
    df.add_column("day", (timestamp//86400 + 4) % 7)
    df.add_column("hour", (timestamp//3600 + 1) % 24)
    df.add_column("min", (timestamp//60) % 60)
    df.add_column("sec", timestamp % 60)


    # cap altitude to min = 0
    # df["altitude"] = df["altitude"].clip(lower=0)
    # df["geoaltitude"] = df["geoaltitude"].clip(lower=0)
    df.setColumValue("altitude", slice(0, len(df)), np.clip(df["altitude"], 0, None))
    df.setColumValue("geoaltitude", slice(0, len(df)), np.clip(df["geoaltitude"], 0, None))

    # add relative track
    track = df["track"]
    relative_track = track.copy()
    for i in range(1, len(relative_track)):
        relative_track[i] = angle_diff(track[i-1], track[i])
    relative_track[0] = 0
    df.add_column("relative_track", relative_track)
    df.setColumValue("timestamp", slice(0, len(df)), df["timestamp"] - df["timestamp"][0])


    if ("toulouse_0" in CTX["USED_FEATURES"]):
        dists = toulouse_airportDistance(df["latitude"], df["longitude"])

        for i in range(len(TOULOUSE)):
            df.add_column("toulouse_"+str(i), dists[:, i])



    # remove too short flights
    if (check_length and len(df) < CTX["HISTORY"]):
        prntC(C.WARNING, "[dfToFeatures]: flight too short")
        return []

    # Cast booleans into numeric
    for col in df.columns:
        if (df[col].dtype == bool):
            df[col] = df[col].astype(int)


    # Remove useless columns
    df = df.getColumns(CTX["USED_FEATURES"])


    array = df.astype(np.float32)

    if (len(array) == 0): return None
    return array

TOULOUSE_LATS = np.array([TOULOUSE[i]['lat'] for i in range(len(TOULOUSE))], dtype=np.float64)
TOULOUSE_LONS = np.array([TOULOUSE[i]['long'] for i in range(len(TOULOUSE))], dtype=np.float64)

def toulouse_airportDistance(lats, lons):
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


def analysis(CTX, dataframe):
    """ dataframe : (sample, timestep, feature)"""
    minValues = np.full(CTX["FEATURES_IN"], np.nan)
    maxValues = np.full(CTX["FEATURES_IN"], np.nan)

    for i in range(len(dataframe)):
        minValues = np.nanmin([minValues, np.nanmin(dataframe[i], axis=0)], axis=0)
        maxValues = np.nanmax([maxValues, np.nanmax(dataframe[i], axis=0)], axis=0)

    return minValues, maxValues

def genPadValues(CTX, dataframe):
    minValues = analysis(CTX, dataframe)[0]
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

def normalize_trajectory(CTX, lat, lon, track, Olat, Olon, Otrack, relative_position, relative_track, random_track):
    ROT = 0
    LAT = -CTX["BOX_CENTER"][0]
    LON = -CTX["BOX_CENTER"][1]
    if relative_position:
        LAT = -Olat
        LON = -Olon
    if relative_track:
        ROT = -Otrack
    if random_track:
        ROT = np.random.randint(0, 360)

    x, y, z = spherical_to_cartesian(lat, lon)
    x, y, z = Zrotation(x, y, z, np.radians(LON)) # Normalize longitude with Z rotation
    x, y, z = Yrotation(x, y, z, np.radians(LAT)) # Normalize latitude with Y rotation
    x, y, z = Xrotation(x, y, z, np.radians(ROT)) # Rotate the fragment with the random angle along X axis
    lat, lon = cartesian_to_spherical(x, y, z)
    track = np.remainder(track + ROT, 360)

    return lat, lon, track


def undo_normalize_trajectory(CTX, lat, lon, Olat, Olon, Otrack, relative_position, relative_track):
    ROT = 0
    LAT = -CTX["BOX_CENTER"][0]
    LON = -CTX["BOX_CENTER"][1]
    if relative_position:
        LAT = -Olat
        LON = -Olon
    if relative_track:
        ROT = -Otrack

    x, y, z = spherical_to_cartesian(lat, lon)
    x, y, z = Xrotation(x, y, z, np.radians(-ROT)) # Rotate the fragment with the random angle along X axis
    x, y, z = Yrotation(x, y, z, np.radians(-LAT)) # Normalize latitude with Y rotation
    x, y, z = Zrotation(x, y, z, np.radians(-LON)) # Normalize longitude with Z rotation
    lat, lon = cartesian_to_spherical(x, y, z)

    return lat, lon
