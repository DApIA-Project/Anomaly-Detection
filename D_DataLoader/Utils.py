
import numpy as np
import math
import pandas as pd
import os
from PIL import Image
from _Utils import Color  
from _Utils.DataFrame import DataFrame  


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


def dfToFeatures(df:DataFrame, CTX, __LIB__=False, __EVAL__=False):
    """
    Convert a complete ADS-B trajectory dataframe into a numpy array
    with the right features and preprocessing
    """

    if not(__LIB__):
        df = pad(df, CTX)

    # if nan in latitude
    if (CTX["INPUT_PADDING"] == "valid"):
        if (np.isnan(df["latitude"]).any()):
            print("NaN in latitude")
            return []
        if (np.isnan(df["longitude"]).any()):
            print("NaN in longitude")
            return []

    # add sec (60), min (60), hour (24) and day_of_week (7) features
    timestamp = pd.to_datetime(df["timestamp"], unit="s")
    df.add_column("sec", timestamp.second)
    df.add_column("min", timestamp.minute)
    df.add_column("hour", timestamp.hour)
    df.add_column("day", timestamp.dayofweek)

    # cap altitude to min = 0
    # df["altitude"] = df["altitude"].clip(lower=0)
    # df["geoaltitude"] = df["geoaltitude"].clip(lower=0)
    df.set("altitude", slice(0, len(df)), np.clip(df["altitude"], 0, None))
    df.set("geoaltitude", slice(0, len(df)), np.clip(df["geoaltitude"], 0, None))

    # add relative track
    track = df["track"]
    relative_track = track.copy()
    for i in range(1, len(relative_track)):
        relative_track[i] = angle_diff(track[i-1], track[i])
    relative_track[0] = 0
    df.add_column("relative_track", relative_track)
    df.set("timestamp", slice(0, len(df)), df["timestamp"] - df["timestamp"][0])



    # remove too short flights
    if (not(__LIB__) and not(__EVAL__) and len(df) < CTX["HISTORY"]):
        print("too short")
        return []

    # Cast booleans into numeric
    for col in df.columns:
        if (df[col].dtype == bool):
            df[col] = df[col].astype(int)

    
    # Remove useless columns
    df = df.getColumns(CTX["USED_FEATURES"])

        
    np_array = df.astype(np.float32)
    return np_array



def compute_shift(start, end, dilatation):
    """
    compute needed shift to have the last timesteps at the end of the array
    """

    d = end - start
    shift = (d - (d // dilatation) * dilatation - 1) % dilatation
    return shift


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
