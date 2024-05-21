import numpy as np

import _Utils.Color            as C
from   _Utils.Color import prntC
import _Utils.FeatureGetter    as FG
import _Utils.geographic_maths as GEO
from   _Utils.Typing import NP, AX

import D_DataLoader.Utils      as U


# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================


def check_sample(CTX:"dict[str, object]", x:"np.ndarray", i:int, t:int) -> bool:
    lats = FG.lat(x[i])
    lons = FG.lon(x[i])
    HORIZON = CTX["HORIZON"]

    if (lats[t] == 0 and lons[t] == 0):
        return False
    if (lats[t+HORIZON] == 0 and lons[t+HORIZON] == 0):
        return False

    # Check there is no missing timestamp between last timestamp t and prediction timestamp t+HORIZON
    ts_actu = FG.timestamp(x[i][t])
    ts_pred = FG.timestamp(x[i][t+HORIZON])
    if (ts_actu + HORIZON != ts_pred):
        return False

    # Check there is no abnormal distance between two consecutive points (only at the end of the trajectory)
    for t in range(t - CTX["DILATION_RATE"] + 1, t + HORIZON + 1):
        d = GEO.distance(lats[t-1], lons[t-1], lats[t], lons[t])
        if (d > 200 or d < 1.0):
            return False

    return True


# |====================================================================================================================
# | BATCH GENERATION
# |====================================================================================================================

def alloc_sample(CTX:dict)\
        -> "NP.float32_2d[AX.time, AX.feature]":

    x_sample = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    return x_sample

def alloc_batch(CTX:dict, size:int) -> """tuple[
        NP.float32_3d[AX.batch, AX.time, AX.feature],
        NP.float32_2d[AX.batch, AX.feature]]""":

    x_batch = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    y_batch = np.zeros((size, CTX["FEATURES_OUT"]))
    return x_batch, y_batch

def gen_random_sample(CTX:dict, x:"list[NP.float32_2d[AX.time, AX.feature]]", PAD:NP.float32_1d)\
        -> "tuple[NP.float32_2d[AX.time, AX.feature], NP.float32_1d[AX.feature]]":
    i, t = pick_random_loc(CTX, x)
    x_sample, _ = gen_sample(CTX, x, PAD, i, t, valid=True)
    y_sample = FG.lat_lon(x[i][t+CTX["HORIZON"]])
    return x_sample, y_sample

def pick_random_loc(CTX:dict, x:"list[NP.float32_2d[AX.time, AX.feature]]") -> "tuple[int, int]":
    HORIZON = CTX["HORIZON"]
    flight_i = np.random.randint(0, len(x))
    t = np.random.randint(0, len(x[flight_i])-HORIZON)

    while not(check_sample(CTX, x, flight_i, t)):
        flight_i = np.random.randint(0, len(x))
        t = np.random.randint(0, len(x[flight_i])-HORIZON)

    return flight_i, t



def gen_sample(CTX:dict,
               x:"list[NP.float32_2d[AX.time, AX.feature]]",
               PAD:NP.float32_1d,
               i:int, t:int, valid:bool=None)\
        -> "tuple[NP.float32_2d[AX.time, AX.feature], bool]":

    if (valid is None): valid = check_sample(CTX, x, i, t)
    x_batch = alloc_sample(CTX)
    if (not(valid)): return x_batch


    start, end, length, pad_lenght, shift = U.window_slice(CTX, t)
    x_batch[pad_lenght:] = x[i][start:end:CTX["DILATION_RATE"]]
    x_batch[:pad_lenght] = PAD

    last_message = U.get_aircraft_last_message(CTX, x_batch)
    lat, lon = FG.lat(last_message), FG.lon(last_message)
    #TODO remove this check
    if (lat == FG.lat(PAD) or lon == FG.lon(PAD)):
        prntC(C.ERROR, "ERROR: lat or lon is 0")
        prntC(list(range(start, end, CTX["DILATION_RATE"])))
        prntC(FG.lat(x[i][start:end]))
        prntC(i, t, start, end, length, pad_lenght, shift)
        prntC(C.ERROR, "ERROR: lat or lon is 0")

    x_batch = U.batch_preprocess(CTX, x_batch, CTX["RELATIVE_POSITION"], CTX["RELATIVE_TRACK"], CTX["RANDOM_TRACK"])
    return x_batch, valid



def batch_preprocess(CTX, flight, relative_position=False, relative_track=False, random_track=False):
    pos = U.get_aircraft_last_message(CTX, flight)
    nan_value = np.logical_and(FG.lat(flight) == FG.lat(PAD), FG.lon(flight) == FG.lon(PAD))
    lat, lon, track = U.normalize_trajectory(CTX,
                                  FG.lat(flight), FG.lon(flight), FG.track(flight),
                                  FG.lat(pos), FG.lon(pos), FG.track(pos),
                                  relative_position, relative_track, random_track)

    # only apply normalization on non zero lat/lon
    flight[~nan_value, FG.lat()] = lat[~nan_value]
    flight[~nan_value, FG.lon()] = lon[~nan_value]
    flight[~nan_value, FG.track()] = track[~nan_value]

def undo_batch_preprocess(CTX, Olat, Olon, Otrack, lat, lon, relative_position=False, relative_track=False, random_track=False):
    return U.undo_normalize_trajectory(CTX, lat, lon, Olat, Olon, Otrack, relative_position, relative_track, random_track)




