
import _Utils.Color            as C
from   _Utils.Color import prntC
import _Utils.FeatureGetter    as FG
import _Utils.geographic_maths as GEO
import _Utils.plotADSB         as PLT
from _Utils.numpy import np, ax

import D_DataLoader.Utils      as U

import matplotlib.pyplot as plt

# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================


def check_sample(CTX:"dict[str, object]", x:"np.ndarray", i:int, t:int) -> bool:
    lats = FG.lat(x[i])
    lons = FG.lon(x[i])
    HORIZON = CTX["HORIZON"]

    if (t <= HORIZON + 1):
        return False

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
    dist_values, i = np.zeros((HORIZON + HORIZON)), 0

    for t in range(t - HORIZON + 1, t + HORIZON + 1):
        d = GEO.distance(lats[t-1], lons[t-1], lats[t], lons[t])
        dist_values[i] = d
        i += 1

    if (np.max(dist_values) > 400 or np.min(dist_values) < 1.0):
        return False

    # print(np.max(np.diff(dist_values)))
    if (np.max(np.diff(dist_values)) > 45):
        return False

    return True


# |====================================================================================================================
# | BATCH GENERATION
# |====================================================================================================================

def alloc_sample(CTX:dict)\
        -> "np.float64_2d[ax.time, ax.feature]":

    x_sample = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    return x_sample

def alloc_batch(CTX:dict, size:int) -> """tuple[
        np.float64_3d[ax.batch, ax.time, ax.feature],
        np.float64_2d[ax.batch, ax.feature]]""":

    x_batch = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]))
    y_batch = np.zeros((size, CTX["FEATURES_OUT"]))
    return x_batch, y_batch

def gen_random_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", PAD:np.float64_1d)\
        -> "tuple[np.float64_2d[ax.time, ax.feature], np.float64_1d[ax.feature], tuple[float, float]]":
    i, t = pick_random_loc(CTX, x)
    x_sample, y_sample, _, origin = gen_sample(CTX, x, PAD, i, t, valid=True)
    y_sample = FG.lat_lon(y_sample)
    return x_sample, y_sample, origin


def eval_curvature(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, t:int) -> float:
    start = max(0, t-CTX["HISTORY"])
    end = t+CTX["HORIZON"]
    lat = FG.lat(x[i][start:end])
    lon = FG.lon(x[i][start:end])
    a_lat, a_lon = lat[0], lon[0]
    b_lat, b_lon = lat[-1], lon[-1]
    m1_lat, m1_lon = lat[len(lat)//3], lon[len(lat)//3]
    m2_lat, m2_lon = lat[2*len(lat)//3], lon[2*len(lat)//3]


    b_a_m1 = GEO.bearing(a_lat, a_lon, m1_lat, m1_lon)
    b_m1_m2 = GEO.bearing(m1_lat, m1_lon, m2_lat, m2_lon)
    b_m2b = GEO.bearing(m2_lat, m2_lon, b_lat, b_lon)

    d1 = abs(U.angle_diff(b_a_m1, b_m1_m2))
    d2 = abs(U.angle_diff(b_m1_m2, b_m2b))

    return d1+d2

def pick_random_loc(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]") -> "tuple[int, int]":
    HORIZON = CTX["HORIZON"]
    flight_i = np.random.randint(0, len(x))
    negative = np.random.randint(0, 100) < 10
    if (negative):
        t = np.random.randint(0, CTX["INPUT_LEN"])
    else:
        t = np.random.randint(0, len(x[flight_i])-HORIZON)

    while not(check_sample(CTX, x, flight_i, t)) or eval_curvature(CTX, x, flight_i, t) < 20:
        flight_i = np.random.randint(0, len(x))
        if (negative):
            t = np.random.randint(0, CTX["INPUT_LEN"])
        else:
            t = np.random.randint(0, len(x[flight_i])-HORIZON)

    return flight_i, t



def gen_sample(CTX:dict,
               x:"list[np.float64_2d[ax.time, ax.feature]]",
               PAD:np.float64_1d,
               i:int, t:int, valid:bool=None, training:bool=True)\
        -> """tuple[np.float64_2d[ax.time, ax.feature],
                    np.float64_1d[ax.feature],
                    bool, tuple[float, float, float]]""":

    if (valid is None): valid = check_sample(CTX, x, i, t)
    x_sample = alloc_sample(CTX)
    if (not(valid)): return x_sample, None, valid, (0, 0, 0)


    start, end, length, pad_lenght, shift = U.window_slice(CTX, t)
    x_sample[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
    x_sample[:pad_lenght] = PAD

    last_message = U.get_aircraft_last_message(CTX, x_sample)
    lat, lon, track = FG.lat(last_message), FG.lon(last_message), FG.track(last_message)

    y_sample = x[i][t+CTX["HORIZON"]]


    random_track = CTX["RANDOM_TRACK"] and training
    relative_track = CTX["RELATIVE_TRACK"]
    if (CTX["RANDOM_TRACK"] and not(training)):
        relative_track = True

    x_sample, y_sample = U.batch_preprocess(CTX, x_sample, PAD,
                                relative_track=relative_track,
                                random_track=random_track,
                                post_flight = np.array([y_sample]))

    return x_sample, y_sample[0], valid, (lat, lon, track)





