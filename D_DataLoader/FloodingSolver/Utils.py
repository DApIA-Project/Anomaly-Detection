
from _Utils.FeatureGetter import FG_flooding as FG
import _Utils.geographic_maths as GEO
import _Utils.plotADSB         as PLT
from numpy_typing import np, ax, ax

import D_DataLoader.Utils      as U


# |====================================================================================================================
# | CACHE
# |====================================================================================================================





# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================

def check_sample(CTX:"dict[str, object]", x:"np.ndarray", i:int, t:int, t_:int, training:bool=True) -> bool:

    lats = FG.lat(x[i])
    lons = FG.lon(x[i])
    HORIZON = CTX["HORIZON"]

    if (t < CTX["HISTORY"] / 2):
        return False

    if (lats[t] == 0 and lons[t] == 0):
        return False
    if (lats[t_] == 0 and lons[t_] == 0):
        return False

    # # count nb lats = 0
    # nb = np.sum(lats == 0)
    # if (nb/float(len(lats))>0.333):
    #     return False

    # Check there is a y value for the prediction
    ts_actu = FG.timestamp(x[i][t])
    ts_pred = FG.timestamp(x[i][t_])
    if (ts_actu + HORIZON != ts_pred):
        return False

    # # Check there is no abnormal distance between two consecutive points (only at the end of the trajectory)
    # dist_values, i = np.zeros((HORIZON + HORIZON)), 0

    # for t in range(t - HORIZON + 1, t + HORIZON + 1):
    #     d = GEO.distance(lats[t-1], lons[t-1], lats[t], lons[t])
    #     dist_values[i] = d
    #     i += 1

    # if (np.min(dist_values) < 1.0):
    #     return False

    # if (training):
    #     if (np.max(dist_values) > 400):
    #         return False

    #     if (np.max(np.abs(np.diff(dist_values))) > 35):
    #         return False

    return True

def get_t_(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, t:int) -> int:
    """
    Find the index of the prediction point in the trajectory.
    usually t_ = t + CTX["HORIZON"] but if there is a small gap we can sometimes find the right index.
    Eg. timestamp = [0, 1, 3, 4] Horizon = 3
    """
    ts_actu = FG.timestamp(x[i][t])
    t_ = t + CTX["HORIZON"]
    while FG.timestamp(x[i][t_]) > ts_actu + CTX["HORIZON"]:
        t_ -= 1
    return t_



def eval_curvature(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, t:int, t_:int) -> float:
    """
    Evaluate the curvature degree of the trajectory.

    Used in order to filter straight trajectories that are too easy for training the model.
    """
    start = max(0, t-CTX["HISTORY"])

    end = t_
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


# |====================================================================================================================
# | RANDOM FLIGHT PICKING
# |====================================================================================================================


def pick_random_loc(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]") -> "tuple[int, int]":
    HORIZON = CTX["HORIZON"]
    flight_i = np.random.randint(0, len(x))
    negative = np.random.randint(0, 100) < 5 # 5% of samples
    t, t_ = -1, -1


    while t < 0 or not(check_sample(CTX, x, flight_i, t, t_)) or eval_curvature(CTX, x, flight_i, t, t_) < 20:
        flight_i = np.random.randint(0, len(x))
        if (negative):
            t = np.random.randint(CTX["HISTORY"]//2, CTX["HISTORY"])
        else:
            t = np.random.randint(CTX["HISTORY"], len(x[flight_i])-HORIZON)
        t_ = get_t_(CTX, x, flight_i, t)

    return flight_i, t, t_


# |====================================================================================================================
# | SAMPLE GENERATION
# |====================================================================================================================


def alloc_sample(CTX:dict)\
        -> "np.float64_2d[ax.time, ax.feature]":

    x_sample = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]), dtype=np.float64)
    return x_sample

def alloc_batch(CTX:dict, size:int) -> """tuple[
        np.float64_3d[ax.sample, ax.time, ax.feature],
        np.float64_2d[ax.sample, ax.feature]]""":


    x_batch = np.zeros((size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]), dtype=np.float64)
    y_batch = np.zeros((size, CTX["FEATURES_OUT"]), dtype=np.float64)
    return x_batch, y_batch



def gen_random_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", PAD:np.float64_1d)\
        -> "tuple[np.float64_2d[ax.time, ax.feature], np.float64_1d[ax.feature], tuple[float, float]]":
    i, t, t_ = pick_random_loc(CTX, x)
    x_sample, y_sample, _, origin = gen_sample(CTX, x, PAD, i, t, t_, valid=True)
    y_sample = FG.lat_lon(y_sample)
    return x_sample, y_sample, origin


def gen_sample(CTX:dict,
               x:"list[np.float64_2d[ax.time, ax.feature]]",
               PAD:np.float64_1d,
               i:int, t:int, t_:int, valid:bool=None, training:bool=True)\
        -> """tuple[np.float64_2d[ax.time, ax.feature],
                    np.float64_1d[ax.feature],
                    bool, tuple[float, float, float]]""":

    if (valid is None): valid = check_sample(CTX, x, i, t, t_, training)
    x_sample = alloc_sample(CTX)
    if (not(valid)): return x_sample, None, valid, (0, 0, 0)


    start, end, _, pad_lenght, shift = U.window_slice(CTX, t)
    x_sample[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
    x_sample[:pad_lenght] = PAD

    last_message = U.get_aircraft_last_message(CTX, x_sample)
    lat, lon, track = FG.lat(last_message), FG.lon(last_message), FG.track(last_message)

    y_sample = x[i][t_]

    if ("distance_var" in CTX["USED_FEATURES"]):
        distance = GEO.distance(lat, lon, FG.lat(x[i][t_]), FG.lon(x[i][t_]))
        x_sample[:, CTX["FEATURE_MAP"]["distance_var"]] = distance



    random_track = CTX["RANDOM_TRACK"] and training
    relative_track = CTX["RELATIVE_TRACK"] or (CTX["RANDOM_TRACK"] and not(training))

    x_sample, y_sample = U.batch_preprocess(CTX, x_sample, PAD,
                                relative_track=relative_track,
                                random_track=random_track,
                                post_flight = np.array([y_sample]))

    return x_sample, y_sample[0], valid, (lat, lon, track)





