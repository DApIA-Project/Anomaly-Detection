import D_DataLoader.Utils as U


from numpy_typing import np, ax, ax
from _Utils.FeatureGetter import FG_replay as FG


# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================

def check_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, t:int) -> bool:
    if (t < CTX["HISTORY"] - 1): return False
    if (t >= len(x[i])): return False

    sample = x[i][t-CTX["HISTORY"]+1:t+1]
    nb_left, nb_wild, nb_right = np.bincount(FG.get(sample, "fingerprint")+1, minlength=3)

    if (nb_wild > CTX["WHILDCARD_LIMIT"]): return False
    if (nb_left < CTX["MIN_DIVERSITY"]): return False
    if (nb_right < CTX["MIN_DIVERSITY"]): return False


    return True

# |====================================================================================================================
# | RANDOM FLIGHT PICKING
# |====================================================================================================================


def pick_random_loc(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]") -> "tuple[int, int]":

    i = np.random.randint(0, len(x))
    t = np.random.randint(CTX["HISTORY"] - 1, len(x[i]))
    while not(check_sample(CTX, x, i, t)):
        i = np.random.randint(0, len(x))
        t = np.random.randint(CTX["HISTORY"] - 1, len(x[i]))

    return i, t


# |====================================================================================================================
# | SAMPLE GENERATION
# |====================================================================================================================


def alloc_batch(CTX:dict, batch_size:int) -> """tuple[
        np.float64_3d[ax.sample, ax.time, ax.feature],
        np.str_1d[ax.sample]]""":

    x_batch = np.zeros((batch_size, CTX["INPUT_LEN"],CTX["FEATURES_IN"]), dtype=np.float64)
    y_batch = np.full((batch_size,), np.nan, dtype="U256")
    return x_batch, y_batch


def gen_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", i:int, t:int, valid:bool=None)\
        -> "tuple[np.float64_2d[ax.time, ax.feature], bool]":

    if (valid is None): valid = check_sample(CTX, x, i, t)
    if (not valid):
        return None, False

    start, end, _, _, _ = U.window_slice(CTX, t)
    x_sample = x[i][start:end]
    return x_sample, valid


def gen_random_sample(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]")\
        -> "tuple[np.float64_2d[ax.time, ax.feature], tuple[int, int]]":

    i, t = pick_random_loc(CTX, x)
    x_sample, _ = gen_sample(CTX, x, i, t, valid=True)
    return x_sample, (i, t)





# |====================================================================================================================
# | SIMPLE GEOMETRIC TRANSFORMATIONS TO CHECK THE ROBUSTNESS OF THE MODEL
# |====================================================================================================================

def alter(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":

    i = np.random.choice(["rotate", "translate", "scale", "flip", "noise", "trim"])
        #, "drop"])

    if (i == "rotate"):
        return __rotate__(x)
    elif (i == "translate"):
        return __translate__(x)
    elif (i == "scale"):
        return __scale__(x)
    elif (i == "flip"):
        return __flip__(x)
    elif (i == "noise"):
        return __noise__(x)
    elif (i == "trim"):
        return __trim__(x)
    elif (i == "drop"):
        return __drop__(x)


def __rotate__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    angle = np.random.uniform(-np.pi, np.pi)
    lat, lon, _ = U.z_rotation(lat, lon, None, angle)
    FG.set(x, "latitude", lat)
    FG.set(x, "longitude", lon)
    return x, "rotate"

def __translate__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    lat += np.random.uniform(-0.1, 0.1)
    lon += np.random.uniform(-0.1, 0.1)
    FG.set(x, "latitude", lat)
    FG.set(x, "longitude", lon)
    return x, "translate"

def __scale__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    slat, slon = np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)
    clat, clon = np.mean(lat), np.mean(lon)
    lat = (lat - clat) * slat + clat
    lon = (lon - clon) * slon + clon
    FG.set(x, "latitude", lat)
    FG.set(x, "longitude", lon)
    return x, "scale"

def __flip__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    slat, slon = np.random.choice(["ab", "ba"])
    slat, slon = (1, -1) if slat == "ab" else (-1, 1)
    lat = (lat) * slat
    lon = (lon) * slon
    FG.set(x, "latitude", lat)
    FG.set(x, "longitude", lon)
    return x, "flip"

def __noise__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    lat += np.random.uniform(-0.000005, 0.000005, size=(len(lat),))
    lon += np.random.uniform(-0.000005, 0.000005, size=(len(lon),))
    FG.set(x, "latitude", lat)
    FG.set(x, "longitude", lon)
    return x, "noise"

def __trim__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    length = np.random.randint(CTX["HISTORY"] * 5, CTX["HISTORY"] * 10)
    if (length > len(lat)):
        length = len(lat)
    start = np.random.randint(0, len(lat) - length)
    return x[start:start+length], "trim"

def __drop__(x:np.float64_2d[ax.time, ax.feature]) -> "tuple[np.float64_2d[ax.time, ax.feature], str]":
    lat, lon = FG.lat(x), FG.lon(x)
    level = np.random.randint(50, 90)
    p = level / 100
    indexs = np.arange(len(lat))
    indexs = indexs[np.random.uniform(0, 1, size=(len(indexs),)) > p]
    x = x[indexs]
    return x, "drop" + str(level)