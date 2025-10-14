
from numpy_typing import np, ax
from _Utils.FeatureGetter import FG_interp as FG
import D_DataLoader.Utils      as U
import _Utils.geographic_maths as GEO


def get_lat_lon(x:"np.float64_2d[ax.time, ax.feature]") -> "np.float64_2d[ax.time, ax.lat_lon]":
    if (FG.lat() is not None):
        traj = FG.lat_lon(x)
    else:
        traj = U.convert_distance_bearing_traj_to_lat_lon(FG.distance(x), FG.bearing(x)) 
    return traj


# |====================================================================================================================
# | CHECKING CLEANESS FOR TRAINING DATA
# |====================================================================================================================

def check_sample(CTX:"dict[str, object]", x:"np.float64_2d[ax.time, ax.feature]", sample_lat_lon:np.float64_2d[ax.time, ax.feature], t:int, training:bool=True) -> bool:

    lats, lons = sample_lat_lon[:,0], sample_lat_lon[:,1]
    
    MISSING_VALUES_THRESOLD = 0
    # if (not(training)):
    #     MISSING_VALUES_THRESOLD = CTX["HISTORY"] // 3 # allow 33% of missing values for validation / test

    if (t < CTX["HISTORY"] / 2):
        return False

    # count nb lats = 0
    nb = np.sum(lats == 0)
    if (nb>MISSING_VALUES_THRESOLD):
        return False
    
    # check if no consecutives lat lon are equals
    nb = 0
    for t in range(1, len(lats)):
        if (lats[t - 1] == lats[t] and \
            lons[t - 1] == lons[t]):
        
            nb += 1
            
    if (nb > MISSING_VALUES_THRESOLD):
        return False
    
    return True



# |====================================================================================================================
# | RANDOM FLIGHT PICKING
# |====================================================================================================================



def pick_random_loc(CTX:dict, x:"list[np.float64_2d[ax.time, ax.feature]]", y:"list[bool]") -> "tuple[int, int]":
    flight_i = -1
    negative = False # TODO np.random.randint(0, 100) < 1 # 1% of samples
    true_y = np.random.randint(0, 2) == 0
    t = -1
    while t < 0 or true_y != y[flight_i] or not(check_sample(CTX, sample, sample_lat_lon, t)) or U.eval_curvature(sample_lat_lon[:, 0], sample_lat_lon[:, 1]) < 15:
        
        flight_i = np.random.randint(0, len(x))
        if (negative):
            t = np.random.randint(CTX["HISTORY"]//2, CTX["HISTORY"])
        else:
            t = np.random.randint(CTX["HISTORY"], len(x[flight_i]))
            
        
        sample = x[flight_i][t-CTX["HISTORY"]+1:t+1]
        sample_lat_lon = get_lat_lon(sample)


    return flight_i, t


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
    y_batch = np.zeros((size, 1), dtype=np.float64)
    return x_batch, y_batch



def gen_random_sample(CTX:dict, 
                      x:"list[np.float64_2d[ax.time, ax.feature]]", 
                      y:"list[bool]", PAD:np.float64_1d)\
        -> "tuple[np.float64_2d[ax.time, ax.feature], np.float64_1d[ax.feature]]":
    
    i, t = pick_random_loc(CTX, x, y)
    x_sample, y_sample, _ = gen_sample(CTX, x, y, PAD, i, t, valid=True)
    return x_sample, y_sample




def gen_sample(CTX:dict,
               x:"list[np.float64_2d[ax.time, ax.feature]]",
               y:"list[bool]", PAD:np.float64_1d,
               i:int, t:int, valid:bool=None, training:bool=True)\
        -> """tuple[np.float64_2d[ax.time, ax.feature],
                    np.float64_1d[ax.feature],
                    bool]""":

    if (valid is None): 
        sample = x[i][t-CTX["HISTORY"]+1:t+1]
        sample_lat_lon = get_lat_lon(sample)
        valid = check_sample(CTX, sample, sample_lat_lon, t, training)
        
    x_sample = alloc_sample(CTX)
    if (not(valid)): return x_sample, None, valid


    start, end, _, pad_lenght, shift = U.window_slice(CTX, t)
    x_sample[pad_lenght:] = x[i][start+shift:end:CTX["DILATION_RATE"]]
    x_sample[:pad_lenght] = PAD
    
    y_sample = [y[i]]
    
    
    
    track = GEO.bearing(FG.lat(x_sample[pad_lenght]), FG.lon(x_sample[pad_lenght]), 
                        FG.lat(x_sample[-1]), FG.lon(x_sample[-1]))
        
    
    # apply random rotation on trajectory
    if ("random_angle_latitude" in CTX["FEATURE_MAP"] and "random_angle_longitude" in CTX["FEATURE_MAP"]):
        
        Olat = FG.lat(x_sample[CTX["INPUT_LEN"]//2])
        Olon = FG.lon(x_sample[CTX["INPUT_LEN"]//2])
        rlat, rlon, rtrack = U.normalize_trajectory(CTX, FG.lat(x_sample), FG.lon(x_sample), FG.track(x_sample),
                                                         Olat,             Olon,             None,  
                                                         relative_position=CTX["RELATIVE_POSITION"], 
                                                         relative_track=False, 
                                                         random_track=True)
        
        x_sample[:, FG.get("random_angle_latitude")] = rlat
        x_sample[:, FG.get("random_angle_longitude")] = rlon
        
        if ("random_angle_track" in CTX["FEATURE_MAP"]):
            x_sample[:, FG.get("random_angle_track")] = rtrack
            


    pre_flight = x_sample[0:CTX["INPUT_LEN"]//2]
    post_flight = x_sample[CTX["INPUT_LEN"]//2:]
    
    x_sample = U.batch_preprocess(CTX, pre_flight, PAD, rotate=track, post_flight=post_flight)
    x_sample = np.concatenate(x_sample)    
    
    

    return x_sample, y_sample, valid





