import numpy as np
from D_DataLoader.Utils import normalize_trajectory, latlondistance, undo_normalize_trajectory

def equals_float(a, b, epsilon=0.0001):
    return abs(a - b) < epsilon
def not_equals_float(a, b, epsilon=0.0001):
    return abs(a - b) > epsilon





def checkTrajectory(CTX, x, i, t):
    LAT_I = CTX["FEATURE_MAP"]["latitude"]
    LON_I = CTX["FEATURE_MAP"]["longitude"]
    HORIZON = CTX["HORIZON"]


    if (x[i][t, CTX["FEATURE_MAP"]["pad"]] > 0.5):
        return False
    if (x[i][t+HORIZON, CTX["FEATURE_MAP"]["pad"]] > 0.5):
        return False
    
    ts_actu = x[i][t, CTX["FEATURE_MAP"]["timestamp"]]
    ts_pred = x[i][t+HORIZON, CTX["FEATURE_MAP"]["timestamp"]]

    if (ts_actu + HORIZON != ts_pred):
        return False
    
    for ts in range(t - CTX["DILATION_RATE"] + 1, t + HORIZON + 1):
        coord_actu = x[i][ts - 1, LAT_I], x[i][ts - 1, LON_I]
        coord_next = x[i][ts, LAT_I], x[i][ts, LON_I]
        
        d = latlondistance(coord_actu[0], coord_actu[1], coord_next[0], coord_next[1])
        if (d > 200 or d < 1.0):
            return False
    
    return True


    

def pick_an_interesting_aircraft(CTX, x):
    HORIZON = CTX["HORIZON"]

    flight_i = np.random.randint(0, len(x))
    t = np.random.randint(0, len(x[flight_i])-HORIZON)

    while not(checkTrajectory(CTX, x, flight_i, t)):

        flight_i = np.random.randint(0, len(x))
        t = np.random.randint(0, len(x[flight_i])-HORIZON)

        
    return flight_i, t





def batchPreProcess(CTX, flight, relative_position=False, relative_track=False, random_track=False):
    # normalisation origin : the last, not null point of the trajectory
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    track = flight[:, CTX["FEATURE_MAP"]["track"]]
    i = len(lat)-2
    while (i >= 0 and (lat[i] == 0 and lon[i] == 0)):
        i -= 1
    if (i == -1):
        return None
    return normalize_trajectory(flight, CTX, lat[i], lon[i], track[i], relative_position, relative_track, random_track)

def undo_batchPreProcess(CTX, Olat, Olon, Otrack, lat, lon, relative_position=False, relative_track=False, random_track=False):
    return undo_normalize_trajectory(CTX, lat, lon, Olat, Olon, Otrack, relative_position, relative_track, random_track)




def distance(CTX, y, y_):
    lat1 = y[:, CTX["PRED_FEATURE_MAP"]["latitude"]]
    lon1 = y[:, CTX["PRED_FEATURE_MAP"]["longitude"]]
    lat2 = y_[:, CTX["PRED_FEATURE_MAP"]["latitude"]]
    lon2 = y_[:, CTX["PRED_FEATURE_MAP"]["longitude"]]
    distances = latlondistance(lat1, lon1, lat2, lon2)

    return np.mean(distances)
