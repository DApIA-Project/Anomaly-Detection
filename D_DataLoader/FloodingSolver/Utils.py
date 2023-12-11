import numpy as np
from D_DataLoader.Utils import compute_shift

def equals_float(a, b, epsilon=0.0001):
    return abs(a - b) < epsilon
def not_equals_float(a, b, epsilon=0.0001):
    return abs(a - b) > epsilon




def getAircraftPosition(CTX, flight):
    # get the aircraft last non zero latitudes and longitudes
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    track = flight[:, CTX["FEATURE_MAP"]["track"]]
    i = len(lat)-1
    while (i >= 0 and (lat[i] == 0 and lon[i] == 0)):
        i -= 1
    if (i == -1):
        return None
    return lat[i], lon[i], track[i]


def latlondistance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two points on earth
    """
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
    
    # start = max(0, t+1-CTX["HISTORY"])
    # end = t+1
    # shift = compute_shift(start, end, CTX["DILATION_RATE"])

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









def Xrotation(x, y, z, r):
    xx = x
    yx = y * np.cos(r) - z * np.sin(r)
    zx = y * np.sin(r) + z * np.cos(r)
    return xx, yx, zx

def Yrotation(x, y, z, r):
    xy = x * np.cos(r) + z * np.sin(r)
    yy = y
    zy = -x * np.sin(r) + z * np.cos(r)
    return xy, yy, zy

def Zrotation(x, y, z, r):
    xz = x * np.cos(r) - y * np.sin(r)
    yz = x * np.sin(r) + y * np.cos(r)
    zz = z
    return xz, yz, zz

def batchPreProcess(CTX, flight, relative_position=False, relative_track=False, random_track=False):
    """
    Additional method of preprocessing after
    batch generation.

    Normalize lat, lon of a flight fragment to 0, 0.
    Rotate the fragment with a random angle to avoid
    biais of the model.
 
    Parameters:
    -----------

    flight: np.array
        A batch of flight fragment

    Returns:
    --------

    flight: np.array
        Same batch but preprocessed


    """
    # Get the index of each feature by name for readability
    FEATURE_MAP = CTX["FEATURE_MAP"]
    lat = flight[:, FEATURE_MAP["latitude"]]
    lon = flight[:, FEATURE_MAP["longitude"]]
    track = flight[:, FEATURE_MAP["track"]]

    last_lat, last_lon, last_track = getAircraftPosition(CTX, flight[:-1])
    non_zeros_lat_lon = np.logical_or(lat != 0, lon != 0)


    # do not change angle, and rotate the whole bounding box to 0, 0 (not relative just normalizing)
    R = 0
    Y = CTX["BOX_CENTER"][0]
    Z = -CTX["BOX_CENTER"][1]



    if relative_position:
        Y = last_lat
        Z = -last_lon

    
    if relative_track:
        R = last_track

    if random_track:
        R = np.random.randint(0, 360)


    # Normalize lat lon to 0, 0
    # Convert lat lon to cartesian coordinates
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))

    # Normalize longitude with Z rotation
    r = np.radians(Z)
    x, y, z = Zrotation(x, y, z, r)

    # Normalize latitude with Y rotation
    r = np.radians(Y)
    x, y, z = Yrotation(x, y, z, r)

    # Rotate the fragment with the random angle along X axis
    r = np.radians(R)
    x, y, z = Xrotation(x, y, z, r)

    # convert back cartesian to lat lon
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    # rotate track as well
    track = track - R
    track = np.remainder(track, 360)


    flight[non_zeros_lat_lon, FEATURE_MAP["latitude"]] = lat[non_zeros_lat_lon]
    flight[non_zeros_lat_lon, FEATURE_MAP["longitude"]] = lon[non_zeros_lat_lon]

    # fill empty lat lon with the first non zero lat lon
    first_non_zero_ts = np.argmax(non_zeros_lat_lon)     # (argmax return the first max value)
    first_lat = lat[first_non_zero_ts]
    first_lon = lon[first_non_zero_ts]
    flight[~non_zeros_lat_lon, FEATURE_MAP["latitude"]] = first_lat
    flight[~non_zeros_lat_lon, FEATURE_MAP["longitude"]] = first_lon

    flight[:, FEATURE_MAP["track"]] = track


    if ("timestamp" in FEATURE_MAP):
        timestamp = flight[:, FEATURE_MAP["timestamp"]]
        timestamp = timestamp - timestamp[-2]
        flight[:, FEATURE_MAP["timestamp"]] = timestamp


    return flight

def lat_lon_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in m
    R = 6373.0 * 1000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    distance = 2 * R * np.arcsin(np.sqrt(a))

    return distance

def distance(CTX, y, y_):
    lat1 = y[:, CTX["PRED_FEATURE_MAP"]["latitude"]]
    lon1 = y[:, CTX["PRED_FEATURE_MAP"]["longitude"]]
    lat2 = y_[:, CTX["PRED_FEATURE_MAP"]["latitude"]]
    lon2 = y_[:, CTX["PRED_FEATURE_MAP"]["longitude"]]
    distances = lat_lon_distance(lat1, lon1, lat2, lon2)

    return np.mean(distances)
