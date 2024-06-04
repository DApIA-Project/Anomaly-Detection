from _Utils.numpy import np, ax
from D_DataLoader.Utils import normalize_trajectory, latlondistance
import _Utils.geographic_maths as GEO




def getRandomFlight(CTX, x_train, y_train, x_test, y_test):
    known = bool(np.random.randint(0, 2))

    if (known):
        f = np.random.randint(len(x_train))
        return x_train[f], y_train[f], known
    else:
        f = np.random.randint(len(x_test))
        return x_test[f], "Unknown-flight", known



def rotate(lon, lat, angle):
        return np.cos(angle) * lon - np.sin(angle) * lat, np.sin(angle) * lon + np.cos(angle) * lat

def alter(x, CTX):
        # Get the index of each feature by name for readability
    FEATURE_MAP = CTX["FEATURE_MAP"]
    lat = x[:, FEATURE_MAP["latitude"]]
    lon = x[:, FEATURE_MAP["longitude"]]
    track = x[:, FEATURE_MAP["track"]]

    i = np.random.choice(["rotate", "translate", "scale", "flip", "noise", "trim"])#, "drop"])

    if (i == "rotate"):
        angle = np.random.uniform(-np.pi, np.pi)
        lat, lon = rotate(lat, lon, angle)
        x[:, FEATURE_MAP["latitude"]] = lat
        x[:, FEATURE_MAP["longitude"]] = lon
        return x, "rotate"
    
    elif (i == "translate"):
        lat += np.random.uniform(-0.1, 0.1)
        lon += np.random.uniform(-0.1, 0.1)
        x[:, FEATURE_MAP["latitude"]] = lat
        x[:, FEATURE_MAP["longitude"]] = lon
        return x, "translate"
    
    elif (i == "scale"):
        slat, slon = np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)
        clat, clon = np.mean(lat), np.mean(lon)
        lat = (lat - clat) * slat + clat
        lon = (lon - clon) * slon + clon
        x[:, FEATURE_MAP["latitude"]] = lat
        x[:, FEATURE_MAP["longitude"]] = lon
        return x, "scale"
    
    elif (i == "flip"):
        slat, slon = np.random.choice(["ab", "ba"])
        slat, slon = (1, -1) if slat == "ab" else (-1, 1)
        lat = (lat) * slat 
        lon = (lon) * slon
        x[:, FEATURE_MAP["latitude"]] = lat
        x[:, FEATURE_MAP["longitude"]] = lon
        return x, "flip"

    
    elif (i == "noise"):
        lat += np.random.uniform(-0.000005, 0.000005, size=(len(lat),))
        lon += np.random.uniform(-0.000005, 0.000005, size=(len(lon),))
        x[:, FEATURE_MAP["latitude"]] = lat
        x[:, FEATURE_MAP["longitude"]] = lon
        return x, "noise"
    
    elif (i == "trim"):
        length = np.random.randint(CTX["HISTORY"] * 5, CTX["HISTORY"] * 10)
        if (length > len(lat)):
            length = len(lat)
        start = np.random.randint(0, len(lat) - length)
        return x[start:start+length], "trim"
    
    # elif (i == "drop"):
    #     level = np.random.randint(50, 90)
    #     p = level / 100
    #     indexs = np.arange(len(lat))
    #     indexs = indexs[np.random.uniform(0, 1, size=(len(indexs),)) > p]
    #     x = x[indexs]
    #     return x, "drop" + str(level)


