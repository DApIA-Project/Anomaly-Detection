import numpy as np
import math




def batchPreProcess(CTX, flight):
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
    lat = flight[:, FEATURE_MAP["lat"]]
    lon = flight[:, FEATURE_MAP["lon"]]
    heading = flight[:, FEATURE_MAP["heading"]]
    baro_altitude = flight[:, FEATURE_MAP["baroaltitude"]]
    geo_altitude = flight[:, FEATURE_MAP["geoaltitude"]]



    # do not change angle, and rotate the whole bounding box to 0, 0 (not relative just normalizing)
    R = 0
    Y = CTX["BOX_CENTER"][0]
    Z = -CTX["BOX_CENTER"][1]

    if CTX["RELATIVE_POSITION"]:
        # R = heading[-1]
        Y = lat[-1]
        Z = -lon[-1]

    
    if CTX["RELATIVE_HEADING"]:
        R = heading[-1]

    if (CTX["RANDOM_HEADING"]):
        R = np.random.uniform(0, 360)

    # Normalize lat lon to 0, 0
    # Convert lat lon to cartesian coordinates
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))

    # Normalize longitude with Z rotation
    r = np.radians(Z)
    xz = x * np.cos(r) - y * np.sin(r)
    yz = x * np.sin(r) + y * np.cos(r)
    zz = z

    # Normalize latitude with Y rotation
    r = np.radians(Y)
    xy = xz * np.cos(r) + zz * np.sin(r)
    yy = yz
    zy = -xz * np.sin(r) + zz * np.cos(r)

    # Rotate the fragment with the random angle along X axis
    r = np.radians(R)
    xx = xy
    yx = yy * np.cos(r) - zy * np.sin(r)
    zx = yy * np.sin(r) + zy * np.cos(r)

    # convert back cartesian to lat lon
    lat = np.degrees(np.arcsin(zx))
    lon = np.degrees(np.arctan2(yx, xx))

    # rotate heading as well
    heading = heading - R
    heading = np.remainder(heading, 360)

    # print(lat[-10:], lon[-10:], heading[-10:], sep="\n\n")
    # exit(0)
    

    # Normalise altitude 
    # peharps not a good idea because vertrate is already 
    # kind of normalized
    # baro_altitude = baro_altitude - baro_altitude[-1]
    # geo_altitude = geo_altitude - geo_altitude[-1]
    
    flight[:, FEATURE_MAP["lat"]] = lat
    flight[:, FEATURE_MAP["lon"]] = lon
    flight[:, FEATURE_MAP["heading"]] = heading
    # To imlement and test : Heading 180
    # Add header 180 to remove the gap between 0 and 360
    # of the original heading feature.
    # flight[:, FEATURE_MAP["heading180"]] = heading180
    flight[:, FEATURE_MAP["baroaltitude"]] = baro_altitude
    flight[:, FEATURE_MAP["geoaltitude"]] = geo_altitude
    

    return flight
    



def add_noise(flight, label, noise, noised_label_min=0.5):
    """Add same noise to the x and y to reduce bias but 
    mostly to have a more linear output probabilites meaning :
    - avoid to have 1 or 0 prediction but more something
    like 0.3 or 0.7.
    """

    if (noise > 1.0):
        print("ERROR: noise must be between 0 and 1")
        # throw error
        raise ValueError
    if (noise <= 0):
        return flight, label

    noise_strength = np.random.uniform(0, noise)

    flight_noise = np.random.uniform(-noise_strength, noise_strength, size=flight.shape)
    flight = flight + flight_noise


    effective_strength = noise_strength / noise
    label = label * (1 - effective_strength * (1-noised_label_min))

    return flight, label



"""
UTILITARY FUNCTION FOR MAP PROJECTION
Compute the pixel coordinates of a given lat lon into map.png
"""
def deg2num_int(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)