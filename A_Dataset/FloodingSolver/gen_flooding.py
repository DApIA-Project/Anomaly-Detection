


import pandas as pd
import numpy as np
import os

def x_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x, y * np.cos(-a) - z * np.sin(-a), y * np.sin(-a) + z * np.cos(-a)

def y_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x * np.cos(-a) + z * np.sin(-a), y, -x * np.sin(-a) + z * np.cos(-a)

def z_rotation(x:float, y:float, z:float, a:float) -> "tuple[float, float, float]":
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z

def spherical_to_cartesian(lat:float, lon:float) -> "tuple[float, float, float]":
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))
    return x, y, z

def cartesian_to_spherical(x:float, y:float, z:float) -> "tuple[float, float]":
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def rotate_geo(lats, lons, Olat, Olon, angle):
    # angle in degrees
    angle = np.radians(angle)

    new_lats = lats - Olat
    new_lons = lons - Olon

    new_lats_rot = new_lats * np.cos(angle) - new_lons * np.sin(angle)
    new_lons_rot = new_lats * np.sin(angle) + new_lons * np.cos(angle)

    return new_lats_rot + Olat, new_lons_rot + Olon

def rotate_sphe(lats, lons, Olat:float, Olon:float, angle:float, scale=1.0):


    LAT = -Olat
    LON = -Olon
    ROT = angle

    x, y, z = spherical_to_cartesian(lats, lons)
    # Normalize longitude with Z rotation
    x, y, z = z_rotation(x, y, z, np.radians(LON))
    # Normalize latitude with Y rotation
    x, y, z = y_rotation(x, y, z, np.radians(LAT))
    
    
    # Rotate the fragment with the random angle along X axis
    x, y, z = x_rotation(x, y, z, np.radians(ROT))
    
    # Denormalize latitude with Y rotation
    x, y, z = y_rotation(x, y, z, np.radians(-LAT))
    # Denormalize longitude with Z rotation
    x, y, z = z_rotation(x, y, z, np.radians(-LON))
    # Convert the fragment back to spherical coordinates
    lats, lons = cartesian_to_spherical(x, y, z)
    
    lats = (lats - Olat) * scale + Olat
    lons = (lons - Olon) * scale + Olon

    return lats, lons,


def lat_lon_dist_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) * np.sin(delta_lambda / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def can_flood(df:pd.DataFrame, at:int):
    lat = df['latitude'].values
    lon = df['longitude'].values
    
    for t in range(at - 8, at + 8):
        if t < 0 or t >= len(df):
            return False
        
        if (t > at - 8):
            d = lat_lon_dist_m(lat[t-1], lon[t-1], lat[t], lon[t])
            if (d < 4):
                return False
            if (d > 400):
                return False
            
    return True
            
    

def flood(df, after=60):
    
    
    # find i value
    i = 0
    while df["timestamp"].values[i] - df["timestamp"].values[0] <= after:
        i += 1
        
    # check that the condition are good
    while i < len(df) and not(can_flood(df, i)):
        i += 1
        
    if i == len(df):
        return {}
    

    O_lat = df['latitude'].values[i]
    O_long = df['longitude'].values[i]

    # concat = pd.DataFrame()
    
    res = {}
    
    # for deriv in [-25, -16, -11, -7, -4, -2, -1, 0, 1, 2, 4, 7, 11, 16, 25]:
    for _ in range(15):
        deriv = np.random.rand() * 50 - 25
        # if (geo):
        #     lats, lons = rotate_geo (df['latitude'].values[i:].copy(), df['longitude'].values[i:].copy(), O_lat, O_long, deriv)
        # else:
        lats, lons = rotate_sphe(df['latitude'].values[i:].copy(), df['longitude'].values[i:].copy(), O_lat, O_long, deriv, np.random.rand() * 0.2 + 0.9)

        sub_df = df.copy()
        sub_df['latitude'][i:] = lats
        sub_df['longitude'][i:] = lons
        sub_df['track'][i:] += deriv

        if (deriv != 0):
            sub_df["icao24"] = str(deriv)

        res["rot"+str(deriv)] = sub_df


    return res


files = os.listdir("../AircraftClassification/Eval")
# shuffle the files
np.random.shuffle(files)

OUT = "./Test"
# delete every directory starting with EVAL
os.system(f"rm -rf {OUT}/EVAL*")

i = 0
f = 0
while i < 30:
    df = pd.read_csv(f'../AircraftClassification/Eval/{files[f]}',  dtype={"icao24":str, "callsign":str})
    res = flood(df, after=120)
    if (len(res) == 0):
        print(f"Could not flood {files[f]}")
        f += 1
    else:
        name = files[f].split(".")[0]
        os.makedirs(f'{OUT}/EVAL_{name}', exist_ok=True)
        for key in res:
            res[key].to_csv(f'{OUT}/EVAL_{name}/{key}.csv', index=False)
            
        f += 1
        i += 1

# df = pd.read_csv('./2022-01-11_19-43-26_SAMU31_39ac45.csv',  dtype={"icao24":str, "callsign":str})

# # "rdm" | "geo" | "sphe"
# MODE = "geo"
# OUT = "./Eval/exp"

# if not(os.path.exists(f'{OUT}_{MODE}')):
#     os.makedirs(f'{OUT}_{MODE}')
# else:
#     os.system(f'rm -rf {OUT}_{MODE}/*')




# if (MODE == "rdm"):

#     for random in [0, 1, 2, 3, 5, 10, 20, 40, 80]:

#         lats = df['latitude'].values[i:].copy()
#         lats = lats + np.random.uniform(-random/10000, random/10000, lats.shape)
#         lons = df['longitude'].values[i:].copy()
#         lons = lons + np.random.uniform(-random/10000, random/10000, lons.shape)

#         sub_df = df.copy()
#         sub_df['latitude'][i:] = lats
#         sub_df['longitude'][i:] = lons
#         if (random != 0):
#             sub_df["icao24"] = str(random)
#         sub_df.to_csv(f'{OUT}_{MODE}/rdm{random}.csv', index=False)
# else:

    # for deriv in [-90, -65, -45, -30, -15, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10, 15, 30, 45, 65, 90]:
    #     if (MODE == "geo"):
    #         lats, lons = rotate_geo (df['latitude'].values[i:].copy(), df['longitude'].values[i:].copy(), O_lat, O_long, deriv)
    #     elif (MODE == "sphe"):
    #         lats, lons = rotate_sphe(df['latitude'].values[i:].copy(), df['longitude'].values[i:].copy(), O_lat, O_long, deriv)

    #     sub_df = df.copy()
    #     sub_df['latitude'][i:] = lats
    #     sub_df['longitude'][i:] = lons
    #     sub_df['track'][i:] += deriv

    #     while sub_df['track'][i] < 0:
    #         sub_df['track'][i:] += 360
    #     while sub_df['track'][i] >= 360:
    #         sub_df['track'][i:] -= 360
    #     if (deriv != 0):
    #         sub_df["icao24"] = str(deriv)

    #     sub_df.to_csv(f'{OUT}_{MODE}/rot{deriv}.csv', index=False)

    #     sub_df['icao24'] = df["icao24"]
    #     concat = pd.concat([concat, sub_df])

# order by timestamp
# concat.to_csv(f'./{OUT}_{MODE}/concat.csv', index=False)