


import pandas as pd
import numpy as np
import os

def rotate(lats, lons, Olat, Olon, angle):
    # angle in degrees
    angle = np.radians(angle)

    new_lats = lats - Olat
    new_lons = lons - Olon
    
    new_lats_rot = new_lats * np.cos(angle) - new_lons * np.sin(angle)
    new_lons_rot = new_lats * np.sin(angle) + new_lons * np.cos(angle)


    return new_lats_rot + Olat, new_lons_rot + Olon


df = pd.read_csv('./2022-01-11_19-43-26_SAMU31_39ac45.csv',  dtype={"icao24":str, "callsign":str})

FLOOD_AFTER = 60
OUT = "./Eval/test"

if not(os.path.exists(OUT)):
    os.makedirs(OUT)



# find i value 
i = 0
while df["timestamp"].values[i] - df["timestamp"].values[0] <= FLOOD_AFTER:
    i += 1

O_lat = df['latitude'].values[i]
O_long = df['longitude'].values[i]

for deriv in [-100, -70, -50, -30, -15, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 50, 70, 100]:

    lats, lons = rotate(df['latitude'].values[i:].copy(), df['longitude'].values[i:].copy(), O_lat, O_long, deriv)

    sub_df = df.copy()
    sub_df['latitude'][i:] = lats
    sub_df['longitude'][i:] = lons
    sub_df['track'][i:] += deriv

    while sub_df['track'][i] < 0:
        sub_df['track'][i:] += 360
    while sub_df['track'][i] >= 360:
        sub_df['track'][i:] -= 360
    if (deriv != 0):
        sub_df["icao24"] = str(deriv)

    sub_df.to_csv(f'{OUT}/rot{deriv}.csv', index=False)

