 
import pandas as pd
import math
import os
from datetime import datetime
import time
import random
import numpy as np

# read parquet file

MIN_DURATION = 15*60 # 15 minutes
SPLIT_FLIGHT_GAP = 5*60 # 5 minutes

minLat = 43.11581
maxLat = 44.07449
minLon = 0.72561
maxLon = 2.16344

def lat_lon_dist_m(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = lat1 * math.pi / 180.0
    lon1 = lon1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    lon2 = lon2 * math.pi / 180.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c * 1000

    return distance



def flight_is_valid(flight_df):
    if (len(flight_df) < MIN_DURATION):
        return False
    
    flight_lenght = flight_df['timestamp'].iloc[-1] - flight_df['timestamp'].iloc[0]
    if flight_lenght.total_seconds() < MIN_DURATION:
        return False

    # check that at least one timestamp is in the box
    in_box = False
    for i in range(len(flight_df)):
        if (minLat <= flight_df['latitude'].iloc[i] <= maxLat) and (minLon <= flight_df['longitude'].iloc[i] <= maxLon):
            in_box = True
            break

    return in_box


os.system("rm ./csv/*.csv")
files = os.listdir('./parquets')
files = [file for file in files if file.endswith('.parquet')]


file = files[0]
i = 0
for i in range(len(files)):

    file = files[i]
    print(i,"-",file)

    df = pd.read_parquet('./parquets/' + file)

    

    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns=["last_position", "hour"])
    # drop each row with a latitute or longitude null
    df = df.dropna(subset=['latitude', 'longitude'])

    if ("serials" in df.columns):
        df = df.drop(columns=["serials"])

    # remplace timestamp (datetime) by timestamp (int)
    # df['timestamp'] = df['timestamp'].astype('int64') // 10**9
    # remplace each None callsign by "None"
    df['callsign'] = df['callsign'].fillna("None")

    # get all (callsign, icao24) unique pairs
    icaos = df['icao24'].drop_duplicates().values


    # # sort df by icao24 
    df = df.sort_values(by=['icao24', 'timestamp'])
    df = df.reset_index(drop=True)

    # for each (callsign, icao24) pair count the number of lines
    # and get the index of the first line


    # get all lines for each icao24 
    ti = time.time()
    print("\nget messages for each icao24")
    start = 0
    end = 0
    lines = {}
    for end in range(len(df)):
        if (df["icao24"][start] != df["icao24"][end]):
            if (df['icao24'][start] in lines): print("ALERT ! " + df['icao24'][start] + " already in lines : ", lines[df['icao24'][start]], " new : ", (start, end))
            lines[df["icao24"][start]] = (start, end)
            start = end
    lines[df["icao24"][start]] = (start, end)

    df_per_icao = []
    for key in lines:
        print(f"\r{key} - {lines[key][1]-lines[key][0]}", end="")
        sub = df.iloc[lines[key][0]:lines[key][1]]
        sub = sub.reset_index(drop=True)
        df_per_icao.append(sub)
    print()
    print("time: ", time.time()-ti)
    del lines




    flight_df = []
    ti = time.time()
    print("\nsplit in sub flight")
    # split df if there is a gap of 30 minutes
    for i in range(len(df_per_icao)):
        print(f"\r{i}/{len(df_per_icao)}, nb : {len(flight_df)}", "time : ", time.time()-ti, end="")

        split_indexs = [0]
        for t in range(1, len(df_per_icao[i])):
            if df_per_icao[i]['timestamp'][t] - df_per_icao[i]['timestamp'][t-1] > pd.Timedelta(seconds=SPLIT_FLIGHT_GAP):
                split_indexs.append(t)
        split_indexs.append(len(df_per_icao[i]))

        for j in range(len(split_indexs)-1):
            flight = df_per_icao[i][split_indexs[j]:split_indexs[j+1]].reset_index(drop=True)
            
            if flight_is_valid(flight):
                flight_df.append(flight)
    print()

        

    print("\nsave csv")
    # convert units
    # groundspeed is
    ti = time.time()
    for i in range(len(flight_df)):
        # convert datetime to timestamp
        

        print(f"\r{i}/{len(flight_df)}, time : ", time.time()-ti, end="")

        ts = flight_df[i]['timestamp'].iloc[0]
        # .strftime('%Y-%m-%d_%H-%M-%S')
        date_str = str(ts.year) + str(ts.month).zfill(2) + str(ts.day).zfill(2) + '_' + str(ts.hour).zfill(2) + str(ts.minute).zfill(2) + str(ts.second).zfill(2)
        name = date_str + '_' + str(flight_df[i]['callsign'].iloc[0]) + '_' + str(flight_df[i]['icao24'].iloc[0])
        # save dataframe to csv
        flight_df[i].to_csv('./csv/' + name + '.csv', index=False)
    print()
    print()

# 4499