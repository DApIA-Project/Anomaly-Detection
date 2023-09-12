 
import pandas as pd
import math
import os
from datetime import datetime
import time
import random

# read parquet file

SPLIT_FLIGHT_GAP = 5*60 # 5  minutes
MIN_DURATION = 15*60 # 15 minutes
MAX_MISSING_PERCENT = 40 # 40%
MIN_LAT_LON_DIFF = 0.5 # 0.5m

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
    # remove onground rows at the beginning and end
    # remove rows where lat, lon diff is less than 0.5m
    #
    # then check if flight is valid
    # - at least 15 minutes of data
    # - less than 40% of data is missing

    # remove onground rows at the beginning and end
    # todrop = set()

    
    # lat = flight_df['latitude'].values
    # lon = flight_df['longitude'].values

    # for i in range(1, len(flight_df)):
    #     if (lat_lon_dist_m(lat[i], lon[i], lat[i-1], lon[i-1]) < MIN_LAT_LON_DIFF):
    #         todrop.add(i)

    # if (len(todrop) == len(flight_df)):
    #     return False

    # todrop = list(todrop)

    # flight_df.drop(todrop, inplace=True)
    # flight_df.reset_index(drop=True, inplace=True)

    flight_lenght = flight_df['timestamp'].iloc[-1] - flight_df['timestamp'].iloc[0]
    if flight_lenght < MIN_DURATION:
        return False
    
    missing_percent = 100.0 - len(flight_df) / flight_lenght * 100
    if missing_percent > MAX_MISSING_PERCENT:
        return False
    
    return True


os.system("rm ./csv/*.csv")
files = os.listdir('./parquets')


file = files[0]
for i in range(len(files)):
    file = files[i]
    print(i,"-",file)

    df = pd.read_parquet('./parquets/' + file)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns=["last_position","serials", "hour"])
    # remplace timestamp (datetime) by timestamp (int)
    df['timestamp'] = df['timestamp'].astype('int64') // 10**9
    # remplace each None callsign by "None"
    df['callsign'] = df['callsign'].fillna("None")

    # get all (callsign, icao24) unique pairs
    pairs = df[['callsign', 'icao24']].drop_duplicates().values

    lines = {}
    for p in pairs:
        lines[p[0] + p[1]] = []

    last_call_ica = None
    last_call_ica_list = None
    nb = 0
    tot = 0
    ti = time.time()
    for line in df.iterrows():
        # line = line[1]
        c = line[1]
        call_ica = c[7] + c[1]

        if (call_ica != last_call_ica):
            print("\rchange icao to ", call_ica.ljust(15, " "), str(nb).ljust(5, " "), tot,"/",len(df), "time: ", time.time()-ti, end="")
            last_call_ica = call_ica
            last_call_ica_list = lines[call_ica]
            nb = 0

        nb += 1
        tot += 1
        last_call_ica_list.append(line[0])

    sub_df = []
    for key in lines:
        sub = df.iloc[lines[key]]
        sub = sub.sort_values(by=['timestamp'])
        sub = sub.reset_index(drop=True)
        sub_df.append(sub)
    print()
    print("time: ", time.time()-ti)
    del lines





    flight_df = []
    ti = time.time()
    print("\nsplit in sub flight")
    # split df if there is a gap of 30 minutes
    for i in range(len(sub_df)):
        print(f"\r{i}/{len(sub_df)}, nb : {len(flight_df)}", "time : ", time.time()-ti, end="")

        split_indexs = [0]
        for t in range(1, len(sub_df[i])):
            if sub_df[i]['timestamp'][t] - sub_df[i]['timestamp'][t-1] > SPLIT_FLIGHT_GAP:
                split_indexs.append(t)
        split_indexs.append(len(sub_df[i]))

        for j in range(len(split_indexs)-1):
            flight = sub_df[i][split_indexs[j]:split_indexs[j+1]].reset_index(drop=True)
            
            if flight_is_valid(flight):
                flight_df.append(flight)
    print()

        

    print("\nsave csv")
    # convert units
    # groundspeed is
    ti = time.time()
    for i in range(len(flight_df)):
        print(f"\r{i}/{len(flight_df)}, time : ", time.time()-ti, end="")

        # YYYYMMDD_HHMMSS_callsign_icao24.csv
        ts = flight_df[i]['timestamp'].iloc[0]
        date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        name = date_str + '_' + str(flight_df[i]['callsign'].iloc[0]) + '_' + str(flight_df[i]['icao24'].iloc[0])
        # save dataframe to csv
        flight_df[i].to_csv('./csv/' + name + '.csv', index=False)
    print()
    print()

# 4499