import pandas as pd
from _Utils.os_wrapper import os
import math

from A_parquet2csv import flight_is_valid, FOLDER



# compute distance based on lat, lon
def lat_lon_dist_m(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2.0)**2 + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    return distance



files = os.listdir('./B_csv/'+FOLDER)
files = [file for file in files if file.endswith('.csv')]

for i in range(len(files)):

    print("\rcleaning " + str(i) + " / " + str(len(files)), end="")

    file = files[i]

    df = pd.read_csv('./B_csv/'+FOLDER+"/"+ file, dtype={'icao24': str})

    lats = df['latitude'].values
    lons = df['longitude'].values
    lats_lons = []
    lats_lons_i = {}

    for i in range(len(lats)):
        text = str(lats[i]) +"_"+ str(lons[i])
        if (len(lats_lons) == 0):
            lats_lons.append(text)
        elif (lats_lons[-1] != text):
            lats_lons.append(str(lats[i]) +"_"+ str(lons[i]))

        if (text not in lats_lons_i):
            lats_lons_i[text] = []
        lats_lons_i[text].append(i)

    lat_lon_nb = {}
    for lat_lon in lats_lons:
        lat_lon_nb[lat_lon] = lat_lon_nb.get(lat_lon, 0) + 1

    #sort
    lat_lon_nb = {k: v for k, v in sorted(lat_lon_nb.items(), key=lambda item: item[1])}


    # # transform to percentage
    # for lat_lon in lat_lon_nb:
    #     lat_lon_nb[lat_lon] = lat_lon_nb[lat_lon] / len(df) * 100.0        

    # remove all lat_lon_nb > 3%
    # (messages that are abnormally too many times at the same place)
    found = False
    to_remove = []
    for lat_lon in lat_lon_nb:
        if lat_lon_nb[lat_lon] > 10:
            to_remove.append(lat_lon)
            found = True
    if (found):
        print("file " + file + " has " + str(len(to_remove)) + " abnormal points")

    to_remove_i = set()
    for lat_lon in to_remove:
        indexs = lats_lons_i[lat_lon]
        for index in indexs:
            to_remove_i.add(index)


    # drop aberrant vertrate
    # vertrate = df['vertical_rate'].values
    # for i in range(len(vertrate)):
    #     if (vertrate[i] > 4224 or vertrate[i] < -4224):
    #         to_remove_i.add(i)

    # drop timestamp duplicates
    timestamp = df['timestamp'].values
    for i in range(1, len(timestamp)):
        if (timestamp[i] == timestamp[i-1]):
            to_remove_i.add(i)


    # drop too far points
    last_lat, last_lon, last_ts, last_drop = lats[0], lons[0], timestamp[0], False
    for i in range(1, len(lats)):
        if (lats[i] == lats[i-1] and lons[i] == lons[i-1]):
            if (last_drop):
                to_remove_i.add(i)
            continue

        last_drop = False

        d = lat_lon_dist_m(lats[i], lons[i], last_lat, last_lon) / 1000.0
        t = (timestamp[i] - last_ts) / 3600.0
        if (d / t > 1234.8): # speed of sound
            print(d/t)
            to_remove_i.add(i)
            last_drop = True
        else:
            last_lat, last_lon, last_ts = lats[i], lons[i], timestamp[i]

    to_remove_i = list(to_remove_i)
            
    if (len(to_remove_i) > 0):
        df.drop(to_remove_i, inplace=True)
        df.reset_index(drop=True, inplace=True)


    # clean altitude, geoaltitude, vertical_rate to integer if they are almost integer
    altitude = df['altitude'].values
    geoaltitude = df['geoaltitude'].values
    vertical_rate = df['vertical_rate'].values

    for feature in [altitude, geoaltitude, vertical_rate]:
        for t in range(len(feature)):
            f = feature[t]
            if (math.isnan(f)):
                continue
            if (f - math.floor(f) < 0.01):
                feature[t] = math.floor(f)
            elif (math.ceil(f) - f < 0.01):
                feature[t] = math.ceil(f)
    
    df['altitude'] = altitude
    df['geoaltitude'] = geoaltitude
    df['vertical_rate'] = vertical_rate


    # clean vertrate abnormal values
    # remplace > 4800 by nan
    df['vertical_rate'] = df['vertical_rate'].apply(lambda x: x if x < 4800 else math.nan)
    df['vertical_rate'] = df['vertical_rate'].apply(lambda x: x if x > -4800 else math.nan)



    df.to_csv('./B_csv/'+FOLDER+"/" + file, index=False)

print("\nDone")





