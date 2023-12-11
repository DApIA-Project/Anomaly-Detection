import pandas as pd
import os
import math


files = os.listdir('./csv/')
files = [file for file in files if file.endswith('.csv')]

for i in range(len(files)):

    print("\rcleaning " + str(i) + " / " + str(len(files)), end="")

    file = files[i]

    df = pd.read_csv('./csv/' + file, dtype={'icao24': str})

    lats = df['latitude'].values
    lons = df['longitude'].values
    lats_lons = []
    lats_lons_i = {}
    for i in range(len(lats)):
        lats_lons.append(str(lats[i]) +"_"+ str(lons[i]))
        if (lats_lons[-1] not in lats_lons_i):
            lats_lons_i[lats_lons[-1]] = []
        lats_lons_i[lats_lons[-1]].append(i)
            

    lat_lon_nb = {}
    for lat_lon in lats_lons:
        lat_lon_nb[lat_lon] = lat_lon_nb.get(lat_lon, 0) + 1

    #sort
    lat_lon_nb = {k: v for k, v in sorted(lat_lon_nb.items(), key=lambda item: item[1])}


    # transform to percentage
    for lat_lon in lat_lon_nb:
        lat_lon_nb[lat_lon] = lat_lon_nb[lat_lon] / len(df) * 100.0        

    # remove all lat_lon_nb > 3%
    # (messages that are abnormally too many times at the same place)
    to_remove = []
    for lat_lon in lat_lon_nb:
        if lat_lon_nb[lat_lon] > 3:
            to_remove.append(lat_lon)

    to_remove_i = set()
    for lat_lon in to_remove:
        indexs = lats_lons_i[lat_lon]
        for index in indexs:
            to_remove_i.add(index)


    # drop aberrant vertrate
    vertrate = df['vertical_rate'].values
    for i in range(len(vertrate)):
        if (vertrate[i] > 4224 or vertrate[i] < -4224):
            to_remove_i.add(i)

    # drop timestamp duplicates
    timestamp = df['timestamp'].values
    for i in range(1, len(timestamp)):
        if (timestamp[i] == timestamp[i-1]):
            to_remove_i.add(i)

    to_remove_i = list(to_remove_i)
            
    if (len(to_remove_i) > 0):
        df.drop(to_remove_i, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_csv('./csv/' + file, index=False)


print("\nDone")



############################################################################################################
############################################################################################################
# UNUSABLE FILES !
import pandas as pd
import os
import math

# check if folder csv_unusable exists
if (not os.path.exists('./csv_unusable/')):
    os.mkdir('./csv_unusable/') 


def angle_diff(a, b):
    a = a % 360
    b = b % 360
    return 180 - abs(abs(a - b) - 180)

SPLIT_FLIGHT_GAP = 5*60 # 5  minutes
MIN_DURATION = 10*60 # 15 minutes
MAX_MISSING_PERCENT = 40 # 40%

files = os.listdir('./csv/')
files = [file for file in files if file.endswith('.csv')]

for i in range(len(files)):

    print("\rcleaning " + str(i) + " / " + str(len(files)), end="")

    file = files[i]

    df = pd.read_csv('./csv/' + file, dtype={'icao24': str, 'callsign': str})

    if (len(df) < 2):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue

  
    timestep = df['timestamp'].values
    timestep_start = timestep[0]
    timestep_end = timestep[-1]
    lat = df["latitude"].values
    lon = df["longitude"].values



    duration = timestep_end - timestep_start

    # check if track is always nan
    if (df['track'].isnull().all()):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue

    if (df["altitude"].isnull().all() and df["geoaltitude"].isnull().all()):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue

    if (df["altitude"].isnull().all()):
        df["altitude"] = df["geoaltitude"]

    if (df["geoaltitude"].isnull().all()):
        df["geoaltitude"] = df["altitude"]

    if duration < MIN_DURATION:
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue


    max_gap = 0
    for i in range(1, len(df)):
        gap = timestep[i] - timestep[i-1]
        if gap > max_gap:
            max_gap = gap

    if (max_gap / duration * 100.0 > 20.0):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue

    missing_percent = 100.0 - len(df) / duration * 100
    if (missing_percent > MAX_MISSING_PERCENT):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue

    lat_lon_angle = []
    for i in range(1, len(lat)):
        if (lat[i] == lat[i-1] and lon[i] == lon[i-1]):
            continue
        lat_lon_angle.append(
            math.atan2(
                lat[i] - lat[i-1],
                lon[i] - lon[i-1]) * 180.0 / math.pi)
        
    mean_angle_diff = 0
    for i in range(1, len(lat_lon_angle)):
        mean_angle_diff += angle_diff(lat_lon_angle[i], lat_lon_angle[i-1])
    mean_angle_diff /= len(lat_lon_angle)


    vertrate = df["vertical_rate"].values
    vertrate_range = max(vertrate) - min(vertrate)
    if (vertrate_range == 0):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue
    
    
    if (mean_angle_diff > 50):
        os.rename('./csv/' + file, './csv_unusable/' + file)
        continue
        
    

print()
print(len(os.listdir('./csv/')))




