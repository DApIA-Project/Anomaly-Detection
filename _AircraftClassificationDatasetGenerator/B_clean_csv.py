import pandas as pd
import os
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

def angle_diff(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


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





# remove unusable files
if (not os.path.exists('./B_csv_unusable/'+FOLDER)):
    os.mkdir('./B_csv_unusable/'+FOLDER)

files = os.listdir('./B_csv/'+FOLDER)
files = [file for file in files if file.endswith('.csv')]

for i in range(len(files)):
    file = files[i]
    df = pd.read_csv('./B_csv/'+FOLDER+"/"+ file, dtype={'icao24': str})
    if (not flight_is_valid(df)):
        os.rename('./B_csv/'+FOLDER+"/"+ file, './B_csv_unusable/'+FOLDER+"/"+ file)
        print("unusable INVALID -" + file)
        continue


    # compute the length of the flight
    lat = df['latitude'].values
    lon = df['longitude'].values
    distance = 0
    for i in range(1, len(df)):
        distance += lat_lon_dist_m(lat[i], lon[i], lat[i-1], lon[i-1])

    if distance < 20 * 1000: # less than 20km
        os.rename('./B_csv/'+FOLDER+"/"+ file, './B_csv_unusable/'+FOLDER+"/"+ file)
        print("unusable DISTANCE<20km : " + file)
        continue


    # count the number of unique lat, lon pair
    lats = df['latitude'].values
    lons = df['longitude'].values
    lats_lons = set()
    for i in range(len(lats)):
        lats_lons.add(str(lats[i]) +"_"+ str(lons[i]))

    if (float(len(lats_lons)) / float(len(lats)) * 100 < 10):
        os.rename('./B_csv/'+FOLDER+"/"+ file, './B_csv_unusable/'+FOLDER+"/"+ file)
        print("unusable TOO MANY POS DUPLICATES: " + file)
        continue


    # check if mean track is logic
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
    
    if (mean_angle_diff > 18):
        os.rename('./B_csv/'+FOLDER+"/"+ file, './B_csv_unusable/'+FOLDER+"/"+ file)
        print("unusable MEAN TRACK > 18 : " + file)
        continue




print("Done")
# ############################################################################################################
# ############################################################################################################
# # UNUSABLE FILES !
# import pandas as pd
# import os
# import math

# # check if folder csv_unusable exists
# if (not os.path.exists('./csv_unusable/')):
#     os.mkdir('./csv_unusable/') 


# def angle_diff(a, b):
#     a = a % 360
#     b = b % 360
#     return 180 - abs(abs(a - b) - 180)

# SPLIT_FLIGHT_GAP = 5*60 # 5  minutes
# MIN_DURATION = 10*60 # 15 minutes
# MAX_MISSING_PERCENT = 40 # 40%

# files = os.listdir('./csv/')
# files = [file for file in files if file.endswith('.csv')]

# for i in range(len(files)):

#     print("\rcleaning " + str(i) + " / " + str(len(files)), end="")

#     file = files[i]

#     df = pd.read_csv('./csv/' + file, dtype={'icao24': str, 'callsign': str})

#     if (len(df) < 2):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue

  
#     timestep = df['timestamp'].values
#     timestep_start = timestep[0]
#     timestep_end = timestep[-1]
#     lat = df["latitude"].values
#     lon = df["longitude"].values



#     duration = timestep_end - timestep_start

#     # check if track is always nan
#     if (df['track'].isnull().all()):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue

#     if (df["altitude"].isnull().all() and df["geoaltitude"].isnull().all()):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue

#     if (df["altitude"].isnull().all()):
#         df["altitude"] = df["geoaltitude"]

#     if (df["geoaltitude"].isnull().all()):
#         df["geoaltitude"] = df["altitude"]

#     if duration < MIN_DURATION:
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue


#     max_gap = 0
#     for i in range(1, len(df)):
#         gap = timestep[i] - timestep[i-1]
#         if gap > max_gap:
#             max_gap = gap

#     if (max_gap / duration * 100.0 > 20.0):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue

#     missing_percent = 100.0 - len(df) / duration * 100
#     if (missing_percent > MAX_MISSING_PERCENT):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue

#     lat_lon_angle = []
#     for i in range(1, len(lat)):
#         if (lat[i] == lat[i-1] and lon[i] == lon[i-1]):
#             continue
#         lat_lon_angle.append(
#             math.atan2(
#                 lat[i] - lat[i-1],
#                 lon[i] - lon[i-1]) * 180.0 / math.pi)
        
#     mean_angle_diff = 0
#     for i in range(1, len(lat_lon_angle)):
#         mean_angle_diff += angle_diff(lat_lon_angle[i], lat_lon_angle[i-1])
#     mean_angle_diff /= len(lat_lon_angle)


#     vertrate = df["vertical_rate"].values
#     vertrate_range = max(vertrate) - min(vertrate)
#     if (vertrate_range == 0):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue
    
    
#     if (mean_angle_diff > 50):
#         os.rename('./csv/' + file, './csv_unusable/' + file)
#         continue
        
    

# print()
# print(len(os.listdir('./csv/')))




