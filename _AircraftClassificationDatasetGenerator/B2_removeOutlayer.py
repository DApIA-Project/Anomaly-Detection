import pandas as pd
import os
import math

from A_parquet2csv import flight_is_valid, FOLDER


def angle_diff(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)

def lat_lon_dist_m(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2.0)**2 + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    return distance


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


    # compute the total length of the flight
    lat = df['latitude'].values
    lon = df['longitude'].values
    distance = 0
    for i in range(1, len(df)):
        distance += lat_lon_dist_m(lat[i], lon[i], lat[i-1], lon[i-1])

    if distance < 20 * 1000: # less than 20km, flith to short to be used !
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