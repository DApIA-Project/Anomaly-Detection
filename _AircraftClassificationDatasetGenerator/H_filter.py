import os
import pandas as pd
import math

BOUNDING_BOX = [
    (43.11581, 0.72561),
    (44.07449, 2.16344)
]

# distance to the border of the bounding box
def lat_lon_distance_km(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of the earth in km
    dLat = (lat2-lat1) * math.pi / 180  # deg2rad below
    dLon = (lon2-lon1) * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# distance to the border of the bounding box
def lat_lon_distance(lat, lon):
    return min(
        lat_lon_distance_km(lat, lon, lat, BOUNDING_BOX[0][1]),
        lat_lon_distance_km(lat, lon, lat, BOUNDING_BOX[1][1]),
        lat_lon_distance_km(lat, lon, BOUNDING_BOX[0][0], lon),
        lat_lon_distance_km(lat, lon, BOUNDING_BOX[1][0], lon)
    )

for folder in ["Train", "Eval"]:
    print("Processing", folder)

    # restore previously filtered files
    if (os.path.exists('./dataset/removed_from_'+folder+'/')):
        
        ignored_files = os.listdir('./dataset/removed_from_'+folder+'/')
        ignored_files = [file for file in ignored_files if file.endswith('.csv')]

        for f in range(len(ignored_files)):
            file = ignored_files[f]
            os.rename("./dataset/removed_from_"+folder+"/"+file, './dataset/'+folder+"/"+file)

    else:
        os.mkdir('./dataset/removed_from_'+folder+'/')


    files = os.listdir('./dataset/'+folder)
    files = [file for file in files if file.endswith('.csv')]

    # apply some filters
    # remove files without takeoff content
    ignored_files = []
    f = 0
    for f in range(len(files)):
        if (f % 100 == 0):
            print("\r", f, "/", len(files), end="")

        file = files[f]

        df = pd.read_csv('./dataset/'+folder+"/" + file, dtype={'icao24': str})

        altitude = df['altitude'].values
        geoaltitude = df['geoaltitude'].values
        lat = df['latitude'].values
        lon = df['longitude'].values

        # if (altitude[0] > 2000 or geoaltitude[0] > 2000):
        #     ignored_files.append(file)
        if (lat_lon_distance(lat[0], lon[0]) < 10 and lat_lon_distance(lat[-1], lon[-1]) < 10):
            ignored_files.append(file)


    for f in range(len(ignored_files)):
        file = ignored_files[f]
        os.rename('./dataset/'+folder+"/" + file, './dataset/removed_from_'+folder+"/" + file)

    print("Ignored files: ", len(ignored_files),"/", len(files))