import pandas as pd
import os
import math


files = os.listdir('./csv/')

# compute distance based on lat, lon
def lat_lon_dist_m(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2.0)**2 + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    return distance


# group distances by km
distances = {}
distances_files = {}

for i in range(len(files)):
    file = files[i]

    print("\r", i, "/", len(files), end="")

    df = pd.read_csv('./csv/' + file, dtype={'icao24': str})

    # compute the whole distance
    lat = df['latitude'].values
    lon = df['longitude'].values

    distance = 0
    for i in range(1, len(df)):
        distance += lat_lon_dist_m(lat[i], lon[i], lat[i-1], lon[i-1])

    if distance < 1000:
        distance = math.floor(distance / 100) * 100 / 1000
    else:
        distance = math.floor(distance / 1000)
    distances[distance] = distances.get(distance, 0) + 1
    if distance not in distances_files:
        distances_files[distance] = []
    distances_files[distance].append(file)


# sort by key
distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[0])}
distances_files = {k: v for k, v in sorted(distances_files.items(), key=lambda item: item[0])}

print()
import matplotlib.pyplot as plt
# plot the concentrations for each distance
plt.bar(distances.keys(), distances.values(), color='g', width=1.0)
plt.xlabel("Distance (km)")
plt.ylabel("Number of flights")
plt.title("Number of flights for each distance")
plt.show()

files_under_1km = []
for distance in distances_files:
    if distance < 1:
        files_under_1km += distances_files[distance]
    else:
        break



# write all files
save = open("./_Artefact/distances.txt", "w")
for distance in distances_files:
    if distance > 100:
        save.write(str(distance) + "km\n")
        for file in distances_files[distance]:
            save.write("\t" + file + "\n")
save.close()

