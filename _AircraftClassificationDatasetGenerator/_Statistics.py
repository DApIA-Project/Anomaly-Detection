import pandas as pd
import os
import math
from _Utils.numpy import np, ax

FOLDER = "ToulouseV2"


DISTANCE = "DISTANCE"
"MIN DISTANCE SHOULD BE =20km"
GROUND_SPEED_MAX = "GROUND_SPEED_MAX"
GROUND_SPEED = "GROUND_SPEED"
TRACK = "TRACK"
VERTRATE = "VERTRATE"
ALTITUDE = "ALTITUDE"


METRIC = VERTRATE


if METRIC == DISTANCE:
    x_label = "Distance (km)"
    y_label = "Number of flights"
    title = "Number of flights"

if METRIC == VERTRATE:
    x_label = "Vertical rate maxs"
    y_label = "Number of flights"
    title = "Number of flights"

if METRIC == TRACK:
    x_label = "Mean angle shift between timesteps"
    y_label = "Number of flights"
    title = "Number of flights"

if METRIC == ALTITUDE:
    x_label = "Altitude (m)"
    y_label = "Number of flights"
    title = "Number of flights"

if METRIC == GROUND_SPEED_MAX:
    x_label = "Max Ground speed (m/s)"
    y_label = "Number of flights"
    title = "Number of flights"

if METRIC == GROUND_SPEED:
    x_label = "Ground speed (m/s)"
    y_label = "Number of flights"
    title = "Number of flights"



files = os.listdir('./B_csv/'+FOLDER)
files = [file for file in files if file.endswith('.csv')]




def angle_diff(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)






LABEL_NAMES = [
    "UNKNOWN",
    "CARGO",
    "PLANE",
    "JET",
    "ATR",
    "MEDIUM",
    "LIGHT",
    "SUPER LIGHT",
    "GLIDER",
    "HELICOPTER",
    "ULM",
    "MILITARY",
    "SAMU"
]

COLORS = [
    "tab:gray", # unknown
    "darkgreen", # cargo
    "tab:green", # plane
    "tab:pink", # jet
    "beige", # ATR
    "tab:purple", # medium
    "tab:blue", # Light
    "tab:red", # super light
    "tab:gray", # glider
    "yellow", # helicopter
    "tab:cyan", # ULM
    "tab:olive", # military
    "black", # samu
]



labels = pd.read_csv('./labels/labels.csv', header=None)
labels = labels.set_index(0)
labels = {k: v for k, v in labels[1].items()}

labels_bincount = np.zeros(len(LABEL_NAMES))
for i in range(len(files)):
    file = files[i]
    file = file.split(".")[0]
    icao24 = file.split("_")[3]
    if icao24 in labels:
        label = labels[icao24]
        labels_bincount[label] += 1
labels_order = np.argsort(labels_bincount)[::-1]


metrics = {}
metric_files = {}
metric_labels = {}

for i in range(len(files)):
    file = files[i]

    print("\r", i, "/", len(files), end="")

    df = pd.read_csv('./B_csv/'+FOLDER+"/" + file, dtype={'icao24': str})

    icao = df['icao24'].values[0]
    if icao not in labels:
        label = 0
    else:
        label = labels[icao]

    lat = df['latitude'].values
    lon = df['longitude'].values


    metric = None

    # compute the whole distance
    if METRIC == DISTANCE:

        distance = 0
        for i in range(1, len(df)):
            distance += lat_lon_dist_m(lat[i], lon[i], lat[i-1], lon[i-1])

        if distance < 1000:
            distance = math.floor(distance / 100) * 100 / 1000
        else:
            distance = math.floor(distance / 1000)

        metric = distance

    if METRIC == VERTRATE:
        if not((df['vertical_rate'].values == np.nan).all()):
            # vertrate = np.nanmax(df['vertical_rate'].values) - np.nanmin(df['vertical_rate'].values)
            # replace nan by 0
            df['vertical_rate'] = df['vertical_rate'].fillna(0)
            vertrate = max(np.nanmax(df['vertical_rate'].values), -np.nanmin(df['vertical_rate'].values))
            # vertrate =

            metric = vertrate



    if METRIC == TRACK:
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
        mean_angle_diff = round(mean_angle_diff, 1)

        metric = mean_angle_diff

    if METRIC == ALTITUDE:
        if not((df['altitude'].values == np.nan).all()):
            altitude = round(np.nanmean(df['altitude'].values) / 25) * 25
            metric = altitude

    if METRIC == GROUND_SPEED_MAX:
        if not((df['groundspeed'].values == np.nan).all()):
            df['groundspeed'] = df['groundspeed'].fillna(0)
            altitude = round(np.nanmax(df['groundspeed'].values))
            metric = altitude

    if METRIC == GROUND_SPEED:
        if not((df['groundspeed'].values == np.nan).all()):
            df['groundspeed'] = df['groundspeed'].fillna(0)
            altitude = round(np.nanmean(df['groundspeed'].values))
            metric = altitude

    ####################
    if (metric == None):
        continue

    metrics[metric] = metrics.get(metric, 0) + 1
    if metric not in metric_files:
        metric_files[metric] = []
    metric_files[metric].append(file)
    if metric not in metric_labels:
        metric_labels[metric] = []
    metric_labels[metric].append(label)

# sort by key
metrics = {k: v for k, v in sorted(metrics.items(), key=lambda item: item[0])}
metric_files = {k: v for k, v in sorted(metric_files.items(), key=lambda item: item[0])}

print()
import matplotlib.pyplot as plt
# plot the concentrations for each distance


keys = list(metrics.keys())
max_bar_height = max(metrics.values())

for b in range(len(keys)):
    # plot rectangles of color for each label
    bar_labels = metric_labels[keys[b]]
    NB_BARS = int(max_bar_height / 6.0)
    labels_nb = np.bincount(bar_labels)
    labels_perc = labels_nb / sum(labels_nb)
    total_height = sum(labels_nb)
    sub_bar_height = total_height / NB_BARS
    actual_height = 0
    for _ in range(NB_BARS):
        combination = np.arange(len(labels_nb))
        np.random.shuffle(combination)
        for i in range(len(labels_nb)):
            l = combination[i]
            h = sub_bar_height * labels_perc[l]
            if (h == 0):
                continue
            plt.bar(b, h, color=COLORS[l], bottom=actual_height, width=1.0)
            actual_height += h

keys = [str(k) for k in keys]

# plt.bar(keys, metric.values(), color='g', width=1.0)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
for l in labels_order:
    plt.bar(0, 0, color=COLORS[l], label=LABEL_NAMES[l])
plt.legend(loc = 'upper right')
NB_X_TICKS = 10
plt.xticks(range(0, len(keys), len(keys)//NB_X_TICKS), keys[::len(keys)//NB_X_TICKS], rotation=45)

# savefig
plt.savefig("./_Artifacts/"+METRIC+".png", dpi=300)
plt.show()





# write all files
save = open("./_Artifacts/"+METRIC+".txt", "w")
for distance in metric_files:
    save.write(str(distance)+"\n")
    for file in metric_files[distance]:
        save.write("\t" + file + "\n")
save.close()

