
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
np.set_printoptions(suppress=True)

def Xrotation(x, y, z, r):
    xx = x
    yx = y * np.cos(r) - z * np.sin(r)
    zx = y * np.sin(r) + z * np.cos(r)
    return xx, yx, zx

def Yrotation(x, y, z, r):
    xy = x * np.cos(r) + z * np.sin(r)
    yy = y
    zy = -x * np.sin(r) + z * np.cos(r)
    return xy, yy, zy

def Zrotation(x, y, z, r):
    xz = x * np.cos(r) - y * np.sin(r)
    yz = x * np.sin(r) + y * np.cos(r)
    zz = z
    return xz, yz, zz


def un_rotate(olat, olon, otrack, lat, lon):
    Y = olat
    Z = -olon
    R = otrack

    x = math.cos(math.radians(lon)) * math.cos(math.radians(lat))
    y = math.sin(math.radians(lon)) * math.cos(math.radians(lat))
    z =                           math.sin(math.radians(lat))

    r = math.radians(-R)
    x, y, z = Xrotation(x, y, z, r)

    r = math.radians(-Y)
    x, y, z = Yrotation(x, y, z, r)

    r = math.radians(-Z) 
    x, y, z = Zrotation(x, y, z, r)

    lat = math.degrees(math.asin(z))
    lon = math.degrees(math.atan2(y, x))

    return lat, lon

def _rotate(olat, olon, otrack, lat, lon):
    Y = olat
    Z = -olon
    R = otrack

    x = math.cos(math.radians(lon)) * math.cos(math.radians(lat))
    y = math.sin(math.radians(lon)) * math.cos(math.radians(lat))
    z =                           math.sin(math.radians(lat))

    r = math.radians(Z)
    x, y, z = Zrotation(x, y, z, r)

    r = math.radians(Y)
    x, y, z = Yrotation(x, y, z, r)

    r = math.radians(R)
    x, y, z = Xrotation(x, y, z, r)

    # convert back cartesian to lat lon
    lat = math.degrees(math.asin(z))
    lon = math.degrees(math.atan2(y, x))

    return lat, lon

def latlon_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two points on earth
    """
    R = 6371e3 # metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(lon2-lon1)

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = R * c
    return d




FOLDER = "takeoff"

ref_files = os.listdir("./Eval/"+FOLDER)
pred_files = os.listdir("./Outputs/"+FOLDER)

ref_files.sort(reverse=True)
pred_files.sort(reverse=True)



f = -1
ref0 = pd.read_csv("./Eval/"+FOLDER+"/"+ref_files[f], dtype={"icao24":str, "callsign":str})
#drop first row
ref0 = ref0.drop(ref0.index[0]).reset_index(drop=True)
pred0 = pd.read_csv("./Outputs/"+FOLDER+"/"+pred_files[f])
pred0["timestamp"] += ref0["timestamp"][0]
pred0["distance"] = latlon_distance(pred0["pred_latitude"], pred0["pred_longitude"], pred0["true_latitude"], pred0["true_longitude"])

print(len(pred0), "/", len(ref0))
# remove in ref all unexisting timestamp
# ref0 = ref0[ref0["timestamp"].isin(pred0["timestamp"])].reset_index(drop=True)




# from_to = [30, 80]
from_to = [2, 50]
# from_to = [320, 350]
s = 4
horizon = 4

pred_traj = []
verif_true = []


traj_lat = ref0["latitude"][from_to[0]-s+3:from_to[1]+s+3].values
traj_lon = ref0["longitude"][from_to[0]-s+3:from_to[1]+s+3].values
traj_ts = ref0["timestamp"][from_to[0]-s+3:from_to[1]+s+3].values

for t in range(from_to[0], from_to[1]):

    timestamp = ref0["timestamp"][t]

    pred = pred0[pred0["timestamp"] == timestamp].reset_index(drop=True)
    if len(pred) == 0:
        continue

    o_lat = ref0["latitude"][t-horizon-1]
    o_lon = ref0["longitude"][t-horizon-1]
    track = ref0["track"][t-horizon-1]
    track = 0

    t_lat = ref0["latitude"][t]
    t_lon = ref0["longitude"][t]

    pred_p_lat_ = pred["pred_latitude"].values[0]
    pred_p_lon_ = pred["pred_longitude"].values[0]
    pred_t_lat_ = pred["true_latitude"].values[0]
    pred_t_lon_ = pred["true_longitude"].values[0]

    
    pred_p_lat, pred_p_lon = un_rotate(o_lat, o_lon, track, pred_p_lat_, pred_p_lon_)
    pred_t_lat, pred_t_lon = un_rotate(o_lat, o_lon, track, pred_t_lat_, pred_t_lon_)

    pred_traj.append([pred_p_lat, pred_p_lon])
    verif_true.append([pred_t_lat, pred_t_lon])

pred_traj_lat, pred_traj_lon = np.array(pred_traj).T
verif_true_lat, verif_true_lon = np.array(verif_true).T


plt.plot(traj_lat, traj_lon, 
         linestyle='-', marker='+', markersize=6,
         color="tab:blue", linewidth=2, 
         label="traj")

plt.plot(traj_lat[0], traj_lon[0], "o", markersize=7, label="A", color="tab:purple")
plt.plot(traj_lat[-1], traj_lon[-1], "o", markersize=7, label="B", color="tab:orange")

plt.plot(pred_traj_lat, pred_traj_lon,
            linestyle='', marker='x', markersize=6,
            color="tab:orange", linewidth=2, 
            label="pred")

plt.plot(verif_true_lat, verif_true_lon,
            linestyle='', marker='x', markersize=6,
            color="tab:green", linewidth=1, 
            label="true")

         
# draw dashed seg  between pred and true
for i in range(len(pred_traj_lat)):
    plt.plot([pred_traj_lat[i], verif_true_lat[i]], [pred_traj_lon[i], verif_true_lon[i]], "k--")


plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")
plt.show()











distances = []
values = []

for i in range(len(ref_files)):

    pred = pd.read_csv("./Outputs/"+FOLDER+"/"+ pred_files[i])
    ref = pd.read_csv("./Eval/"+FOLDER+"/"+ ref_files[i], dtype={"icao24":str, "callsign":str})
    

    values.append([])
    for t in range(len(pred)):
        values[i].append([pred["pred_latitude"][t], pred["pred_longitude"][t], pred["true_latitude"][t], pred["true_longitude"][t]])

    distance = latlon_distance(pred["pred_latitude"], pred["pred_longitude"], pred["true_latitude"], pred["true_longitude"])
    distances.append(distance)
        

# values = np.array(values)
# distances = np.array(distances)

# plot
plt.figure(figsize=(20, 10))
for i in range(len(distances)):
    plt.plot(distances[i][:], label=ref_files[i])
# plot y 0 to 100
# plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")
plt.title("Distance between predicted and true position")
plt.show()


mean_distances = []
for i in range(len(distances)):
    mean_distances.append(np.mean(distances[i]))
mean_distances = np.array(mean_distances)

# plot
order = np.argsort(mean_distances)
# bar plot
for i in range(len(mean_distances)):
    filename = ref_files[order[i]]
    plt.bar(i, mean_distances[order[i]], label=ref_files[order[i]])

plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")
plt.title("Mean distance between predicted and true position")
plt.show()


# plot distances[:, 61] in bar and sorted

ts = [8, 30]
dist = np.array([np.mean(distances[i][ts[0]:ts[1]]) for i in range(len(distances))])
order = np.argsort(dist)
# bar plot
for i in range(len(order)):
    plt.bar(i, dist[order[i]], label=ref_files[order[i]])
plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")
plt.title("Distance between predicted and true position on saturation")
plt.show()

