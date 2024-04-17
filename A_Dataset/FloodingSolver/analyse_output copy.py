
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import _Utils.geographic_maths as GEO

# folder = "./Outputs/test/"
folder = "./Outputs/my_one/"
# folder = "./Outputs/realism/"
# folder = "./Outputs/norealism/"
# folder = "./Outputs/takeoff/"
files = os.listdir(folder)
lats = []
dists = []
for file in files:
    df = pd.read_csv(folder+file, dtype={"icao24":str, "callsign":str})

    df_lat = df['df_latitude'].values
    df_long = df['df_longitude'].values
    true_lat = df['true_latitude'].values
    true_long = df['true_longitude'].values
    pred_lat = df['pred_latitude'].values
    pred_long = df['pred_longitude'].values

    lats.append(df_lat)

    # euclidian distance
    dist = GEO.distance_m(true_lat, true_long, pred_lat, pred_long)
    dists.append(dist)


lens = [len(d) for d in dists]
# get the shortest len
min_len = min(lens)
dists = [d[:min_len] for d in dists]
dists = np.array(dists)

lats = [l[:min_len] for l in lats]
lats = np.array(lats)


# sort by mean distance per time step
mean = np.nanmean(dists, axis=0)
# get the loc of the max mean omitting nan




# find where the true lat change between files
i = 0
while lats[0, i]-lats[4, i] <= 2e-4:
    i += 1
# i = 65

max_i = i
start = max_i - 5
end = max_i + 20

distant_on_attack = np.nanmean(dists[:, start:end], axis=1)

means_dist = [[distant_on_attack[i], files[i]] for i in range(len(distant_on_attack))]


# sort means_dist by distance but file as well
order = np.arange(len(means_dist))
order = sorted(order, key=lambda x: means_dist[x][0])

means_dist = [means_dist[i] for i in order]
files = [files[i] for i in order]
files = files[::-1]




# # for file in files:
fig, axs = plt.subplots(3, 1, figsize=(6.5,6.5))
j = 0
for file in ["rot80.csv", "rot30.csv", "rot0.csv"]:

    df = pd.read_csv(folder+file, dtype={"icao24":str, "callsign":str})

    df_lat = df['df_latitude'].values
    df_long = df['df_longitude'].values
    true_lat = df['true_latitude'].values
    true_long = df['true_longitude'].values
    pred_lat = df['pred_latitude'].values
    pred_long = df['pred_longitude'].values

    s = max_i - 20
    win = [s-30, s+35]
    # win = [650, 670]
    # start, end = win

    df_lat_ = df_lat[win[0]:win[1]]
    df_long_ = df_long[win[0]:win[1]]
    true_lat_ = true_lat[win[0]:win[1]]
    true_long_ = true_long[win[0]:win[1]]
    pred_lat_ = pred_lat[win[0]:win[1]]
    pred_long_ = pred_long[win[0]:win[1]]

    # fig, ax = plt.subplots(2, 1, figsize=(16,12), gridspec_kw={'height_ratios': [2, 1]})

    # fig, ax = plt.subplots(1, 1, figsize=(8,12))
    ax = [0, axs[j]]

    # ax[0].plot(df_lat_[0], df_long_[0], 'o', color='tab:green')
    # ax[0].plot(df_lat_, df_long_, '+', color='tab:orange')
    # ax[0].plot(true_lat_, true_long_, 'x-', color='tab:green')
    # ax[0].plot(pred_lat_, pred_long_, 'x', color='tab:blue')
    # for t in range(len(true_lat_)):
    #     ax[0].plot([true_lat_[t], pred_lat_[t]], [true_long_[t], pred_long_[t]], "--", color="black", linewidth=1)
    # ax[0].set_aspect('equal', 'box')


    # # euclidian distance
    dist = GEO.distance_m(true_lat, true_long, pred_lat, pred_long)
    ax[1].plot(dist[:400], color='tab:red', label="Error")
    if (j == 0):
        ax[1].plot([0, 0], color='tab:blue', label="Attack area")
    # draw a transparent box between start and end
    y_min = 0
    y_max = 295
    # ax[1].fill_between([start, end], y_min, y_max, color='tab:blue', alpha=0, linewidth=2)
    ax[1].add_patch(plt.Rectangle((start, y_min), end-start - 10, y_max-y_min, color='tab:blue', fill=None, alpha=1, linewidth=2))
    ax[1].set_xlabel("time-step (s)")
    ax[1].set_ylabel("distance (m)")

    ax[1].set_ylim([0, 300])
    if (j == 0):
        ax[1].legend(fontsize="12", loc ="upper right")
    ax[1].set_title(file, loc="left")

    j+=1


plt.tight_layout()
plt.show()






print(max_i)
for i in range(len(means_dist)):
    print(means_dist[i][0], means_dist[i][1])

