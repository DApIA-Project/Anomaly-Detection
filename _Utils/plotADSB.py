import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from D_DataLoader.AircraftClassification.Utils import MAP, deg2num_int, deg2num, num2deg, compute_confidence



def genPlotMapBG(bounds):
    """Generate an image of the map from the bounds"""
    #######################################################
    # Convert lat, lon to px
    # thoses param are constants used to generate the map 
    zoom = 13
    min_lat, min_lon, max_lat, max_lon = 43.01581, 0.62561,  44.17449, 2.26344
    # conversion
    xmin, ymax = deg2num_int(min_lat, min_lon, zoom)
    xmax, ymin = deg2num_int(max_lat, max_lon, zoom)
    #######################################################

    x_min, y_max = deg2num(bounds[0], bounds[1], zoom)
    x_max, y_min = deg2num(bounds[2], bounds[3], zoom)
    x_min = int((x_min - xmin)*255.0)
    x_max = int((x_max - xmin)*255.0)
    y_min = int((y_min - ymin)*255.0)
    y_max = int((y_max - ymin)*255.0)

    # print(x_min, x_max, y_min, y_max)

    
    img = MAP[
        y_min:y_max,
        x_min:x_max, :]
    
    return img



def plotADSB(CTX, classes_, title, timestamp, lat, lon, groundspeed, track, vertical_rate, altitude, geoaltitude, probabilities, true, additional):
    
    LABEL_NAMES = CTX["LABEL_NAMES"]
    CLASSES_ = classes_
    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black", "gray", "gold"]


    # Y
    labels = np.argmax(probabilities, axis=1)
    labels = np.array([CLASSES_[label] for label in labels])
    confiance = compute_confidence(probabilities)

    # X
    timestamp = timestamp - timestamp[0]
    features = [[lat, lon], groundspeed, track, vertical_rate, altitude, geoaltitude]
    plot_names = ["Trace", "Groundspeed", "Track", "Vertical_rate", "Altitude", "Geoaltitude"]

    for i in range(len(additional)):
        features.append(additional[i][0])
        plot_names.append(additional[i][1])

    if (CTX["INPUT_PADDING"] == "valid"):
        features.append(confiance) # append confiance before padding because it is not padded

    # add nan row where timestamp[i-1] + 1 != timestamp[i]
    # this is to avoid plotting lines between two points that are not consecutive
    i = 0
    while (i < len(timestamp)-1):
        if (timestamp[i+1] != timestamp[i] +1):
            nb_row_to_add = timestamp[i+1] - timestamp[i] - 1
            for j in range(nb_row_to_add):
                timestamp = np.insert(timestamp, i+1+j, timestamp[i]+1+j)
                for f in range(len(features)):
                    if (f == 0): # trace
                        features[f][0] = np.insert(features[f][0], i+1+j, np.nan)
                        features[f][1] = np.insert(features[f][1], i+1+j, np.nan)
                    else:
                        features[f] = np.insert(features[f], i+1+j, np.nan)
            i += nb_row_to_add
        i += 1     

    
    if (CTX["INPUT_PADDING"] != "valid"):
        features.append(confiance) # append confiance after padding because it already padded

    plot_names.append("Confiance")

    # check if there is a missing timestamp
    for i in range(len(timestamp)-1):
        if (timestamp[i+1] != timestamp[i] +1):
            print("Missing timestamp: ", timestamp[i], timestamp[i+1])
            break


    # split the time series into zones of same, consectuive label
    zones = []
    start = 0
    for t in range(1, len(labels)):
        if (labels[t] != labels[t-1]):
            zones.append((start, t, labels[t-1]))
            start = t
    zones.append((start, len(labels), labels[-1]))

    # generate the map background
    bounds = [np.min(lat), np.min(lon), np.max(lat), np.max(lon)]
    img = genPlotMapBG(bounds)

    # compute the size of the figure
    lat_lon_ratio = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
    width = 10
    img_height = width * lat_lon_ratio
    height = img_height + (len(features)-1.0) * 1.0

    # start plotting
    fig, ax = plt.subplots(len(features), 1, figsize=(width, height), gridspec_kw={'height_ratios': [img_height] + [1] * (len(features)-1)})

    NB_TICKS = 5
    x_ticks_i = np.linspace(0, len(timestamp)-1, NB_TICKS, dtype=np.int32)
    x_ticks_f = timestamp[x_ticks_i]

    for i in range(len(features)):
        if (i == 0):
            # # prepare the map background + print a first trace for highlighting
            ax[i].imshow(img, extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)])
            ax[i].plot(lon, lat, color="black", linewidth=1.1, alpha=0.5)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            pass

        for z in range(len(zones)):
            zone = zones[z]
            start, end, label = zone
            end += 1

            if (i == 0):
                lat = features[i][0][start:end] 
                lon = features[i][1][start:end]
                ax[i].plot(lon, lat, color=COLORS[label], linewidth=3 if (label == true) else 1)
                
            else:
                feature = features[i][start:end] 
                ax[i].plot(timestamp[start:end], feature, color=COLORS[label], linewidth=2 if (label == true) else 1)
                ax[i].grid(True)
                # ax[i].set_title(plot_names[i]+":")
                ax[i].set_ylabel(plot_names[i])
                
                ax[i].set_xticks(x_ticks_f)

    # make global a legend on ax[1]
    for label in CLASSES_:
        ax[0].plot([], [], color=COLORS[label], label=LABEL_NAMES[label])
    ax[0].legend(loc='upper left')

    # add title on the very top
    ax[0].set_title(title)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, ax
    



