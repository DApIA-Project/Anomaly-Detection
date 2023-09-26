import os
import pandas as pd
import numpy as np

BOUNDING_BOX = [
    (43.11581, 0.72561),
    (44.07449, 2.16344)
]

MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4, 5], # PLANE
    # 5: [5], # Normal
    6: [6, 7, 10], # SMALL
    9: [9, 12], # HELICOPTER
    # 12: [12], # SAMU
    11: [11], # military
}

SCALER_LABELS = [2, 6, 9, 11]

def lat_lon_dist_meters(lat1, lon1, lat2, lon2):
    """
    return the distance between two points in meters
    """
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000 # meters

def distance_with_bounding_box_border(lat, lon):
    """
    return the distance between the point (lat, lon) and the closest border of the bounding box
    """
    lat_min, lon_min = BOUNDING_BOX[0]
    lat_max, lon_max = BOUNDING_BOX[1]

    borders_pos=[
        (lat_min, lon),
        (lat_max, lon),
        (lat, lon_min),
        (lat, lon_max),
    ]

    dists = [lat_lon_dist_meters(lat, lon, lat_b, lon_b) for lat_b, lon_b in borders_pos]

    return np.min(dists)



   

def post_process_preds(y_, df):
    """
    post process predictions
    y_: np.array of shape (timesteps, class_probability)

    return: 0, 1, 2, ..., n, the most probable class
    """

    # remove predictions with a distance to the border of the bounding box < 1000m
    start = 0
    end = len(y_) - 1
    while (start < len(y_) and distance_with_bounding_box_border(df["latitude"].iloc[start], df["longitude"].iloc[start]) < 5000):
        start += 1
    while (end >= 0 and distance_with_bounding_box_border(df["latitude"].iloc[end], df["longitude"].iloc[end]) < 5000):
        end -= 1

    sub_y_ = y_.copy()
    sub_df = df.copy()

    if (end - start >= 500):
        sub_y_ = y_[start:end+1]
        sub_df = df.iloc[start:end+1]
    


    models_confidency = np.max(sub_y_, axis=1)

    if (False):
        tmp_labels = np.argmax(sub_y_, axis=1)
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        # plot confidence line, with color depending on the predicted class
        for i in range(len(SCALER_LABELS)):
            wre = np.where(tmp_labels==i)
            conf = models_confidency[wre]
            ts = df["timestamp"].iloc[wre]
            plt.scatter(ts, conf, c=colors[i], s=1, label=SCALER_LABELS[i])
        plt.legend()
        plt.show()
    

    # sort y_ by confidence
    y_sorted = sub_y_[np.argsort(models_confidency)]

    P = 100
    y_important = y_sorted[-P:]

    # mean
    mean = np.mean(y_important, axis=0)
    return np.argmax(mean)



# def post_process_preds(y_, df):
#     """
#     post process predictions
#     y_: np.array of shape (timesteps, class_probability)

#     return: 0, 1, 2, ..., n, the most probable class
#     """

#     # remove predictions with a distance to the border of the bounding box < 1000m
#     start = 0
#     end = len(y_) - 1
#     while (start < len(y_) and distance_with_bounding_box_border(df["latitude"].iloc[start], df["longitude"].iloc[start]) < 5000):
#         start += 1
#     while (end >= 0 and distance_with_bounding_box_border(df["latitude"].iloc[end], df["longitude"].iloc[end]) < 5000):
#         end -= 1

#     sub_y_ = y_.copy()
#     sub_df = df.copy()

#     if (end - start >= 500):
#         sub_y_ = y_[start:end+1]
#         sub_df = df.iloc[start:end+1]
    

#     per_timestep_label_pred = np.argmax(sub_y_, axis=1)

#     # split into area of consecutive same predictions
#     # [
#     #   [Start, End, Label],
#     #   [Start, End, Label],
#     #   ...
#     # ]
#     split = []
#     start = 0
#     end = 0
#     label = per_timestep_label_pred[0]
#     for i in range(1, len(per_timestep_label_pred)):
#         if (per_timestep_label_pred[i] != label):
#             split.append([start, end, label])
#             start = i
#             end = i
#             label = per_timestep_label_pred[i]
#         else:
#             end = i
#     split.append([start, end, label])

#     # remove area with less than 5% of the total time
#     to_remove = []
#     for i in range(len(split)):
#         lenght = split[i][1] - split[i][0]
#         if (lenght / len(sub_y_) * 100 < 5):
#             to_remove.append(i)
    
#     for i in reversed(to_remove):
#         split.pop(i)

#     pred = np.zeros(len(SCALER_LABELS))

#     for i in range(len(split)):
#         window = sub_y_[split[i][0]:split[i][1]+1]
#         label = split[i][2]

#         pred[label] += np.sum(window[:, label], axis=0)

#     return np.argmax(pred)








# def post_process_preds(y_):
#     """
#     post process predictions
#     y_: np.array of shape (timesteps, class_probability)

#     return: 0, 1, 2, ..., n, the most probable class
#     """

#     models_confidency = np.max(y_, axis=1)
#     pred = np.argmax(models_confidency)
#     return np.argmax(y_[pred])

# def post_process_preds(y_):
#     """
#     post process predictions
#     y_: np.array of shape (timesteps, class_probability)

#     return: 0, 1, 2, ..., n, the most probable class
#     """

#     # mean
#     mean = np.mean(y_, axis=0)
#     return np.argmax(mean)




    




files = os.listdir('./A_Dataset/AircraftClassification/Outputs/Eval/')
icao2label = os.path.join("./A_Dataset/AircraftClassification/labels.csv")


icao2label = pd.read_csv(icao2label, sep=",", header=None, dtype={"icao24":str})
icao2label.columns = ["icao24", "label"]
icao2label = icao2label.fillna("NULL")



 # merge labels asked as similar
for label in MERGE_LABELS:
    icao2label["label"] = icao2label["label"].replace(MERGE_LABELS[label], label)

# to dict
icao2label = icao2label.set_index("icao24").to_dict()["label"]




files = os.listdir('./A_Dataset/AircraftClassification/Outputs/Eval/')

confusion_matrix = np.zeros((len(SCALER_LABELS), len(SCALER_LABELS)), dtype=np.int32)

for i in range(len(files)):
    file = files[i]

    df = pd.read_csv(os.path.join('./A_Dataset/AircraftClassification/Outputs/Eval/', file), sep=",", dtype={"icao24":str})

    if (df["icao24"].iloc[0] not in icao2label):
        continue

    true_label = icao2label[df["icao24"].iloc[0]]
    y = np.zeros(len(SCALER_LABELS))
    y[SCALER_LABELS.index(true_label)] = 1

    y_str = df["y_"].to_numpy()
    y_ = np.array([y_str[i].split(";") for i in range(len(y_str))], dtype=np.float32)

    pred = post_process_preds(y_, df)
    pred = SCALER_LABELS[pred]

    confusion_matrix[SCALER_LABELS.index(true_label), SCALER_LABELS.index(pred)] += 1

    nb = 20
    r = int((i+1)/nb)

    if (pred != true_label):
        print(file, "\tY : ", true_label, " Y_ : ", pred, sep="")


acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

# plot confusion matrix
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.xticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
plt.yticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
plt.gca().xaxis.tick_bottom()
plt.title('Accuracy ' + str(round(acc*100, 1))+"%", fontsize=18)
plt.show()








    
