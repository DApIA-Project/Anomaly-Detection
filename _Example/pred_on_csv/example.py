from AdsbAnomalyDetector import predict, AnomalyType
import pandas as pd
import numpy as np

# using the model on two simultaneous flights
flight_1 = pd.read_csv("./2022-01-12_13-19-13_SAMU31_39ac45.csv", dtype=str)
flight_2 = pd.read_csv("./2022-01-15_14-25-40_FHJAT_39a413.csv", dtype=str) # replayed flight

# synchronizing the timestamps
timestamps_1 = flight_1["timestamp"].astype(np.int64)
timestamps_2 = flight_2["timestamp"].astype(np.int64)

flight_2["timestamp"] = timestamps_1[0] + (timestamps_2 - timestamps_2[0])
flight_2["timestamp"] = flight_2["timestamp"].astype(str)


# simulating the data stream
max_length = 400
for t in range(0, max_length):
    if (t % 100 == 0):
        print(t, "/", max_length)

    # retrieving messages that arrived at time t
    messages = []
    if (t < len(flight_1)):
        messages.append(flight_1.iloc[t].to_dict())
    if (t < len(flight_2)):
        messages.append(flight_2.iloc[t].to_dict())

    # making predictions for these new messages
    # returns a prediction for each aircraft in a dictionary icao -> proba_array

    messages = predict(messages, compress=True, debug=True)

    for i in range(len(messages)):
        print(messages[i]["icao24"] \
            + " - Spoofing: " + str(messages[i]["anomaly"] == AnomalyType.SPOOFING)
            + " - Replay: "   + str(messages[i]["anomaly"] == AnomalyType.REPLAY)
            + " - Flooding: " + str(messages[i]["anomaly"] == AnomalyType.FLOODING)
            + " - Tag: "      + str(messages[i]["tag"]))





