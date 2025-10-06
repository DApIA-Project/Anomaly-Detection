from AdsbAnomalyDetector import predict, AnomalyType, clear_cache
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

DIR = "../../A_Dataset/AircraftClassification/Train/"
csv = os.listdir(DIR)
csv = csv[:]
dfs = []

min_len = 512
for i in range(len(csv)):
    file = csv[i]
    df = pd.read_csv(DIR + file, dtype={"icao24": str, "callsign": str})
    df["tag"] = str(i+1) 
    if (len(df) < 512):
        continue
    
    dfs.append(df)
    dfs[-1]["timestamp"] = dfs[-1]["timestamp"] - (dfs[-1]["timestamp"][0] - dfs[0]["timestamp"].iloc[0])
    
for i in range(len(dfs)):
    dfs[i] = dfs[i][:min_len]
    
NB_SIMULTANEOUS_AIRCRAFT = 8
SIMULATION_SPEED = 1

x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
y = []
nb_calls = []


for NB_SIMULTANEOUS_AIRCRAFT in x:
    
    print()
    print()
    print("NB_SIMULTANEOUS_AIRCRAFT:", NB_SIMULTANEOUS_AIRCRAFT)
    print()

    times = []
    nb_call = 0
    
    for t in range(0, min_len-1, SIMULATION_SPEED):
        print("\r", t, "/", min_len, end="")
        
        messages = []
        
        for s in range(SIMULATION_SPEED):
            for i in range(NB_SIMULTANEOUS_AIRCRAFT):
                messages.append(dfs[i].iloc[t+s].to_dict())
            
        start = time.time()
        anomalies = predict(messages)                
        nb_call += 1
        end = time.time()
        times.append(end-start)
        
    for i in range(NB_SIMULTANEOUS_AIRCRAFT):
        icao24 = dfs[i]["icao24"].iloc[0]
        tag = dfs[i]["tag"].iloc[0]
        clear_cache(icao24, tag)
        
        
    pred_time = sum(times)
    real_duration = min_len
    
    y.append(pred_time)
    nb_calls.append(nb_call)
    
print(x)
print(y)
print(nb_calls)



