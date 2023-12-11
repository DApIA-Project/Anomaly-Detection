import pandas as pd
import os

# rm ./out/*
os.system("rm ./out/*")


df = pd.read_csv("./base3.csv", dtype={"icao24":str, "callsign":str})

# split on icao24
sub_db = []
for icao24 in df["icao24"].unique():
    sub_db.append(df[df["icao24"] == icao24])

true_icao = df["icao24"][0]
true_icao_i = 0
for i, sub in enumerate(sub_db):
    if sub["icao24"].iloc[0] == true_icao:
        true_icao_i = i
        break

for i in range(len(sub_db)):
    if (i == true_icao_i):
        # save
        sub_db[i].to_csv("./out/" + true_icao + ".csv", index=False)
        continue

    ts_0 = sub_db[i]["timestamp"].iloc[0]

    # insert at beginning row where 
    before_ts_0 = sub_db[true_icao_i][sub_db[true_icao_i]["timestamp"] < ts_0]
    # insert at the beginning
    icao = sub_db[i]["icao24"].iloc[0]
    sub_db[i] = pd.concat([before_ts_0, sub_db[i]])
    sub_db[i]["icao24"] = icao

    print(len(before_ts_0))

    # drop last_position,lastcontact,hour columns
    sub_db[i] = sub_db[i].drop(["last_position", "lastcontact", "hour"], axis=1)

    # save csv to out/ with name icao24.csv
    sub_db[i].to_csv("./out/" + icao + ".csv", index=False)




    

