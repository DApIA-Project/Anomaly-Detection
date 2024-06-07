from _Utils.os_wrapper import os
import pandas as pd

icao2aircraft = open("./labels/B_icao2aircraft.csv", "r")
icao2aircraft = icao2aircraft.readlines()
icao2aircraft = [line.strip().split(",") for line in icao2aircraft]
icao2aircraft = {line[0]: line[1] for line in icao2aircraft}


type2Label = open("./labels/C_aircraft2label.csv", "r")
type2Label = type2Label.readlines()
type2Label = [line.strip().split(",") for line in type2Label]
type2Label = {line[0]: line[1] for line in type2Label}

database = {}


for icao in icao2aircraft:
    aircraftType = icao2aircraft[icao]

    if (aircraftType not in type2Label):
        print("Missing label for " + aircraftType)
        continue

    database[icao] = type2Label[aircraftType]

# add samu label wich depands on callsign
folders = os.listdir("./B_csv")
# only keep folders
folders = [folder for folder in folders if os.path.isdir("./B_csv/" + folder)]
files = []
for folder in folders:
    f = os.listdir("./B_csv/" + folder)
    f = [folder + "/" + file for file in f]
    files += f

for file in files:
    df = pd.read_csv("./B_csv/" + file, dtype={"icao24": str, "callsign": str})
    # check if any row has a samu in it's callsign
    if (df["callsign"].str.contains("SAMU").any()):
        icao = df["icao24"].values[0]

        if (icao not in database or database[icao] != 12):
            database[icao] = 12
            print("SAMU " + icao)


# save database.csv
file = open("./labels/labels.csv", "w")
for icao in database:
    file.write(icao + "," + str(database[icao]) + "\n")
file.close()

    