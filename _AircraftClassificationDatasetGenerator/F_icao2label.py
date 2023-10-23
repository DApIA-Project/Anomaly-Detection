import os
import pandas as pd

icao2aircraft = open("./labels/icao2aircraft.csv", "r")
icao2aircraft = icao2aircraft.readlines()
icao2aircraft = [line.strip().split(",") for line in icao2aircraft]
icao2aircraft = {line[0]: line[1] for line in icao2aircraft}


type2Label = open("./labels/aircraft2label.csv", "r")
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
files = os.listdir("./csv")
files = [file for file in files if file.endswith(".csv")]
files = [file for file in files if "SAMU" in file]
for file in files:
    df = pd.read_csv("./csv/" + file, dtype={"icao24": str})
    icao = df["icao24"].values[0]
    database[icao] = 12


# save database.csv
file = open("./labels/labels.csv", "w")
for icao in database:
    file.write(icao + "," + str(database[icao]) + "\n")
file.close()

    
# install
os.system("cp ./labels/labels.csv ../A_Dataset/AircraftClassification/labels.csv")
