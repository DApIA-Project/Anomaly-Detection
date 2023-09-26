import folium
from C_Constants.AircraftClassification import DefaultCTX as CTX
import os
import pandas as pd

# CTX.BOUNDING_BOX
center = [
    (CTX.BOUNDING_BOX[0][0] + CTX.BOUNDING_BOX[1][0]) / 2,
    (CTX.BOUNDING_BOX[0][1] + CTX.BOUNDING_BOX[1][1]) / 2
]

# Create a Map instance
m = folium.Map(location=center, zoom_start=11)


labels = pd.read_csv("./A_Dataset/AircraftClassification/labels.csv", sep=",", header=None)
labels.columns = ["icao24", "label"]
labels = {row["icao24"]:row["label"] for _, row in labels.iterrows()}

folder = "./A_Dataset/AircraftClassification/Train"
csvs = os.listdir(folder)
# filter .csv
csvs = [csv for csv in csvs if csv.endswith(".csv")]

COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "white", "pink", "lightblue", "lightgreen", "gray", "black", "lightgray"]

LABELS = {
    1 : "CARGO",
    2 : "COMMERCIAL",
    3 : "JET",
    4 : "ATR",
    5 : "MEDIUM",
    6 : "LIGHT",
    7 : "VERY_LIGHT",
    9 : "HELICOPTER",
    8 : "GLIDER",
    10 : "ULM",
    11 : "MILITARY",
    12 : "SAMU",
    13 : "TOO_HIGH"
}

lgd_txt = '<span style="color: {col};">{txt}</span>'
groups = {label:folium.FeatureGroup(name=lgd_txt.format(col=COLORS[label], txt=LABELS[label])) for label in LABELS}

for i in range(len(csvs)):
    csv = csvs[i]
    df = pd.read_csv(os.path.join(folder, csv), sep=",", dtype={"icao24":str})

    icao = df["icao24"].iloc[0]
    callsign = df["callsign"].iloc[0]

    if (icao in labels):
        lat = df["latitude"].iloc[0]
        lon = df["longitude"].iloc[0]
        label = labels[icao]
        if (label == 0):
            continue


        if (df["altitude"].iloc[0] > 1000):
            label = 13

        # if ("SAMU" in callsign):
        #     label = 12

        # add a cross marker to the map at lat, lon
        folium.CircleMarker(location=[lat, lon], radius=4, color=COLORS[label], fill=True, fill_color=COLORS[label]).add_to(groups[label])

for label in groups:
    groups[label].add_to(m)

folium.LayerControl('topleft', collapsed= False).add_to(m)


m