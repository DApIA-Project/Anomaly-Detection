 
import pandas as pd

df = pd.read_csv("./aircraft.txt", sep=",", header=None, names=["callsign", "icao24", "label"])

# remove callsing column
df = df.drop(columns=["callsign"])

# save to csv labeled.csv
df.to_csv("./labels.csv", index=False, header=False)