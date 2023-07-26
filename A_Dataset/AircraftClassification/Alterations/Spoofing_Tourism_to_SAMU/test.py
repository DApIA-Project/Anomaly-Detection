import pandas 
import os

# files = os.listdir('.')
files = ["./SAMU31.csv"]

date = '2022-01-01 00:10:01'
date_ts_sec = pandas.Timestamp(date).value / 1000000000

for file in files:
    if (file.endswith('.csv')):
        df = pandas.read_csv(file, dtype={"icao24":str})

        df["timestamp"] = df['time'] - df['time'][0] + date_ts_sec

        df.to_csv(file, index=False)

