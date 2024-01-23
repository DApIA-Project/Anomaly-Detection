
import os
import pandas as pd

FOLDER = "NiceMonacoV1"


files = os.listdir("./A_parquets/"+FOLDER)
files = [f for f in files if f.endswith('.csv')]


for f in files:
    df = pd.read_csv("./A_parquets/"+FOLDER +"/" + f, dtype={'icao24': str, 'callsign': str})
    # set "onground":bool, "alert":bool, "spi":bool
    df['onground'] = df['onground'].astype(bool)
    df['alert'] = df['alert'].astype(bool)
    df['spi'] = df['spi'].astype(bool)

    
    # sort by timestamp
    df = df.sort_values(by=['time'])

    # do some conversion :
    # convert csv time (timestamp) to timestamp (Datetime)
    # rename time to timestamp
    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # rename lat and lon to latitude and longitude
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    # rename velocity to groundspeed
    # and convert from m/s to knts
    df = df.rename(columns={'velocity': 'groundspeed'})
    df['groundspeed'] = df['groundspeed'] * 1.9438452

    # rename heading to track
    df = df.rename(columns={'heading': 'track'})

    # rename vertrate to vertical_rate
    # and convert from m/s to ft/min
    df = df.rename(columns={'vertrate': 'vertical_rate'})
    df['vertical_rate'] = df['vertical_rate'] * 196.85039370079


    # rename baroaltitude to altitude
    # and convert from m to ft
    df = df.rename(columns={'baroaltitude': 'altitude'})
    df['altitude'] = df['altitude'] * 3.2808398950131

    # convert geoaltitude to ft
    df['geoaltitude'] = df['geoaltitude'] * 3.2808398950131

    # rename lastposupdate to last_position 
    df = df.rename(columns={'lastposupdate': 'last_position'})

    # remove lastcontact
    df = df.drop(columns=['lastcontact'])

    df.to_parquet("./A_parquets/"+FOLDER+"/"+f[:-4]+'.parquet', compression='gzip')

os.system("rm ./A_parquets/"+FOLDER+"/*.csv")