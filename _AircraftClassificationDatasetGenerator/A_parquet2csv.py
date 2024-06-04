 
import pandas as pd
import math
import os
from datetime import datetime
import time
import random
from _Utils.numpy import np, ax

""" read parquet file and split it into separates flights """

MIN_DURATION = 15*60 # 15 minutes
SPLIT_FLIGHT_GAP = 5*60 # 5 minutes

FOLDER = "ToulouseV1"

BOXS = {
    "ToulouseV1" : [(43.11581, 0.72561), (44.07449, 2.16344)],
    "ToulouseV2" : [(43.11581, 0.72561), (44.07449, 2.16344)],
    "NiceMonacoV1" : [(43.469632, 6.894108), (43.764860, 7.361892)],
}
BOX = BOXS[FOLDER]





"""
Simple, basic and fast check to first filter out VERY invalid flights
"""
def flight_is_valid(flight_df):
    if (len(flight_df) < MIN_DURATION):
        return False
    
    flight_lenght = flight_df['timestamp'].iloc[-1] - flight_df['timestamp'].iloc[0]
    if flight_lenght < MIN_DURATION:
        return False

    # check that at least one timestamp is in the box
    in_box = 0
    for i in range(len(flight_df)):
        if (BOX[0][0] <= flight_df['latitude'].iloc[i] <= BOX[1][0]) and (BOX[0][1] <= flight_df['longitude'].iloc[i] <= BOX[1][1]):
            in_box += 1
            if (in_box > MIN_DURATION//3):
                return True
    return False




# main
if __name__ == "__main__":

    # clean or create working directory
    if (os.path.isdir("./B_csv/"+FOLDER)):
        os.system("rm -r ./B_csv/"+FOLDER+"/*")
    else:
        os.mkdir("./B_csv/"+FOLDER)

    # list parquet to work on
    files = os.listdir('./A_parquets/'+FOLDER)
    files = [file for file in files if file.endswith('.parquet')]

    # for easier debugging in jupyter
    file = files[0]
    i = 0

    for i in range(len(files)):

        file = files[i]
        print(i,"-",file)

        df = pd.read_parquet('./A_parquets/'+FOLDER+'/'+file)
        
        # save df
        df.to_parquet('./A_parquets/'+FOLDER+'/'+file, compression='gzip')
        
        # pre-clean the dataframe
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["last_position", "hour"]) # drop last_position and hour columns
        df = df.dropna(subset=['latitude', 'longitude']) # drop each row with a latitute or longitude null
        if ("serials" in df.columns): df = df.drop(columns=["serials"]) # drop serials column

            
        df['callsign'] = df['callsign'].fillna("None") # remplace each None callsign by "None"

        # get all unique icao24
        icaos = df['icao24'].drop_duplicates().values
        # sort df by icao24 and then timestamp for easier spliting
        df = df.sort_values(by=['icao24', 'timestamp'])
        df = df.reset_index(drop=True)

        # get all lines for each icao24 
        ti = time.time()
        print("\nSort messages by icao24")
        start, end = 0, 0 # try to find the slice
        lines = {}
        for end in range(len(df)):
            if (df["icao24"][start] != df["icao24"][end]): # icao change => HOP ! start a new flight !

                if (df['icao24'][start] in lines): 
                    raise Exception("ALERT ! " + df['icao24'][start] + " already exist but shouldn't")

                lines[df["icao24"][start]] = (start, end)
                start = end
        lines[df["icao24"][start]] = (start, end)

        df_per_icao = []
        for key in lines:
            print(f"\r{key} - {lines[key][1]-lines[key][0]}", end="")
            sub = df.iloc[lines[key][0]:lines[key][1]]
            sub = sub.reset_index(drop=True)
            df_per_icao.append(sub)
        print()
        print("time: ", time.time()-ti)
        del lines




        flight_df = []
        ti = time.time()
        print("\nsplit messages in flights")
        for i in range(len(df_per_icao)):
            print(f"\r{i}/{len(df_per_icao)}, nb : {len(flight_df)}", "time : ", time.time()-ti, end="")

            split_indexs = [0]
            for t in range(1, len(df_per_icao[i])):
                # check the gap between consecutives message do not exeed SPLIT_FLIGHT_GAP
                if df_per_icao[i]['timestamp'][t] - df_per_icao[i]['timestamp'][t-1] > SPLIT_FLIGHT_GAP:
                    split_indexs.append(t)
            split_indexs.append(len(df_per_icao[i]))

            for j in range(len(split_indexs)-1):
                flight = df_per_icao[i][split_indexs[j]:split_indexs[j+1]].reset_index(drop=True)
                if flight_is_valid(flight):
                    flight_df.append(flight)
        print()

            

        print("\nsave csv")
        ti = time.time()
        for i in range(len(flight_df)):
            # convert datetime to timestamp
            
            print(f"\r{i}/{len(flight_df)}, time : ", time.time()-ti, end="")
            # gen file name
            ts = datetime.fromtimestamp(flight_df[i]['timestamp'].iloc[0])
            date_str = str(ts.year) + str(ts.month).zfill(2) + str(ts.day).zfill(2) + '_' + str(ts.hour).zfill(2) + str(ts.minute).zfill(2) + str(ts.second).zfill(2)
            name = date_str + '_' + str(flight_df[i]['callsign'].iloc[0]) + '_' + str(flight_df[i]['icao24'].iloc[0])
            flight_df[i].to_csv('./B_csv/'+FOLDER+'/' + name + '.csv', index=False)

        print()
        print()
