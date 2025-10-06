
import pandas as pd
from AdsbAnomalyDetector import predict
df = pd.read_csv("./15-15-33_FJDGY_3a2cbc.csv")

# Convert the DataFrame to a list of dictionaries
adsb_msg_list = df.to_dict(orient='records')

# call anomaly detection models
adsb_msg_list = predict(adsb_msg_list, compress=False)

# do whatever you want with the messages
for message in adsb_msg_list:
    if (message["anomaly"]):
        print(f"""Anomaly detected: 
                    for: {message['icao24']}, 
                    at: {message['timestamp']}""")




