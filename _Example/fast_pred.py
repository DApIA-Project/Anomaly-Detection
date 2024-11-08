from AdsbAnomalyDetector import predict, AnomalyType


messages = predict([
    {
        'timestamp': '1641993553',
        'icao24': '39ac45', 'callsign': 'SAMU31',
        'latitude': '43.76673889160156', 
        'longitude': '1.0297127657158431',
        'groundspeed': '31.0', 'track': '0.0', 
        'vertical_rate': '-192.0',
        'onground': 'False', 'alert': 'False', 
        'spi': 'False', 'squawk': "0",
        'altitude': '550.0', 'geoaltitude': '1250.0'
    },
    {   'timestamp': '1641993553',
        'icao24': '39a413', 'callsign': 'FHJAT',
        'latitude': '43.587066650390625', 
        'longitude': '1.4989950490552326',
        'groundspeed': '58.0', 'track': '338.749494', 
        'vertical_rate': '832.0',
        'onground': 'False', 'alert': 'True',
        'spi': 'False', 'squawk': "0",
        'altitude': '0.0', 'geoaltitude': '650.0'
    }
], compress=False)

if (messages[0]["anomaly"] == AnomalyType.VALID):
    print(messages[0]["icao24"], "is normal")
elif (messages[0]["anomaly"] == AnomalyType.SPOOFING):
    print(messages[0]["icao24"], "is spoofing")
elif (messages[0]["anomaly"] == AnomalyType.REPLAY):
    print(messages[0]["icao24"], "is replaying")
elif (messages[0]["anomaly"] == AnomalyType.FLOODING):
    print(messages[0]["icao24"], "is flooding")



