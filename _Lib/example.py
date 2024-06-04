from AdsbAnomalyDetector import predict
import pandas as pd
from _Utils.numpy import np, ax

# utilisation du modèle sur deux vols simultanés
flight_1 = pd.read_csv("./2022-01-12_13-19-13_SAMU31_39ac45.csv", dtype=str)
flight_2 = pd.read_csv("./2022-04-04_16-37-21_FJDGY_3a2cbc.csv", dtype=str)

# enregistrement des prédictions dans un dictionnaire qui associe
# l'icao à la liste des prédictions de l'avion
predictions = {}
predictions[flight_1["icao24"][0]] = []
predictions[flight_2["icao24"][0]] = []

# simulation du flux de données
max_lenght = 400
for t in range(0, max_lenght):
    if (t % 100 == 0):
        print(t, "/", max_lenght)

    # récupération des messages arrivés à l'instant t
    messages = []
    if (t < len(flight_1)):
        messages.append(flight_1.iloc[t].to_dict())
    if (t < len(flight_2)):
        messages.append(flight_2.iloc[t].to_dict())

    # réalisation de la prédiction pour ces nouveaux messages
    # retourne une prédiction pour chaque avion dans un dictionnaire icao -> proba_array
    messages = predict(messages)

    prnt = [(messages[i]["icao24"] + " - Spoofing: " + str(messages[i]["spoofing"])) for i in range(len(messages))]
    
    print(prnt)
