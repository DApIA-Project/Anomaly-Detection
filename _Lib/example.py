from AdsbAnomalyDetector import predictAircraftType, probabilityToLabel, labelToName, getTruthLabelFromIcao
import pandas as pd
import numpy as np

# utilisation du modèle sur deux vols simultanés
flight_1 = pd.read_csv("./2022-01-15_14-25-40_FHJAT_39a413.csv", dtype=str)
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
    a = predictAircraftType(messages)

    # stockage des prédictions
    for icao, proba in a.items():
        predictions[icao].append(list(proba.values())[0])
print("done")

# traitement des prédictions, transformation des probabilités en labels
# verification de la cohérence des labels avec la vérité terrain
labels_flight_1 = probabilityToLabel(predictions[flight_1["icao24"][0]])
major_label_flight_1 = np.bincount(labels_flight_1).argmax()
print("flight with icao", flight_1["icao24"][0], "is a", labelToName(major_label_flight_1))
truth = getTruthLabelFromIcao(flight_1["icao24"][0])
if (truth == 0):
    print("Unknon icao24 : impossible to check aircraft's has the right icao24")
else:
    if (truth == major_label_flight_1):
        print("The flight is legit")
    else:
        print("ALERT ! The flight pretend to be", labelToName(truth), "but is a", labelToName(major_label_flight_1))


labels_flight_2 = probabilityToLabel(predictions[flight_2["icao24"][0]])
major_label_flight_2 = np.bincount(labels_flight_2).argmax()
print("flight with icao", flight_2["icao24"][0], "is a", labelToName(major_label_flight_2))
truth = getTruthLabelFromIcao(flight_2["icao24"][0])
if (truth == 0):
    print("Unknon icao24 : impossible to check aircraft's has the right icao24")
else:
    if (truth == major_label_flight_2):
        print("The flight is legit")
    else:
        print("ALERT ! The flight pretend to be", labelToName(truth), "but is a", labelToName(major_label_flight_2))





