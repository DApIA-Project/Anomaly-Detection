# create a bot wich simulate key and mouse

import os
import requests
import json
import time

TERMINAL_COLOR = {

    "BLACK": "\033[30m",
    "DARK RED": "\033[31m",
    "DARK GREEN": "\033[32m",
    "DARK YELLOW": "\033[33m",
    "DARK BLUE": "\033[34m",
    "DARK MAGENTA": "\033[35m",
    "DARK CYAN": "\033[36m",
    "GRAY": "\033[37m",
    "DARK GRAY": "\033[90m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "RESET": "\033[0m",
}

LABELS = {
    "SUPER_HEAVY": 1, # machandises, passagers : > 255,000 lb (Boeing 747, Airbus A340)
    "HEAVY": 2, # autres avions commerciaux
    "JET": 3, # jets régionaux
    "TURBOPROP": 4, # avions à hélices (ATR)
    "MEDIUM": 5, # avions légers, multirotors, multi places
    "LIGHT": 6, # avions mono/bi place légers
    "SUPER LIGHT" : 7, # ULM (ultra léger motorisé)
    "GLIDER": 8, # planeur
    "HELICOPTER": 9, # hélicoptère
    "ULM": 10, # drone
    "MILITARY": 11, # militaire
    "SAMU":12 # samu
}


icao_to_labelize = open("./labels/A_new_icao.csv", "r").read().splitlines()
icao2aircraft = {} # icao24 -> type


min_label = -1
max_label = -1

for key in LABELS:
    if (min_label == -1 or LABELS[key] < min_label):
        min_label = LABELS[key]
    if (max_label == -1 or LABELS[key] > max_label):
        max_label = LABELS[key]


def save(icao2aircraft):
    # save db
    file = open("./labels/B_icao2aircraft.csv", "w")
    for icao in icao2aircraft:
        file.write(icao + "," + icao2aircraft[icao] + "\n")
    file.close()
    

def load():
    _icao2aircraft = {}

    if (os.path.isfile("./labels/B_icao2aircraft.csv")):
        file = open("./labels/B_icao2aircraft.csv", "r")
        for line in file:
            line = line[:-1]
            l = line.split(",")
            icao = l[0]
            type = l[1]
            _icao2aircraft[icao] = type
        file.close()

    return _icao2aircraft




icao2aircraft = load()
i = 0
for i in range(len(icao_to_labelize)):
    icao = icao_to_labelize[i]

    if (icao in icao2aircraft):
        continue

    # https://api.flightradar24.com/common/v1/search.json?fetchBy=reg&query=$icao
    url = "https://api.flightradar24.com/common/v1/search.json?fetchBy=reg&query=" + icao
    #header mozila
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    }
    req = requests.get(url, headers=headers)
    text = req.text

    if ("Our engineers are working hard to make the service available as soon as possible." in text):
        print(TERMINAL_COLOR["RED"] + "server down !" + TERMINAL_COLOR["RESET"])
        time.sleep(10)
        continue

    res = json.loads(req.text)
    aircrafts = res["result"]["response"]["aircraft"]["data"]
    if (aircrafts == None): aircrafts = []

    to_remove = []
    for aircraft in aircrafts:
        if (aircraft["hex"] != icao.upper()):
            to_remove.append(aircraft)
    for aircraft in to_remove:
        aircrafts.remove(aircraft)

    if (len(aircrafts) == 0):
        print(TERMINAL_COLOR["RED"] + "No aircraft found for " + icao + TERMINAL_COLOR["RESET"])

    elif (len(aircrafts) > 1):
        print(TERMINAL_COLOR["RED"] + "Multiple aircraft found for " + icao + TERMINAL_COLOR["RESET"])

    else:
        aircraft = aircrafts[0]
        aircraft_name = aircraft["model"]["text"]
        if (aircraft_name == None): aircraft_name = "UNKNOWN"
        elif (aircraft_name == ""): aircraft_name = "UNKNOWN"
        elif (aircraft_name == ".."): aircraft_name = "UNKNOWN"
        elif (aircraft_name == "--"): aircraft_name = "UNKNOWN"

        icao2aircraft[icao] = aircraft_name
        print(TERMINAL_COLOR["GREEN"] + icao + " -> " + aircraft_name + TERMINAL_COLOR["RESET"])
        save(icao2aircraft)

    time.sleep(0.5) # avoid ban