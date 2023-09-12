import os
import pyautogui
import time 


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
}
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



icaoToType = open("./labels/icao2type.csv", "r")
icaoToType = icaoToType.readlines()
icaoToType = [line.strip().split(",") for line in icaoToType]

typesToLabel = set([line[1] for line in icaoToType])


labeledType = {}
if os.path.exists("./labels/aircraft2label.csv"):
    labeledType = open("./labels/aircraft2label.csv", "r")
    labeledType = labeledType.readlines()
    labeledType = [line.strip().split(",") for line in labeledType]
    labeledType = {line[0]: line[1] for line in labeledType}

for key in labeledType.keys():
    typesToLabel.remove(key)

os.system("open google.com")

for key in typesToLabel:

    if ("Airbus A350" in key):
        labeledType[key] = 1
        continue
    if ("Airbus A380" in key):
        labeledType[key] = 1
        continue
    if ("Boeing 747" in key):
        labeledType[key] = 1
        continue
    if ("Boeing 737" in key):
        labeledType[key] = 2
        continue

    # search on a new chrome tab the aircraft
    research = key

    if ("helicopter" in key.lower()):
        pass
    elif ("airbus" in key.lower() or "boeing" in key.lower()):
        research = research + " weight lbs"
    else:        
        research = research + " aircraft"
        

    
    research = research.replace(" ", "+").replace("&", "%26")

    url = "https://www.google.com/search?q=" + research + ""
    os.system("open \"" + url + "\"")

    time.sleep(1)
    pyautogui.hotkey("alt", "tab")

    print("Define the type of the aircraft: " + TERMINAL_COLOR["DARK YELLOW"] + key + TERMINAL_COLOR["RESET"])
    for label in LABELS:
        print(TERMINAL_COLOR["DARK YELLOW"] + str(LABELS[label]) + TERMINAL_COLOR["RESET"] + ": " + label)

    label = -1
    while label not in LABELS.values() and label != 0:
        label = int(input("> "))

    labeledType[key] = label

    # save the label
    with open("./labels/aircraft2label.csv", "a") as file:
        file.write(key + "," + str(label) + "\n")



    time.sleep(0.1)
    pyautogui.hotkey("alt", "tab")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "w")




# save the label
with open("./labels/aircraft2label.csv", "w") as file:
    for key in labeledType.keys():
        file.write(key + "," + str(labeledType[key]) + "\n")
