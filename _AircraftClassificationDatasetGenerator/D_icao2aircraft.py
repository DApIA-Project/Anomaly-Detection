# create a bot wich simulate key and mouse

import pyautogui
import time
import pyperclip
import os

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

# while True:
#     time.sleep(0.01)
#     # print the mouse position
#     pos = pyautogui.position()
#     print(pos)
#     # print the pixel color at the mouse position
#     print(pyautogui.pixel(pos[0], pos[1]))


icao_to_labelize = open("./labels/icaos_list.csv", "r").read().splitlines()
icao2type = {} # icao24 -> type

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

min_label = -1
max_label = -1

for key in LABELS:
    if (min_label == -1 or LABELS[key] < min_label):
        min_label = LABELS[key]
    if (max_label == -1 or LABELS[key] > max_label):
        max_label = LABELS[key]


def save(icao2type):
    # save db
    file = open("./labels/icao2type.csv", "w")
    for icao in icao2type:
        file.write(icao + "," + icao2type[icao] + "\n")
    file.close()
    

def load():
    _icao2type = {}

    if (os.path.isfile("./labels/icao2type.csv")):
        file = open("./labels/icao2type.csv", "r")
        for line in file:
            line = line[:-1]
            l = line.split(",")
            icao = l[0]
            type = l[1]
            _icao2type[icao] = type
        file.close()

    return _icao2type






def _workaround_write(text):
    """
    This is a work-around for the bug in pyautogui.write() with non-QWERTY keyboards
    It copies the text to clipboard and pastes it, instead of typing it.
    """
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')
    pyperclip.copy('')



# read not.txt file
icao2type = load()


shortcut = [
    ("SAMU*", 6)
]

for i in range(len(icao_to_labelize)):
    icao = icao_to_labelize[i]

    if (icao in icao2type):
        continue


    # reseach in flight 24 database
    # click on search bar
    pyautogui.moveTo(1550, 319)
    pyautogui.click()

    # clear search bar
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')

    _workaround_write(icao+" ")

    # wait for the response
    pyautogui.moveTo(1487, 352)
    i = 0
    time.sleep(0.05)

    mouse_pos = pyautogui.position()
    pix = pyautogui.pixel(mouse_pos[0], mouse_pos[1])

    while (pix != (239, 248, 254)):
        time.sleep(0.1)
        i+=1
        if (i > 10):
            if (pix != (255, 255, 255)):
                print("Error: l'utilsateur à touché à la souris !")
                exit()
            break
        else: pix = pyautogui.pixel(mouse_pos[0], mouse_pos[1])


    # check if aircraft is found
    # pyautogui.moveTo(1492, 340)
    # time.sleep(0.05)
    if (pix == (239, 248, 254)):
        pyautogui.click()
        time.sleep(0.5)
        pyautogui.moveTo(990, 360)

        print(TERMINAL_COLOR["CYAN"]+icao+TERMINAL_COLOR["RESET"] + " is in the FLIGHT24 database")

        # copy name
        for i in range(3):
            time.sleep(0.2)
            pyautogui.click()
        time.sleep(0.5)

        # get name
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.1)
        aircraft_type = pyperclip.paste().strip()
        pyperclip.copy(aircraft_type)  
        if (aircraft_type == "-"): aircraft_type = ""
        print("AIRCRAFT type :"+TERMINAL_COLOR["CYAN"], aircraft_type, end=TERMINAL_COLOR["RESET"]+" ")


        # check if aircraft is in the aircraft database (never seen but known aircraft type)
        if (aircraft_type != ""):
            icao2type[icao] = aircraft_type
        else:
            icao2type[icao] = "UNKNOWN"

        # come back to former page
        pyautogui.moveTo(1492, 340)
        pyautogui.click()
        pyautogui.hotkey('alt', 'left')

        while (pyautogui.pixel(mouse_pos[0], mouse_pos[1]) != (255, 255, 255)):
            time.sleep(0.3)
        time.sleep(0.5)
            
    

    print("[save]\n")
    save(icao2type)

    # check if there is a .screenshot* file
    files = os.listdir("./")
    for file in files:
        if (file.startswith(".screenshot")):
            os.system("rm ./.screenshot*")
            break
