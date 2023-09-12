icao2Type = open("./labels/icao2type.csv", "r")
icao2Type = icao2Type.readlines()
icao2Type = [line.strip().split(",") for line in icao2Type]
icao2Type = {line[0]: line[1] for line in icao2Type}


type2Label = open("./labels/aircraft2label.csv", "r")
type2Label = type2Label.readlines()
type2Label = [line.strip().split(",") for line in type2Label]
type2Label = {line[0]: line[1] for line in type2Label}

database = {}


for icao in icao2Type:
    aircraftType = icao2Type[icao]

    if (aircraftType not in type2Label):
        print("Missing label for " + aircraftType)
        continue

    database[icao] = type2Label[aircraftType]

# save database.csv
file = open("./labels/labels.csv", "w")
for icao in database:
    file.write(icao + "," + str(database[icao]) + "\n")
file.close()

    
