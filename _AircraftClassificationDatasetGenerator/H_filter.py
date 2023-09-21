import os
import pandas as pd

for folder in ["Train", "Eval"]:
    print("Processing", folder)


    if (os.path.exists('./dataset/removed_from_'+folder+'/')):
        # move back ignored files 
        ignored_files = os.listdir('./dataset/removed_from_'+folder+'/')
        ignored_files = [file for file in ignored_files if file.endswith('.csv')]

        for f in range(len(ignored_files)):
            file = ignored_files[f]
            os.rename("./dataset/removed_from_"+folder+"/"+file, './dataset/'+folder+"/"+file)

    else:
        os.mkdir('./dataset/removed_from_'+folder+'/')


    files = os.listdir('./dataset/'+folder)
    files = [file for file in files if file.endswith('.csv')]

    # remove outliers from training dataset
    ignored_files = []
    f = 0
    for f in range(len(files)):
        if (f % 100 == 0):
            print("\r", f, "/", len(files), end="")

        file = files[f]

        df = pd.read_csv('./dataset/'+folder+"/" + file, dtype={'icao24': str})

        altitude = df['altitude'].values
        geoaltitude = df['geoaltitude'].values

        if (altitude[0] > 2000 or geoaltitude[0] > 2000):
            ignored_files.append(file)


    for f in range(len(ignored_files)):
        file = ignored_files[f]
        os.rename('./dataset/'+folder+"/" + file, './dataset/removed_from_'+folder+"/" + file)

    print("Ignored files: ", len(ignored_files),"/", len(files))