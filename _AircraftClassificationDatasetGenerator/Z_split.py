import pandas as pd
import os

files = os.listdir('./csv/')
labels = {l.split(",")[0]:l.split(",")[1] for l in open('database.txt', 'r').read().split('\n')}

icaos = {} 

for file in files:
    df = pd.read_csv('./csv/' + file, dtype={'icao24': str})
    # check if altitude is never < 500
    if (df['altitude'].max() < 1200):
        print("ERROR: " + file + " has altitude < 500")
        # exit(1)

    # icao = df["icao24"][0]
    # if (icao in labels):
    #     label = labels[icao]

    #     if (label in icaos):
    #         icaos[label].append(file)
    #     else:
    #         icaos[label] = [file]
        
    # else:
    #     print("ERROR: " + file + " is not in database")
    #     exit(1)

# sort icaos dict by key
icaos = {int(k): v for k, v in sorted(icaos.items(), key=lambda item: int(item[0]))}

for label, files in icaos.items():
    print(label,":", len(files))

# split 10% test 90% train, with equilibrated classes
train = []
test = []

for label, files in icaos.items():
    if (label == 0):
        continue
    split_index = int(len(files) * (1-0.9))
    train += files[split_index:]
    test += files[:split_index]

print("train:", len(train))
print("test:", len(test))

# create split folder
if not os.path.exists('./split'):
    os.makedirs('./split')
else:
    # clean
    os.system('rm -rf ./split/*')

# copy test files into test/ and train files into train/
os.makedirs('./split/Train')
os.makedirs('./split/Eval')

for file in train:
    os.system('cp ./csv/' + file + ' ./split/Train/')

for file in test:
    os.system('cp ./csv/' + file + ' ./split/Eval/')




