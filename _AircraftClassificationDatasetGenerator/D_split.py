import pandas as pd
import os
from _Utils.numpy import np, ax

EVAL_PROP = 0.075
from A_parquet2csv import FOLDER

files = os.listdir('./B_csv/'+FOLDER)
files = [file for file in files if file.endswith('.csv')]

np.random.shuffle(files)
labels = pd.read_csv('./labels/labels.csv', dtype={'icao24': str}, header=None)
# to dict
labels = {row[0]: row[1] for index, row in labels.iterrows()}

icaos = {} 
file_per_labels = {}

for file in files:
    df = pd.read_csv('./B_csv/'+FOLDER+"/" + file, dtype={'icao24': str})
    icao = df["icao24"][0]

    if (icao in labels):
        label = labels[icao]
        if (label in file_per_labels):
            file_per_labels[label].append(file)
        else:
            file_per_labels[label] = [file]


eval_files = []
train_files = []

for label in file_per_labels:
    tot = len(file_per_labels[label])
    nb_eval_label = int(EVAL_PROP * tot)
    nb_train_label = tot - nb_eval_label

    eval_files += file_per_labels[label][:nb_eval_label]
    train_files += file_per_labels[label][nb_eval_label:]

if (not os.path.exists('./C_dataset')):
    os.mkdir('./C_dataset')
if (not os.path.exists('./C_dataset/'+FOLDER)):
    os.mkdir('./C_dataset/'+FOLDER)
if (not os.path.exists('./C_dataset/'+FOLDER+'/Train')):
    os.mkdir('./C_dataset/'+FOLDER+'/Train')
if (not os.path.exists('./C_dataset/'+FOLDER+'/Eval')):
    os.mkdir('./C_dataset/'+FOLDER+'/Eval')

for file in eval_files:
    os.rename('./B_csv/'+FOLDER+"/" + file, './C_dataset/'+FOLDER+'/Eval/' + file)

for file in train_files:
    os.rename('./B_csv/'+FOLDER+"/" + file, './C_dataset/'+FOLDER+'/Train/' + file)

print("Done !")









