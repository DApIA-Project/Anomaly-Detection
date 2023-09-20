# unsplit ./dataset/Train/.csv and ./dataset/Eval/.csv into csv/

import os

train = os.listdir('./dataset/Train/')
train = [file for file in train if file.endswith('.csv')]
eval = os.listdir('./dataset/Eval/')
eval = [file for file in eval if file.endswith('.csv')]

train = [file for file in train if file.endswith('.csv')]
eval = [file for file in eval if file.endswith('.csv')]

for file in train:
    os.rename('./dataset/Train/' + file, './csv/' + file)

for file in eval:
    os.rename('./dataset/Eval/' + file, './csv/' + file)

    
