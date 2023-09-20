# move the Train and Eval folders to ../A_Dataset/AircraftClassification/

import os

if (os.path.exists('./dataset/Train')):
    os.rename('./dataset/Train', '../A_Dataset/AircraftClassification/Train')

if (os.path.exists('./dataset/Eval')):
    os.rename('./dataset/Eval', '../A_Dataset/AircraftClassification/Eval')

    