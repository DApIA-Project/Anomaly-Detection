

import os

if (os.path.exists('../A_Dataset/AircraftClassification/Train')):
    os.rename('../A_Dataset/AircraftClassification/Train', './dataset/Train')

# if (os.path.exists('../A_Dataset/AircraftClassification/Eval')):
#     os.rename('../A_Dataset/AircraftClassification/Eval', './dataset/Eval')

    