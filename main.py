import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import sys

import numpy as np
np.set_printoptions(linewidth=200)



##################################
# Choose your model here         #
# You can also use               #
# python main.py <model>         #
# python main.py <algo> <model>  #
##################################
# algo = "AircraftClassification"
# model = "CNN2"

# algo = "FloodingSolver"
# model = "LSTM"

# algo = "ReplaySolver"
# model = "HASH"

algo = "TrajectorySeparator"
model = "GEO"
###################################
argv = sys.argv
if ("-ui" in argv):
    from _Utils.DebugGui import activate
    activate()
    argv.remove("-ui")

if (len(sys.argv) >= 2):
    model =argv[1]

elif (len(sys.argv) >= 3):
    algo =argv[1]
    model =argv[2]





if (algo == "AircraftClassification"):
    if model == "CNN1":
        import G_Main.AircraftClassification.exp_CNN1 as CNN1
        CNN1.__main__()

    if model == "CNN2":
        import G_Main.AircraftClassification.exp_CNN2 as CNN2
        CNN2.__main__()

    elif model == "LSTM":
        import G_Main.AircraftClassification.exp_LSTM as LSTM
        LSTM.__main__()

    elif model == "Transformer":
        import G_Main.AircraftClassification.exp_Transformer as Transformer
        Transformer.__main__()

    elif model == "Reservoir":
        import G_Main.AircraftClassification.exp_Reservoir as Reservoir
        Reservoir.__main__()



elif (algo == "FloodingSolver"):
    if (model == "CNN"):
        import G_Main.FloodingSolver.exp_CNN as CNN
        CNN.__main__()

    elif (model == "LSTM"):
        import G_Main.FloodingSolver.exp_LSTM as LSTM
        LSTM.__main__()


elif (algo == "ReplaySolver"):
    if (model == "HASH"):
        import G_Main.ReplaySolver.exp_HASH as HASH
        HASH.__main__()

elif (algo == "TrajectorySeparator"):
    if (model == "GEO"):
        import G_Main.TrajectorySeparator.exp_GEO as GEO
        GEO.__main__()
    if (model == "DEV"):
        import G_Main.TrajectorySeparator.exp_DEV as DEV
        DEV.__main__()
