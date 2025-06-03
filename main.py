import warnings
warnings.filterwarnings("ignore")
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
# algo = "FloodingSolver"
# algo = "ReplaySolver"
# algo = "ReplaySolver"
algo = "InterpolationDetector"

model = "encoder"

###################################
argv = sys.argv
if ("-ui" in argv):
    from _Utils.DebugGui import activate
    activate()
    argv.remove("-ui")

if (len(sys.argv) >= 3):
    algo =argv[1]
    model =argv[2]
    
elif (len(sys.argv) >= 2):
    model =argv[1]


################################### 935329

if ("_" in model):
    model.split("_")


if (algo == "AircraftClassification"):

    if model == "CNN1":
        import G_Main.AircraftClassification.exp_CNN_V1 as CNN
        CNN.__main__()
    
    if model == "CNN2":
        import G_Main.AircraftClassification.exp_CNN_V2 as CNN
        CNN.__main__()
    
    elif model == "encoder":
        import G_Main.AircraftClassification.exp_encoder as encoder
        encoder.__main__()
    
    elif model == "FCN":
        import G_Main.AircraftClassification.exp_FCN as FCN
        FCN.__main__()
        
    elif model == "inception":
        import G_Main.AircraftClassification.exp_inception as inception
        inception.__main__()
        
    elif model == "LSTM":
        import G_Main.AircraftClassification.exp_LSTM as LSTM
        LSTM.__main__()
        
    elif model == "MCDCNN":
        import G_Main.AircraftClassification.exp_MCDCNN as MCDCNN
        MCDCNN.__main__()
        
    elif model == "MLP":
        import G_Main.AircraftClassification.exp_MLP as MLP
        MLP.__main__()
        
    elif model == "reservoir":
        import G_Main.AircraftClassification.exp_Reservoir as Reservoir
        Reservoir.__main__()

    elif model == "resnet":
        import G_Main.AircraftClassification.exp_resnet as resnet
        resnet.__main__()
        
    elif model == "TLeNet":
        import G_Main.AircraftClassification.exp_TLeNet as TLeNet
        TLeNet.__main__()

        
    elif model == "Transformer":
        import G_Main.AircraftClassification.exp_Transformer as Transformer
        Transformer.__main__()

elif (algo == "FloodingSolver"):
    
    if ("CatBoost" in model):
        import G_Main.FloodingSolver.exp_CatBoost as CatBoost
        CatBoost.__main__()
        
    if ("CNN" in model):
        import G_Main.FloodingSolver.exp_CNN as CNN
        CNN.__main__()
        
    if ("encoder" in model):
        import G_Main.FloodingSolver.exp_encoder as encoder
        encoder.__main__()

    if ("FCN" in model):
        import G_Main.FloodingSolver.exp_FCN as FCN
        FCN.__main__()
        
    if ("inception" in model):
        import G_Main.FloodingSolver.exp_inception as inception
        inception.__main__()

    if ("LSTM" in model):
        import G_Main.FloodingSolver.exp_LSTM as LSTM
        LSTM.__main__()
    
    if ("Math" in model):
        import G_Main.FloodingSolver.exp_Math as Math
        Math.__main__()
        
    if ("MCDCNN" in model):
        import G_Main.FloodingSolver.exp_MCDCNN as MCDCNN
        MCDCNN.__main__()
        
    if ("MLP" in model):
        import G_Main.FloodingSolver.exp_MLP as MLP
        MLP.__main__()
    
    if ("reservoir" in model):
        import G_Main.FloodingSolver.exp_Reservoir as Reservoir
        Reservoir.__main__()
        
    if ("resnet" in model):
        import G_Main.FloodingSolver.exp_resnet as resnet
        resnet.__main__()
        
    if ("TLeNet" in model):
        import G_Main.FloodingSolver.exp_TLeNet as TLeNet
        TLeNet.__main__()    

    if ("Transformer" in model):
        import G_Main.FloodingSolver.exp_Transformer as Transformer
        Transformer.__main__()






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




elif (algo == "InterpolationDetector"):

    if (model == "CNN"):
        import G_Main.InterpolationDetector.exp_CNN as CNN
        CNN.__main__()
        
    if (model == "encoder"):
        import G_Main.InterpolationDetector.exp_encoder as encoder
        encoder.__main__()
        
    if (model == "FCN"):
        import G_Main.InterpolationDetector.exp_FCN as FCN
        FCN.__main__()
        
    if (model == "inception"):
        import G_Main.InterpolationDetector.exp_inception as inception
        inception.__main__()
        
    if (model == "LSTM"):
        import G_Main.InterpolationDetector.exp_LSTM as LSTM
        LSTM.__main__()
        
    if (model == "MCDCNN"):
        import G_Main.InterpolationDetector.exp_MCDCNN as MCDCNN
        MCDCNN.__main__()
        
    if (model == "MLP"):
        import G_Main.InterpolationDetector.exp_MLP as MLP
        MLP.__main__()
        
    if (model == "reservoir"):
        import G_Main.InterpolationDetector.exp_Reservoir as Reservoir
        Reservoir.__main__()
        
    if (model == "resnet"):
        import G_Main.InterpolationDetector.exp_resnet as resnet
        resnet.__main__()
        
    if (model == "TLeNet"):
        import G_Main.InterpolationDetector.exp_TLeNet as TLeNet
        TLeNet.__main__()
        
    if (model == "Transformer"):
        import G_Main.InterpolationDetector.exp_Transformer as Transformer
        Transformer.__main__()

else:
    print("Unknown algo")
    exit(1)
