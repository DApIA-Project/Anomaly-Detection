import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import _Utils.numpy_print



#############################
# Choose your model here    #
#############################
# algo = "AircraftClassification"
algo = "AircraftClassification"
model = "CNN2"
#############################

if (algo == "AircraftClassification"):

    if model== "CNN":
        import G_Main.AircraftClassification.exp_CNN as CNN
        CNN.__main__()

    if model== "CNN2":
        import G_Main.AircraftClassification.exp_CNN2 as CNN2
        CNN2.__main__()

    elif model== "LSTM":
        import G_Main.AircraftClassification.exp_LSTM as LSTM
        LSTM.__main__()

    elif model== "Transformer":
        import G_Main.AircraftClassification.exp_Transformer as Transformer
        Transformer.__main__()

    elif model== "Reservoir":
        import G_Main.AircraftClassification.exp_Reservoir as Reservoir
        Reservoir.__main__()

if (algo == "FloodingSolver"):
    if (model == "CNN"):
        import G_Main.FloodingSolver.exp_CNN as CNN
        CNN.__main__()


# # restore rocm warnings
# if gpus:
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'




# from D_DataLoader.AircraftClassification.DataLoader import DataLoader
# from _Utils.module import module_to_dict
# import C_Constants.AircraftClassification.CNN as CTX
# import C_Constants.AircraftClassification.DefaultCTX as default_CTX

# CTX = module_to_dict(CTX)
# default_CTX = module_to_dict(default_CTX)
# for param in default_CTX:
#     if (param not in CTX):
#         CTX[param] = default_CTX[param]
# dl = DataLoader(CTX, "./A_Dataset/AircraftClassification/Train")
# x_train, y_train = dl.genEpochTrain(8, 4)
