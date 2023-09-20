import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])
    except RuntimeError as e:
        print(e)

    # hide rocm warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'



#############################
# Choose your model here    #
#############################
model = "CNN"
#############################


if model== "CNN":
    import G_Main.AircraftClassification.exp_CNN as CNN
    CNN.__main__()

elif model== "LSTM":
    import G_Main.AircraftClassification.exp_LSTM as LSTM
    LSTM.__main__()

elif model== "Transformer":
    import G_Main.AircraftClassification.exp_Transformer as Transformer
    Transformer.__main__()

elif model== "Reservoir":
    import G_Main.AircraftClassification.exp_Reservoir as Reservoir
    Reservoir.__main__()


# restore rocm warnings
if gpus:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'




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

# print(dl.xScaler.mins)
# print(dl.xScaler.maxs)