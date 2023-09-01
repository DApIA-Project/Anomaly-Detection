import tensorflow as tf

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
model = "CNN_img"
#############################


if model== "CNN":
    import G_Main.AircraftClassification.exp_CNN as CNN
    CNN.__main__()

if model== "CNN_img":
    import G_Main.AircraftClassification.exp_CNN_img as CNN_img
    CNN_img.__main__()

if model== "LSTM_img":
    import G_Main.AircraftClassification.exp_LSTM_img as LSTM_img
    LSTM_img.__main__()

elif model== "LSTM":
    import G_Main.AircraftClassification.exp_LSTM as LSTM
    LSTM.__main__()

elif model== "Transformer":
    import G_Main.AircraftClassification.exp_Transformer as Transformer
    Transformer.__main__()

elif model== "Transformer_img":
    import G_Main.AircraftClassification.exp_Transformer_img as Transformer
    Transformer.__main__()

elif model== "Reservoir":
    import G_Main.AircraftClassification.exp_Reservoir as Reservoir
    Reservoir.__main__()


# restore rocm warnings
if gpus:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
