


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



