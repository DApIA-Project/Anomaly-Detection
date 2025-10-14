
LEARNING_RATE = 0.0003
EPOCHS = 0
BATCH_SIZE = 32
NB_BATCH = 128

# LOAD_EP = 80

HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = True


MODEL_PADDING = "same"

LAYERS = 2
DROPOUT = 0.1
UNITS = 4
RESIDUAL = 1

USED_FEATURES = [
    "latitude", "longitude",
    "bearing", "bearing_diff",
    "distance", "distance_diff",
    "random_angle_latitude", "random_angle_longitude",# "random_angle_track"
    # "track", "track_diff"
    
    
    # "groundspeed", 
    # "vertical_rate",
    # "alert", "spi", "squawk",
    # "altitude", "geoaltitude",
    # "timestamp"
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

RESUDUAL = 1.0

# possibilities "valid", "last", "nan"
INPUT_PADDING = "nan"

