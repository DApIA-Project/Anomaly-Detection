
LEARNING_RATE = 0.0003
EPOCHS = 60
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 32
DILATION_RATE = 2
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = True

HORIZON = 3


LAYERS = 3
DROPOUT = 0.3


USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    # "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    "timestamp","pad"
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

PRED_FEATURES = [
    "latitude", "longitude"
]
FEATURES_OUT = len(PRED_FEATURES)
PRED_FEATURE_MAP = dict([[PRED_FEATURES[i], i] for i in range(FEATURES_OUT)])

RESUDUAL = 0.5

# possibilities "valid", "last", "nan"
INPUT_PADDING = "nan"
