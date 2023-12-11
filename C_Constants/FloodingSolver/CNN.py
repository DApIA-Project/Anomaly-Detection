
LEARNING_RATE = 0.0003
EPOCHS = 0
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 256
DILATION_RATE = 4
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = True

HORIZON = 4


LAYERS = 2
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


INPUT_PADDING = "nan" # "valid", "last", "nan"
