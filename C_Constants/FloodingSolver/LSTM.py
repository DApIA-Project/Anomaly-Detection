
LEARNING_RATE = 0.0003
EPOCHS = 60
BATCH_SIZE = 64
NB_BATCH = 32

# LOAD_EP = 80

HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = True
RANDOM_TRACK = False

HORIZON = 3

THRESHOLD = 25.0

BLOCKS = 2
LAYERS = 1
RESUDUAL = 1

UNITS = 128
DROPOUT = 0.2
ACTIVATION = "linear"



USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate",
    # "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    "timestamp", "pred_distance"
]

FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

PRED_FEATURES = [
    "latitude", "longitude"
]
FEATURES_OUT = len(PRED_FEATURES)
PRED_FEATURE_MAP = dict([[PRED_FEATURES[i], i] for i in range(FEATURES_OUT)])


# possibilities "valid", "last", "nan"
INPUT_PADDING = "nan"

