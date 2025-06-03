GENERATE_ARTIFACTS=False

LEARNING_RATE = 0.0003
EPOCHS = 80
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = True
RANDOM_TRACK = False

HORIZON = 5


DEC_LEN = 1
D_MODEL = 64
N_HEADS = 8
E_LAYERS = 2
D_LAYERS = 2
D_FF = 512
ACTIVATION = "gelu" # TODO check !!!
EMBED = "timeF" # "fixed", "learned"
EMBED_IN = 1
FACTOR = 1
DROPOUT = 0.3




USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate",
    "distance", "bearing",
    "distance_diff", "bearing_diff",
    # "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    "timestamp", "pad", "pred_distance"
]

FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

ENBED_FEATURES = [
    "timestamp"
]


PRED_FEATURES = [
    "latitude", "longitude"
]
FEATURES_OUT = len(PRED_FEATURES)
PRED_FEATURE_MAP = dict([[PRED_FEATURES[i], i] for i in range(FEATURES_OUT)])


# "valid", "last", "nan"
INPUT_PADDING = "nan"


