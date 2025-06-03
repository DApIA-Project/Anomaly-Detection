
LEARNING_RATE = 0.0003
EPOCHS = 80
BATCH_SIZE = 32
NB_BATCH = 128

# LOAD_EP = 80

HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = True


DROPOUT = 0.1
UNITS = 128

USED_FEATURES = [
    "latitude", "longitude",
    "bearing", "bearing_diff",
    "distance", "distance_diff",
    "random_angle_latitude", "random_angle_longitude",
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

RESUDUAL = 1.0

# possibilities "valid", "last", "nan"
INPUT_PADDING = "nan"

