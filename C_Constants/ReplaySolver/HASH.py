
LEARNING_RATE = 0
EPOCHS = 1
BATCH_SIZE = None
NB_BATCH = 1


HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = False
RELATIVE_TRACK = False
RANDOM_TRACK = False



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


INPUT_PADDING = "valid" # "valid", "last", "nan"
