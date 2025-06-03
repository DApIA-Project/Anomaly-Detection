GENERATE_ARTIFACTS=True


HISTORY = 16
DILATION_RATE = 1


HORIZON = 5


# |--------------------------------------------------------------------------------------------------------------------
# | DO NOT MODIFY THE FOLLOWING VALUES AS THE MATH MODEL IS "STATIC"
# |--------------------------------------------------------------------------------------------------------------------


EPOCHS = 0 # just to generate the padding, after first run it can be let to 0
BATCH_SIZE = 64
NB_BATCH = 32

# LOAD_EP = 80


RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = False


USED_FEATURES = [
    "latitude", "longitude", "timestamp"
]


FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

PRED_FEATURES = [
    "latitude", "longitude"
]
FEATURES_OUT = len(PRED_FEATURES)
PRED_FEATURE_MAP = dict([[PRED_FEATURES[i], i] for i in range(FEATURES_OUT)])


# possibilities "valid", "last", "nan"
INPUT_PADDING = "valid"

SCALER = "dummy"
HAS_WEIGHT = False
INPUT_LEN = HISTORY // DILATION_RATE
