INT_MAX = 2**31 - 1
 # undetermined (depending on the size of the dataset the number of epochs will be adjusted)
EPOCHS =  0 #INT_MAX

MAX_BATCH_SIZE = 512
MAX_NB_BATCH = 512

WHILDCARD_LIMIT = 5
MIN_DIVERSITY = 5

HISTORY = 32
INPUT_LEN = HISTORY

USED_FEATURES = [
    "fingerprint"
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

# "valid", "last", "nan"
INPUT_PADDING = "nan"
