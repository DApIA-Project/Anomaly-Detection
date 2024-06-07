
EPOCHS = 1

WHILDCARD_LIMIT = 5

HISTORY = 32
INPUT_LEN = HISTORY

USED_FEATURES = [
    "latitude", "longitude"
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

# "valid", "last", "nan"
INPUT_PADDING = "nan"
