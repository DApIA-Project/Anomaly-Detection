
LEARNING_RATE = 0.0002
EPOCHS = 80
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 128
DILATION_RATE = 2
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = False

# "valid" do not pad convolutions
# "same" pad convolutions
MODEL_PADDING = "same"

# "valid" do not pad
# "last" duplicate the last row
# "nan" fill with nan (as the model "know" what is a nan value)
INPUT_PADDING = "valid"


DROPOUT = 0.1

ACTIVATION = "softmax"




ADD_TAKE_OFF_CONTEXT = False
ADD_MAP_CONTEXT = False
IMG_SIZE = 128
ADD_AIRPORT_CONTEXT = False

USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    # "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    "track_diff",
    "timestamp",
    # "toulouse"
]

FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])


MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4, 5], # PLANE
    6: [6, 7, 10], # SMALL
    9: [9, 12], # HELICOPTER

    0: [8, 11] # not classified
}
LABELS_OUT = len(MERGE_LABELS)-1
USED_LABELS = [k for k in MERGE_LABELS.keys() if k != 0]


