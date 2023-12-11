
LEARNING_RATE = 0.0002
EPOCHS = 80
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 64*3
DILATION_RATE = 3
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = False
TRAINING_NOISE = 0.1

# "valid" do not pad convolutions
# "same" pad convolutions
MODEL_PADDING = "same"

# "valid" do not pad convolutions
# "last" duplicate the last row
# "nan" fill with nan (as the model "know" what is a nan value)
INPUT_PADDING = "nan" 
# TODO study padding, maybe, it is preprocessed in batchPreProcess()


LAYERS = 1
DROPOUT = 0.1
SKIP_CONNECTION = 0.1


USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", #"onground",
    # "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    # 
    "relative_track", "timestamp",
    # "selected"
]

ADD_TAKE_OFF_CONTEXT = True
ADD_MAP_CONTEXT = True


FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])


MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4, 5], # PLANE
    6: [6, 7, 10], # SMALL
    9: [9, 12], # HELICOPTER

    0: [8, 11] # not classified
}
FEATURES_OUT = len(MERGE_LABELS)-1
USED_LABELS = [k for k in MERGE_LABELS.keys() if k != 0]

# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
# TRAIN_WINDOW = 8
# STEP = 2

ACTIVATION = "sigmoid"


IMG_SIZE = 128

NB_TRAIN_SAMPLES = 2

