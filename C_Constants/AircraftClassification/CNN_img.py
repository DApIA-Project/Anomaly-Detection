
LEARNING_RATE = 0.00004
EPOCHS = 80
BATCH_SIZE = 128
NB_BATCH = 32


HISTORY = 128
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_HEADING = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.0



LAYERS = 2
DROPOUT = 0.3


USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    # 
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(len(USED_FEATURES))])

ADD_TAKE_OFF_CONTEXT = False

MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4], # PLANE
    5: [5,], # MEDIUM
    9: [9,], # HELICOPTER
    10: [6, 7, 10], # ULM
    12: [12,], # SAMU
    11: [11,], # military

    0: [  8] # not classified
}
FEATURES_OUT = len(MERGE_LABELS)-1

# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
# TRAIN_WINDOW = 8
# STEP = 2



IMG_SIZE = 128

