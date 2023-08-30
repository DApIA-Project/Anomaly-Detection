LEARNING_RATE = 0.00006
EPOCHS = 80
BATCH_SIZE = 128
NB_BATCH = 32

HISTORY = 128
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = False
RELATIVE_HEADING = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.02


LAYERS = 7
UNITS = 128
RESIDUAL = 2
DROPOUT = 0.3


USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    "alert", "spi", "squawk",
    "altitude", "geoaltitude",
    
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(len(USED_FEATURES))])


# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
TRAIN_WINDOW = 8
STEP = 2

