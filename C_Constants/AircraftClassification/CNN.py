
LEARNING_RATE = 0.00006
EPOCHS = 100
BATCH_SIZE = 128
NB_BATCH = 32

HISTORY = 64
DILATION_RATE = 2
TIMESTEPS = HISTORY // DILATION_RATE
RELATIVE_POSITION = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.05

LAYERS = 2
DROPOUT = 0.3


USED_FEATURES = [
    "lat", "lon",
    "velocity", "heading",
    "vertrate", "onground",
    "alert", "spi", "squawk",
    "baroaltitude", "geoaltitude",
    "sec", "min", "hour", "day"
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(len(USED_FEATURES))])


# PAD_MISSING_TIMESTEPS = True


# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
TRAIN_WINDOW = 8
STEP = 2
