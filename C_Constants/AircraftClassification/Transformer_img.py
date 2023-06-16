
LEARNING_RATE = 0.0003
EPOCHS = 50
BATCH_SIZE = 128
NB_BATCH = 32


HISTORY = 512
DILATION_RATE = 4
TIMESTEPS = HISTORY // DILATION_RATE
RELATIVE_POSITION = True
RELATIVE_HEADING = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.05

LAYERS = 2
DROPOUT = 0.3


HEAD_SIZE = 6
NUM_HEADS = 2
FF_DIM = 64


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



# PAD_MISSING_TIMESTEPS = True TODO



# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
TRAIN_WINDOW = 8
STEP = 2



IMG_SIZE = 128
