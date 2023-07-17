
LEARNING_RATE = 0.00006
EPOCHS = 80
BATCH_SIZE = 128
NB_BATCH = 32


HISTORY = 64
DILATION_RATE = 1
TIMESTEPS = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_HEADING = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.0

# PAD_MISSING_TIMESTEPS = 



LAYERS = 2
DROPOUT = 0.1


USED_FEATURES = [
    "lat", "lon",
    "velocity", "heading",
    "vertrate", "onground",
    "alert", "spi", "squawk",
    "baroaltitude", "geoaltitude",
    # 
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(len(USED_FEATURES))])



# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
TRAIN_WINDOW = 8
STEP = 2



IMG_SIZE = 128
