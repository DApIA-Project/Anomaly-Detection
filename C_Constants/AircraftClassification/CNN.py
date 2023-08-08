
LEARNING_RATE = 0.00006
EPOCHS = 100
BATCH_SIZE = 128
NB_BATCH = 32

HISTORY = 64
DILATION_RATE = 2
TIMESTEPS = HISTORY // DILATION_RATE

LAYERS = 2
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




MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4], # PLANE
    6: [5, 6, 7, 8, 10], # MEDIUM
    9: [9], # HELICOPTER
    11: [11] # military
}
FEATURES_OUT = len(MERGE_LABELS)


# for training a batch concerning a single aircraft flight
# the step is the jump between two consecutive batches
# each element of a batch start at [t, t+STEP, t+2*STEP, ...]
TRAIN_WINDOW = 8
STEP = 2
