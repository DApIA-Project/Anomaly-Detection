
LEARNING_RATE = 0.00005
EPOCHS = 0
BATCH_SIZE = 64
NB_BATCH = 32


HISTORY = 128
DILATION_RATE = 2
INPUT_LEN = HISTORY // DILATION_RATE


LAYERS = 2
DROPOUT = 0.3


HEAD_SIZE = 6
NUM_HEADS = 2
FF_DIM = 64


USED_FEATURES = [
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    "alert", "spi", "squawk",
    "altitude", "geoaltitude",

]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])



MERGE_LABELS = { # no merge by default
    2: [1, 2, 3, 4, 5], # PLANE
    6: [6, 7, 10], # SMALL
    9: [9, 12], # HELICOPTER

    0: [8, 11] # not classified
}
LABELS_OUT = len(MERGE_LABELS) - 1
USED_LABELS = [k for k in MERGE_LABELS.keys() if k != 0]


