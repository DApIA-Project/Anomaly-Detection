
LEARNING_RATE = 0.0003
<<<<<<< HEAD
EPOCHS = 80
=======
EPOCHS = 0
>>>>>>> master
BATCH_SIZE = 32
NB_BATCH = 128

# LOAD_EP = 80

HISTORY = 32
DILATION_RATE = 1
INPUT_LEN = HISTORY // DILATION_RATE

RELATIVE_POSITION = True
RELATIVE_TRACK = False
RANDOM_TRACK = True

LAYERS = 2
DROPOUT = 0.1
UNITS = 64

R_UNITS = 1000
LR = 1
SR = 0.85
INPUT_CONNECTIVITY = 0.01
RC_CONNECTIVITY = 0.01

USED_FEATURES = [
    "latitude", "longitude",
    "bearing", "bearing_diff",
    "distance", "distance_diff",
    "random_angle_latitude", "random_angle_longitude",
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

RESUDUAL = 1.0

# possibilities "valid", "last", "nan"
INPUT_PADDING = "nan"

<<<<<<< HEAD
=======
THRESHOLD = 0.80
LOSS_MOVING_AVERAGE = 16 # kernel size for moving average

>>>>>>> master
