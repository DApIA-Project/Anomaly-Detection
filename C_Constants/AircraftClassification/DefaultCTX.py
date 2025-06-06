TEST_RATIO = 1.0/8.0

RELATIVE_POSITION = False
RELATIVE_TRACK = False
RANDOM_TRACK = False
TRAINING_NOISE = 0.0

MAX_BATCH_SIZE = 1024
MIN_BATCH_SIZE = 1

MAP_LAYERS = 2
USE_DYT = False
DYNAMIC_LABEL_RATIO = True

LABEL_NAMES = [
    "UNKNOWN",
    "CARGO",
    "PLANE",
    "JET",
    "TURBO PROPELLER",
    "MEDIUM",
    "LIGHT",
    "SUPER LIGHT",
    "GLIDER",
    "HELICOPTER",
    "ULM",
    "MILITARY",
    "SAMU"
]

# by default labels are not merged
MERGE_LABELS = {
    1: [1],
    2: [2],
    # 3: [3], # remove JET not enought data
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    # 8: [8], # remove GLIDER no data
    9: [9],
    10: [10],
    11: [11],
    12: [12]
}

LABELS_OUT = len(MERGE_LABELS)
USED_LABELS = [k for k in MERGE_LABELS.keys() if k != 0]



BOUNDING_BOX = [
    (43.11581, 0.72561),
    (44.07449, 2.16344)
]
BOX_CENTER = (
    (BOUNDING_BOX[0][0] + BOUNDING_BOX[1][0]) / 2,
    (BOUNDING_BOX[0][1] + BOUNDING_BOX[1][1]) / 2
)

ADD_TAKE_OFF_CONTEXT = False
ADD_MAP_CONTEXT = False
ADD_AIRPORT_CONTEXT = False

SCALER = "standard"

INPUT_PADDING = "valid"
MODEL_PADDING = "valid"


