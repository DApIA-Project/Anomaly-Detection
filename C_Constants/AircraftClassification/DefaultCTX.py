
TEST_RATIO = 0.1


RELATIVE_POSITION = False
RELATIVE_HEADING = False
RANDOM_HEADING = False
TRAINING_NOISE = 0.0


LABEL_FILTER = [
    1, # CARGO
    2, # PLANE
    3, # JET
    4, # TURBO PROPELLER
    5, # MEDIUM
    6, # LIGHT
    7, # SUPER LIGHT
    8, # GLIDER
    9,  # HELICOPTER
    10,  # ULM
    11 # military 
]

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
    "MILITARY"
]

MERGE_LABELS = { # no merge by default
    1: [1],
    2: [2],
    # 3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    # 8: [8],
    9: [9],
    10: [10],
    11: [11]
}

FEATURES_OUT = len(MERGE_LABELS)


BOUNDING_BOX = [
    (43.11581, 0.72561),
    (44.07449, 2.16344)
]
BOX_CENTER = (
    (BOUNDING_BOX[0][0] + BOUNDING_BOX[1][0]) / 2,
    (BOUNDING_BOX[0][1] + BOUNDING_BOX[1][1]) / 2
)


PAD_MISSING_TIMESTEPS = False

ACTIVATION = "softmax"
