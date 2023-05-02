# ALL CONSTANTS HAS TO BE PUT HERE !!!!!!!
# These values are tracked by mlflow


TEST_SIZE = 0.2


LEARNING_RATE = 0.0003
EPOCHS = 70
BATCH_SIZE = 256
NB_BATCH = 64


DROPOUT = 0.3

NB_DENSE = 3
NB_NEURONS = 64


HISTORY = 128
FEATURES_IN = 9
FEATURES_OUT = 3



USED_FEATURES = [
    "lat", "lon",
    "velocity", "heading",
    "vertrate", "onground",
    "alert", "spi", "squawk",
    "baroaltitude", "geoaltitude",
    "interpolated"]
