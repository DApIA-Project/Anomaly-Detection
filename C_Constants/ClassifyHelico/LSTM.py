# ALL CONSTANTS HAS TO BE PUT HERE !!!!!!!
# These values are tracked by mlflow


TEST_SIZE = 0.2


LEARNING_RATE = 0.0003
EPOCHS = 50
BATCH_SIZE = 64
NB_BATCH = 64

NB_LSTM = 3
NB_UNITS = 128
DROPOUT = 0.3

NB_DENSE = 3
NB_NEURONS = 64


HISTORY = 64
FEATURES_IN = 9
FEATURES_OUT = 3



USED_FEATURES = [
    "lat", "lon",
    "velocity", "heading",
    "vertrate", "onground",
    "alert", "spi", "squawk",
    "baroaltitude", "geoaltitude",
    "interpolated"]
