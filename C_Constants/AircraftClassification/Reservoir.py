
EPOCHS = 1 # reservoir fit instantly
BATCH_SIZE = 32 * 128
NB_BATCH = 1 # reservoir fit instantly


HISTORY = 64
DILATION_RATE = 2
TIMESTEPS = HISTORY // DILATION_RATE


USED_FEATURES = [
    "lat", "lon",
    "velocity", "heading",
    "vertrate", "onground",
    "alert", "spi", "squawk",
    "baroaltitude", "geoaltitude",
    
]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(len(USED_FEATURES))])




TRAIN_WINDOW = 8
STEP = 2


# Reservoir hyperparameters

n_internal_units = 450        # size of the reservoir
spectral_radius = 0.59        # largest eigenvalue of the reservoir
leak = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
connectivity = 0.25           # percentage of nonzero connections in the reservoir
input_scaling = 0.1           # scaling of the input weights
noise_level = 0.01            # noise in the reservoir state update
n_drop = 5                    # transient states to be dropped
bidir = True                  # if True, use bidirectional reservoir
circ = False                  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
dimred_method ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
n_dim = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
mts_rep = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
w_ridge_embedding = 10.0      # regularization parameter of the ridge regression

# Type of readout
readout_type = 'mlp'          # readout used for classification: {'lin', 'mlp', 'svm'}

# Linear readout hyperparameters
w_ridge = 5.0                 # regularization of the ridge regression readout

# SVM readout hyperparameters
svm_gamma = 0.005             # bandwith of the RBF kernel
svm_C = 5.0                   # regularization for SVM hyperplane

# MLP readout hyperparameters
mlp_layout = (3,3)          # neurons in each MLP layer
num_epochs = 2000             # number of epochs 
w_l2 = 0.001                  # weight of the L2 regularization
nonlinearity = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}
