import numpy as np


np.set_printoptions(
    formatter={'float': lambda x: "{0:0.6f}".format(x)},
    linewidth=np.inf,
    threshold=128 * 128 * 128)
