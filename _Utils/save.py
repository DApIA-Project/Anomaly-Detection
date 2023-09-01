import tensorflow as tf

import pickle as pkl



def write(path, array, level=0):
    """
    Write the array in the given path
    """

    if (isinstance(array, (tf.Variable, tf.Tensor))):
        array = array.numpy()

    file = open(path, "wb")
    pkl.dump(array, file)
    file.close()




def load(path):
    """
    Load the array from the given path
    """
    file = open(path, "rb")
    array = pkl.load(file)
    file.close()
    return array



        


