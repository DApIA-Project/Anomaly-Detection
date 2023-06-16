
import numpy as np



def accuracy(y, y_):

    # y [0, 0, 1] with shape (batch_size, nb_class)
    # y_[0.2, 0.1, 0.7] with shape (batch_size, nb_class)

    y = np.argmax(y, axis=1)
    y_ = np.argmax(y_, axis=1)

    return np.sum(y == y_) / len(y) * 100

def confusionMatrix(y, y_):
    """
    y : [0, 0, 1] with shape (batch_size, nb_class)
    y_ : [0.2, 0.1, 0.7] with shape (batch_size, nb_class)
    """

    nb_class = len(y[0])

    y = np.argmax(y, axis=1)
    y_ = np.argmax(y_, axis=1)


    matrix = np.zeros((nb_class, nb_class), dtype=int)

    for i in range(len(y)):
        matrix[y[i], y_[i]] += 1

    return matrix


def nbSamplePerClass(y):
    """
    y_ : [0.2, 0.1, 0.7] with shape (batch_size, nb_class)
    """

    nb_class = len(y[0])

    y = np.argmax(y, axis=1)

    matrix = np.zeros(nb_class)

    for i in range(len(y)):
        matrix[y[i]] += 1

    return matrix

def perClassAccuracy(y, y_):
    mat = confusionMatrix(y, y_)
    # get diagonal
    diag = np.diag(mat)
    diag = diag / np.sum(mat, axis=1)
    return diag * 100