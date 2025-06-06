from _Utils.os_wrapper import os
from typing import TypeVar

from numpy_typing import np, ax
from _Utils.plotADSB import Color


T = TypeVar("T")



# |====================================================================================================================
# | Accuracy
# |====================================================================================================================

def accuracy(y:np.ndarray, y_:np.ndarray) -> float:

    # y [0, 0, 1] with shape (batch_size, nb_class)
    # y_[0.2, 0.1, 0.7] with shape (batch_size, nb_class)

    yi = np.argmax(y, axis=1)
    y_i = np.argmax(y_, axis=1)

    acc = 0
    nb = 0
    for i in range(len(y)):
        if (y_[i, y_i[i]] > 0):
            if (yi[i] == y_i[i]):
                acc += 1
            nb += 1
    return acc / nb

def binary_accuracy(y:np.float64_2d[ax.sample, ax.feature], y_:np.float64_2d[ax.sample, ax.feature])-> np.float64_1d[ax.feature]:
    acc = np.zeros(y.shape[1], dtype=np.float64)
    for i in range(len(y)):
        for f in range(len(y[i])):
            if ((y[i][f] > 0.5) == (y_[i][f] > 0.5)):
                acc[f] += 1
    return acc / len(y)
                

def accuracy_per_class(y:np.ndarray, y_:np.ndarray) -> np.ndarray:
    mat = confusion_matrix(y, y_)
    # get diagonal
    diag = np.diag(mat)
    diag = diag / np.sum(mat, axis=1)
    return diag * 100


def confidence(y_:np.ndarray)->np.ndarray:
    return np.max(y_, axis=1)


# |====================================================================================================================
# | Losses
# |====================================================================================================================

def mse(y:np.ndarray, y_:np.ndarray) -> float:
    return np.mean((y - y_)**2)



# |====================================================================================================================
# | Confusion matrix
# |====================================================================================================================

def confusion_matrix(y:np.ndarray, y_:np.ndarray) -> np.int32_2d[ax.label, ax.label]:
    """
    y : [0, 0, 1] with shape (batch_size, nb_class)
    y_ : [0.2, 0.1, 0.7] with shape (batch_size, nb_class)
    """

    nb_class = y.shape[-1]
    yi = np.argmax(y, axis=1)
    y_i = np.argmax(y_, axis=1)

    matrix = np.zeros((nb_class, nb_class), dtype=np.int32)

    for i in range(len(y)):
        if (y_[i, y_i[i]] > 0):
            matrix[yi[i], y_i[i]] += 1

    return matrix



def plot_confusion_matrix(confusion_matrix:np.ndarray, path:str, label_names:"list[str]"=None)->None:
    # plot confusion matrix
    import matplotlib.pyplot as plt

    confusion_matrix_percent = np.zeros(confusion_matrix.shape)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            confusion_matrix_percent[i, j] = confusion_matrix[i, j] / np.sum(confusion_matrix[i, :])


    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix_percent, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            s_rep = str(confusion_matrix[i, j]) + "\n" + str(round(confusion_matrix_percent[i, j]*100, 1))+"%"
            ax.text(x=j, y=i,s=s_rep, va='center', ha='center', size='xx-large')

    acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    if (label_names is None):
        label_names = [str(i) for i in range(confusion_matrix.shape[0])]

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.xticks(range(len(label_names)), label_names, fontsize=14)
    plt.yticks(range(len(label_names)), label_names, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.title('Accuracy ' + str(round(acc*100, 1))+"%", fontsize=18)
    plt.savefig(path)
    plt.close()







def plot_loss(train:np.ndarray, test:np.ndarray,
             train_avg:np.ndarray, test_avg:np.ndarray,
             type:str="loss", path:str="") -> None:

    # Plot the loss curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()

    ax.plot(np.array(train), c=Color.TRAIN, linewidth=0.5)
    ax.plot(np.array(test), c=Color.TEST, linewidth=0.5)

    label = "loss"
    if (type != None):
        label = type

    ax.plot(np.array(train_avg), c=Color.TRAIN, ls="--", label=f"train {label}")
    ax.plot(np.array(test_avg), c=Color.TEST, ls="--", label=f"test {label}")
    ax.set_ylabel(label)


    ax.set_xlabel("Epoch number")
    # x start at 1 to len(train)
    ax.set_xlim(1, len(train))
    ax.legend()
    fig.savefig(path)



def sigmoid(x:T)->T:
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x:T)->T:
    return np.log(x / (1 - x))



def moving_average_at(x:np.float64_1d, i:int, w:int) -> float:
    return np.mean(x[max(0, i-w):i+1])

def moving_average(x:np.float64_1d, w:int) -> np.float64_1d:
    r = np.zeros(len(x))
    for i in range(len(x)):
        r[i] = moving_average_at(x, i, w)
    return r