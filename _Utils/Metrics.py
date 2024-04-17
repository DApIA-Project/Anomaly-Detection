
import numpy as np
import os

# TODO Some of these function sould be in a specialized Metrics.py file


def accuracy(y:np.ndarray, y_:np.ndarray) -> float:

    # y [0, 0, 1] with shape (batch_size, nb_class)
    # y_[0.2, 0.1, 0.7] with shape (batch_size, nb_class)

    y = np.argmax(y, axis=1)
    y_ = np.argmax(y_, axis=1)

    return np.sum(y == y_) / len(y) * 100

def mse(y:np.ndarray, y_:np.ndarray) -> float:
    return np.mean((y - y_)**2)

def confidence(y_:np.ndarray):
    return np.max(y_, axis=1)


# TODO put this function in a specialized Metrics.py file
def spoofing_training_statistics(y:np.ndarray, y_:np.ndarray) -> "tuple[float, float]":
    acc = accuracy(y, y_)
    loss = mse(y, y_)
    return acc, loss


def confusionMatrix(y:np.ndarray, y_:np.ndarray) -> np.ndarray:
    """
    y : [0, 0, 1] with shape (batch_size, nb_class)
    y_ : [0.2, 0.1, 0.7] with shape (batch_size, nb_class)
    """

    nb_class = y.shape[-1]
    y = np.argmax(y, axis=-1)
    y_ = np.argmax(y_, axis=-1)

    matrix = np.zeros((nb_class, nb_class), dtype=int)

    for i in range(len(y)):
        matrix[y[i], y_[i]] += 1

    return matrix



def plotConfusionMatrix(confusion_matrix, png, SCALER_LABELS=None):
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

    if (SCALER_LABELS is None):
        SCALER_LABELS = [str(i) for i in range(confusion_matrix.shape[0])]

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.xticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
    plt.yticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
    plt.gca().xaxis.tick_bottom()
    plt.title('Accuracy ' + str(round(acc*100, 1))+"%", fontsize=18)
    plt.savefig(png)
    plt.close()



def perClassAccuracy(y:np.ndarray, y_:np.ndarray) -> np.ndarray:
    mat = confusionMatrix(y, y_)
    # get diagonal
    diag = np.diag(mat)
    diag = diag / np.sum(mat, axis=1)
    return diag * 100



def plotLoss(train:np.ndarray, test:np.ndarray,
             train_avg:np.ndarray, test_avg:np.ndarray,
             type:str="loss", path:str=""):

    # Plot the loss curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()

    ax.plot(np.array(train), c="tab:blue", linewidth=0.5)
    ax.plot(np.array(test), c="tab:orange", linewidth=0.5)

    label = "loss"
    if (type != None):
        label = type

    ax.plot(np.array(train_avg), c="tab:blue", ls="--", label=f"train {label}")
    ax.plot(np.array(test_avg), c="tab:orange", ls="--", label=f"test {label}")
    ax.set_ylabel(label)


    ax.set_xlabel("Epoch number")
    # x start at 1 to len(train)
    ax.set_xlim(1, len(train))
    ax.legend()
    fig.savefig(path)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))



def moving_average_at(x, i, w):
    return np.mean(x[max(0, i-w):i+1])

def moving_average(x, w):
    r = np.zeros(len(x))
    for i in range(len(x)):
        r[i] = moving_average_at(x, i, w)
    return r