
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


def computeTimeserieVarienceRate(x):

    # x : [len]

    if (len(x) == 0):
        return 0
    if (len(x) == 1):
        return 0

    return np.mean(np.abs(np.diff(x)))


def plotConusionMatrix(png, confusion_matrix, SCALER_LABELS):
    # plot confusion matrix
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

    acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.xticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
    plt.yticks(range(len(SCALER_LABELS)), SCALER_LABELS, fontsize=14)
    plt.gca().xaxis.tick_bottom()
    plt.title('Accuracy ' + str(round(acc*100, 1))+"%", fontsize=18)
    plt.savefig(png)
    plt.close()



def plotLoss(history, history_avg):
    # Plot the loss curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()
    ax.plot(np.array(history[0]) * 100.0, c="tab:blue", linewidth=0.5)
    ax.plot(np.array(history[1]) * 100.0, c="tab:orange", linewidth=0.5)
    ax.plot(np.array(history_avg[0]) * 100.0, c="tab:blue", ls="--", label="train loss")
    ax.plot(np.array(history_avg[1]) * 100.0, c="tab:orange", ls="--", label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig("./_Artefact/loss.png")


def plotAccuracy(history, history_avg):
        # Plot the loss curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()
    ax.plot(np.array(history[2]) * 100.0, c="tab:blue", linewidth=0.5)
    ax.plot(np.array(history[3]) * 100.0, c="tab:orange", linewidth=0.5)
    ax.plot(np.array(history_avg[2]) * 100.0, c="tab:blue", ls="--", label="train loss")
    ax.plot(np.array(history_avg[3]) * 100.0, c="tab:orange", ls="--", label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    fig.savefig("./_Artefact/accuracy.png")