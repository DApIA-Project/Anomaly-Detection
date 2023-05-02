
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# managing the data preprocessing
class DataLoader:
    """
    Class for managing the data processing

    Attributes :
    ------------

    xScaler : Scaler
    yScaler : Scaler

    x : np.array
        represent the input data
    y : np.array
        represent the output data

    TEST_SIZE : float
        represent the size of the test set
    x_train : np.array
    y_train : np.array
    x_test : np.array
    y_test : np.array
    


    Methods :
    ---------

    __load_dataset__(CTX, path):
        return from a path the x and y dataset in the format of your choice
        define here the preprocessing you want to do
        no need to call directly this method, use instead __get_dataset__ with a cache

    __get_dataset__(self, path):
        return x and y + use a cache to avoid reloading the dataset
    
    __init__(self, CTX, path):
        Constructor of the class, generate x_train, y_train, x_test, y_test
        3 tasks :
            1. load dataset
            2. create and fit the scalers on x and y
            3. split the data into train and test

    genEpochTrain(nb_batch, batch_size):
        return x, y batch list in format [nb_batch, batch_size, ...]
        The output must be directly usable by the model for the training
    
    genEpochTest():
        return x, y selfbatch list in format [nb_row, ...]
        The output must be directly usable by the model for the testing
    """



    __train_dataset__x:np.array = None
    __train_dataset__y:np.array = None

    @staticmethod
    def __load_dataset__(CTX, path):
        """
        process a dataframe to generate x and y vectors
        """
        raise NotImplementedError("You must implement the __load_dataset__ function")
        # return x, y

    def __get_dataset__(self, path):
        """ load dataset on the first call, return it on the next calls """

        if (DataLoader.__train_dataset__x is None or DataLoader.__train_dataset__y is None):
            DataLoader.__train_dataset__x , DataLoader.__train_dataset__y = self.__load_dataset__(self.CTX, path)

        return DataLoader.__train_dataset__x , DataLoader.__train_dataset__y



    def __init__(self, CTX, path) -> None:    
        raise NotImplementedError("Canot instantiate abstract class DataLoader")



    def genEpochTrain(self, nb_batch, batch_size):
        """
        Generate the x train and y train batches.
        The returned format is usually [nb_batch, batch_size, ...]
        But it can be different depending on you're model implementation
        The output must be directly usable by the model for the training

        Called between each epoch by the trainer
        """

        raise NotImplementedError("You must implement the genEpochTrain method")


    def genEpochTest(self):
        """
        Generate the x test and y test dataset.
        The returned format is usually [nb_row, ...]
        But it can be different depending on you're model implementation
        The output must be directly usable by the model for the testing

        Called between each epoch by the trainer
        """

        raise NotImplementedError("You must implement the genEpochTest method")



    def genEval(self, path):
        """
        Generate the x eval and y eval batches.

        called by the trainer at the end, to get the final evaluation in the real world condition
        """

        raise NotImplementedError("You must implement the genEval method")