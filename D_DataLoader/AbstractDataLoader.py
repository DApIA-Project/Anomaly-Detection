
import pandas as pd
import numpy as np


# managing the data preprocessing
class DataLoader:
    """
    Abstract Class for managing the data processing

    /!\ Your custom implementation must always inherit from this class
    and implement the methods marked "MENDATORY" below

    Parameters :
    ------------

    CTX : dict
        The hyperparameters context

    path : str
        The path to the dataset

        
    Attributes :
    ------------

    xScaler: Scaler
        Scaler for the input data

    yScaler: Scaler
        Scaler for the output data

    x: np.array
        The input data

    y: np.array
        The associated output desired to be predicted

        
    x_train: np.array
        isolated x train dataset

    y_train: np.array
        isolated y train dataset

    x_test: np.array
        isolated x test dataset

    y_test: np.array
        isolated y test dataset
    

    Methods :
    ---------

    static __load_dataset__(CTX, path): x, y (MENDATORY)
        Read dataset and make all the global preprocessing
        return x, y dataset

        WARNING :
        This function is generally heavy, and if you want to make
        several training on the same dataset, USE the __get_dataset__ method
        wich save the dataset on the first call and return it on the next calls

        For evaluation, generally the dataset
        you want to use is independant. Hence, you 
        can use the __load_dataset__ method directly on the Eval folder

    __get_dataset__(path): x, y
        Return dataset with caching
        it will save the dataset on the first call and return it on the next calls
        so you MUST ONLY use it on one dataset (generally training dataset)
        

    genEpochTrain(nb_batch, batch_size): (MENDATORY)
        Generate the training batches for one epoch from x_train, y_train.
        return x, y batch in format [nb_batch, batch_size, ...]
        The output must be directly usable by the model for the training
    
    genEpochTest(): (MENDATORY)
        Generate the testing batches for one epoch from x_test, y_test.
        return x, y batch in format [nb_batch, batch_size, ...]
        The output must be directly usable by the model

    genEval(path): (MENDATORY)
        Generate the evaluation dataset from the path.
        The output format must be usable by the Trainer for the evaluation
    """


    # saves of the dataset for caching (see __get_dataset__ method)
    __dataset__ = None

    @staticmethod
    def __load_dataset__(CTX, path):
        """
        Process a dataset to generate x and y vectors

        Parameters:
        -----------

        CTX: dict
            The hyperparameters context

        path: str
            The path to the dataset

        Returns:
        --------
        x, y: np.array or list(np.array) if time series lenght is variable
            The input and output data
        """
        raise NotImplementedError("You must implement the __load_dataset__ function")
        # return x, y

    def __get_dataset__(self, path):
        """
        Do the same as __load_dataset__ but with caching.
        See __load_dataset__ for more details
        """
        if (self.CTX["CHANGED"]): DataLoader.__dataset__ = None

        if (DataLoader.__dataset__ is None):
            DataLoader.__dataset__ = self.__load_dataset__(self.CTX, path)

        return DataLoader.__dataset__



    def __init__(self, CTX, path) -> None:    
        raise NotImplementedError("Canot instantiate an abstract DataLoader")



    def genEpochTrain(self, nb_batch, batch_size):
        """
        Generate the training batches for one epoch from x_train, y_train.
        
        Parameters:
        -----------

        nb_batch: int
            The number of batch to generate

        batch_size: int
            The size of each batch

        Returns:
        --------

        x, y: np.ndarray
            Data in format [nb_batch, batch_size, ...]
            The output must be directly usable by the model for the training
        """

        raise NotImplementedError("You must implement the genEpochTrain method")


    def genEpochTest(self):
        """
        Generate the testing batches for one epoch from x_test, y_test.

        Returns:
        --------

        x, y: np.ndarray
            Data in format [nb_batch, batch_size, ...]
            The output must be directly usable by the model
        """

        raise NotImplementedError("You must implement the genEpochTest method")



    def genEval(self, path):
        """
        Generate the evaluation dataset from the path.

        Parameters:
        -----------

        path: str
            The path to the dataset

        Returns:
        --------
        Whatever you want.
        BUT: The output format must be usable by the Trainer for the evaluation
        """

        raise NotImplementedError("You must implement the genEval method")