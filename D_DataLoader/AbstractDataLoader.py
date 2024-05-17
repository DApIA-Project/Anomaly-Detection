


import D_DataLoader.Utils as U



# managing the data preprocessing
class DataLoader:

    # saves of the dataset for caching (see __get_dataset__ method)
    __dataset__ = None


    def __load_dataset__(CTX, path):
        raise NotImplementedError("You must implement the __load_dataset__ function")

    def __get_dataset__(self, path):
        if (DataLoader.__dataset__ is None):
            DataLoader.__dataset__ = self.__load_dataset__(self.CTX, path)

        return DataLoader.__dataset__

    def __split__(self, x, y=None):
        if (y is None):
            split = U.splitDataset([x], self.CTX["TEST_RATIO"])
            return split[0][0], split[1][0]

        split = U.splitDataset([x, y], self.CTX["TEST_RATIO"])
        return split[0][0], split[0][1], split[1][0], split[1][1]


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