from typing import TypeVar

import D_DataLoader.Utils as U


T1 = TypeVar('T1')
T2 = TypeVar('T2')

# managing the data preprocessing
class DataLoader:

    # saves of the dataset for caching (see __get_dataset__ method)
    __dataset__ = None

    def __load_dataset__(self, CTX:dict, path:str) -> object:
        raise NotImplementedError("You must implement the __load_dataset__ function")



    def __get_dataset__(self, path:str) -> object:
        if (DataLoader.__dataset__ is None):
            DataLoader.__dataset__ = self.__load_dataset__(self.CTX, path)

        return DataLoader.__dataset__



    def __split__(self, x:T1, y:T2=None, size:int=None) -> "T1|tuple[T1, T2]":
        if (size is None):
            size = self.CTX["TEST_RATIO"]

        if (y is None):
            split = U.splitDataset([x], size)
            return split[0][0], split[1][0]
        split = U.splitDataset([x, y], size)
        return split[0][0], split[0][1], split[1][0], split[1][1]



    def __init__(self, CTX:dict, path:str) -> None:
        raise NotImplementedError("Canot instantiate an abstract DataLoader")



    def get_train(self) -> object:
        raise NotImplementedError("You must implement the get_train method")



    def get_test(self) -> object:
        raise NotImplementedError("You must implement the get_test method")


