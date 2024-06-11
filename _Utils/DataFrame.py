


from _Utils.numpy import np, ax
import pandas as pd

from typing import overload
from typing_extensions import Self

class DataFrame:
    @overload
    def __init__(self, len:int) -> None: ...
    @overload
    def __init__(self, df:pd.DataFrame) -> None: ...
    @overload
    def __init__(self, array:np.ndarray) -> None: ...

    def __init__(self, arg:"int|pd.DataFrame|np.ndarray", dtype:type=np.float64) -> None:
        self.array = None
        self.dtype = dtype

        if (isinstance(arg, int)):
            self.array = np.zeros((16, arg), dtype=dtype)
            self.len = 0
            self.columns = {str(i):i for i in range(arg)}

        elif (isinstance(arg, pd.DataFrame)):
            cols = [c for c in arg.columns if arg[c].dtype != object]
            self.from_numpy(arg[cols].to_numpy())
            self.columns = {cols[i]:i for i in range(len(cols))}

        elif (isinstance(arg, np.ndarray)):
            self.from_numpy(arg)
            self.columns = {str(i):i for i in range(arg.shape[1])}


    def copy(self) -> Self:
        df = DataFrame(self.array.shape[1])
        df.array = self.array.copy()
        df.len = self.len
        df.columns = self.columns.copy()
        return df

    def cast(self, dtype:type) -> None:
        self.array = self.array.astype(dtype)
        self.dtype = dtype


    def __insert__(self, i, value):
        if (i > self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")

        if (self.len == len(self.array)):
            self.array = np.resize(self.array, (self.len*2, self.array.shape[1]))

        self.array[i+1:self.len+1] = self.array[i:self.len]
        self.array[i] = value
        self.len += 1

    def __set__(self, i, value):
        if (i >= self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")
        self.array[i] = value

    def __remove__(self, i):
        if (i > self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")

        self.array[i:self.len-1] = self.array[i+1:self.len]
        self.len -= 1

    def __append__(self, value):
        if (type(value) == np.ndarray):
            if (len(value.shape) == 2):

                new_len = self.len + value.shape[0]
                l = 2**int(np.ceil(np.log2(new_len)))
                self.array = np.resize(self.array, (l, self.array.shape[1]))
                self.array[self.len:new_len] = value
                self.len = new_len
                return


        if (self.len == len(self.array)):
            self.array = np.resize(self.array, (self.len*2, self.array.shape[1]))
        self.array[self.len] = value
        self.len += 1


    def add(self, value):
        key = value[0]
        left = 0
        right = self.len
        mid = (left + right)//2

        while (left < right):
            if (self.array[mid][0] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2

        if (mid < self.len and self.array[mid][0] == key):
            return False
        self.__insert__(mid, value)
        return True

    def set(self, value):
        key = value[0]
        left = 0
        right = self.len
        mid = (left + right)//2

        while (left < right):
            if (self.array[mid][0] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2

        if (mid < self.len and self.array[mid][0] == key):
            self.__set__(mid, value)
            return False
        self.__insert__(mid, value)
        return True

    def add_column(self, name, value):
        if (name in self.columns):
            self.array[:self.len, self.columns[name]] = value
            return

        self.columns[name] = len(self.columns)
        self.array = np.append(self.array, np.zeros((len(self.array), 1)), axis=1)
        self.array[:self.len, -1] = value

    def setColumValue(self, name:str, i:slice, value:float):
        self.array[i, self.columns[name]] = value

    def getColumns(self, names:list)-> np.float64_2d[ax.time, ax.feature]:
        return self.array[:self.len, [self.columns[name] for name in names]]
    def setColums(self, names:list):
        self.columns = {name:i for i, name in enumerate(names)}

    def get(self, key) -> np.float64_1d[ax.feature]:
        left = 0
        right = self.len
        mid = (left + right)//2

        while (left < right):
            if (self.array[mid][0] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2

        if (mid >= self.len or self.array[mid][0] != key):
            return None
        return self.array[mid]

    def subset(self, key_end):
        left = 0
        right = self.len
        mid = (left + right)//2

        while (left < right):
            if (self.array[mid][0] < key_end):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2

        if (mid >= self.len or self.array[mid][0] != key_end):
            return None

        sub_df = DataFrame(self.array.shape[1])
        sub_df.array = self.array[:mid+1]
        sub_df.len = len(sub_df.array)
        sub_df.columns = self.columns.copy()
        return sub_df


    def to_numpy(self) -> np.float64_2d[ax.time, ax.feature]:
        return self.array[:self.len]

    def from_numpy(self, array:np.float64_2d[ax.time, ax.feature], dtype:type=None) -> None:
        if (dtype is not None):
            self.dtype = dtype

        if self.array is None:
            l = len(array)
            l = 2**int(np.ceil(np.log2(l)))
            self.len = len(array)
            self.array = np.zeros((l, len(array[0])), dtype=self.dtype)
            self.array[:len(array)] = array
            return

        if (self.array.shape[1] != len(array[0])):
            del self.array
            self.from_numpy(array)

        if self.array.shape[0] >= len(array):
            self.len = len(array)
            self.array[:len(array)] = array
            return

        # extend array
        l = len(array)
        l = 2**int(np.ceil(np.log2(l)))
        self.array = np.resize(self.array, (l, self.array.shape[1]))
        self.len = len(array)
        self.array[:len(array)] = array

    def clear(self) -> None:
        self.len = 0
        self.array = np.zeros((16, self.array.shape[1]), dtype=self.dtype)



    def __str__(self) -> str:
        return str(self.array[:self.len])
    def __repr__(self) -> str:
        return str(self.array[:self.len])
    def __len__(self) -> int:
        return self.len
    # [] operator
    def __getitem__(self, key:"int|str|slice") -> np.ndarray:
        if isinstance(key, str):
            return self.array[:self.len, self.columns[key]]

        value = self.array[:self.len][key]

        if isinstance(value, np.ndarray) and len(value.shape) == 2:
            sub = DataFrame(value)
            sub.columns = self.columns.copy()
            return sub

        return value

    def __setitem__(self, key:"int|str|slice", value:np.ndarray) -> None:
        if isinstance(key, str):
            self.setColumValue(key, slice(0, self.len), value)
        if (isinstance(key, tuple)):
            self.setColumValue(key[0], key[1], value)