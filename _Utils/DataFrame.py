import pandas as pd
from typing import overload
from typing_extensions import Self

from numpy_typing import np, ax
import _Utils.Color as C
from   _Utils.Color import prntC




class DataFrame:

    array:np.float64_2d[ax.time, ax.feature]

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


# |====================================================================================================================
# | PRIVATE METHODS
# |====================================================================================================================

    def __resize__(self) -> None:
        self.array = np.resize(self.array, (self.array.shape[0]+16, self.array.shape[1]))
    def __need_resize__(self) -> bool:
        return self.len == len(self.array)
    def __eval_size__(self, l:int) -> int:
        return ((l-1) // 16 + 1) * 16



    def __insert__(self, i:int, value:np.float64_1d[ax.feature]) -> None:
        if (i > self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")

        if self.__need_resize__(): self.__resize__()

        self.array[i+1:self.len+1] = self.array[i:self.len]
        self.array[i] = value
        self.len += 1

    def __set__(self, i:int, value:np.float64_1d[ax.feature]) -> None:
        if (i >= self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")
        self.array[i] = value

    def __remove__(self, i:int) -> None:
        if (i > self.len):
            raise IndexError("Index out of range")
        if (i < 0):
            raise IndexError("Index out of range")

        self.array[i:self.len-1] = self.array[i+1:self.len]
        self.len -= 1

    def __append__(self, value:np.float64_1d[ax.feature]) -> None:
        if self.__need_resize__(): self.__resize__()
        self.array[self.len] = value
        self.len += 1


# |====================================================================================================================
# | PUBLIC METHODS
# |====================================================================================================================

    def cast(self, dtype:type) -> None:
        self.array = self.array.astype(dtype)
        self.dtype = dtype

    def copy(self) -> Self:
        df = DataFrame(self.array.shape[1])
        df.array = np.copy(self.array)
        df.len = self.len
        df.columns = self.columns.copy()
        return df


    def get_relative_loc(self, key:int)->int:
        """
        Get the index of the first element equal or greater than the key
        return len if the key is greater than all elements
        """
        left = 0
        right = self.len
        mid = (left + right)//2

        while (left < right):
            if (self.array[mid][0] < key):
                left = mid+1
            else:
                right = mid

            mid = (left + right)//2
        return mid

    def get_absolute_loc(self, key:int)->int:
        """
        Get the index of the exact key in the array
        return -1 if the key is not in the array
        """
        i = self.get_relative_loc(key)
        if (i < self.len and self.array[i][0] == key):
            return i
        return -1


    def add(self, value:np.float64_1d[ax.feature]) -> bool:
        i = self.get_relative_loc(value[0])
        if (i < self.len and self.array[i][0] == value[0]):
            return False
        self.__insert__(i, value)
        return True


    def set(self, value:np.float64_1d[ax.feature]) -> bool:
        """
        Update the value if the key is already in the array
        Insert the value if the key is not in the array
        return True if the value has been inserted
        """
        i = self.get_relative_loc(value[0])
        if (i < self.len and self.array[i][0] == value[0]):
            self.__set__(i, value)
            return False
        self.__insert__(i, value)
        return True

    def add_column(self, name:str, value:np.float64_1d[ax.time]) -> None:
        if (name in self.columns):
            self.array[:self.len, self.columns[name]] = value
            return

        self.columns[name] = len(self.columns)
        self.array = np.append(self.array, np.zeros((len(self.array), 1)), axis=1)
        self.array[:self.len, -1] = value

    def get_columns(self, names:list)-> np.float64_2d[ax.time, ax.feature]:
        return self.array[:self.len, [self.columns[name] for name in names]]
    def rename_columns(self, names:list) -> None:
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

    def before(self, key_end:int) -> "Self":
        """return a new DataFrame with all elements up to key_end (excluded)"""
        i = self.get_relative_loc(key_end)
        return self[:i]

    def until(self, key_end:int) -> "Self":
        """return a new DataFrame with all elements up to key_end (included)"""
        i = self.get_relative_loc(key_end)
        return self[:i+1]

    def index_of(self, key_end:int) -> "int":
        return self.get_relative_loc(key_end)


    def to_numpy(self) -> np.float64_2d[ax.time, ax.feature]:
        return self.array[:self.len]

    def from_numpy(self, array:np.float64_2d[ax.time, ax.feature], dtype:type=None) -> None:
        if (dtype is not None):
            if (array.dtype != dtype):
                del self.array
            self.dtype = dtype

        if self.array is None or self.array.shape[1] != len(array[0]):
            self.len = len(array)
            l = self.__eval_size__(self.len)
            self.array = np.zeros((l, array.shape[1]), dtype=self.dtype)
            self.array[:len(array)] = array
            self.columns = {str(i):i for i in range(self.array.shape[1])}
            return

        if self.array.shape[0] >= len(array):
            self.len = len(array)
            self.array[:len(array)] = array
            l = self.__eval_size__(self.len)
            self.array = self.array[:l]
            return

        # extend array
        self.len = len(array)
        l = self.__eval_size__(self.len)
        self.array = np.resize(self.array, (l, self.array.shape[1]))
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
    def __getitem__(self, key:"str \
                              |tuple[str, int]|tuple[int, int]|tuple[str, slice]|tuple[int, slice] \
                              |int|slice") \
            -> "float | np.ndarray | Self":

        if isinstance(key, str):
            return self.array[:self.len, self.columns[key]]

        if isinstance(key, tuple):
            col = self.columns[key[0]] if isinstance(key[0], str) else key[0]
            idx = key[1]
            if isinstance(idx, slice):
                start, stop, step = key.start, key.stop, key.step
                if (start == None): start = 0
                elif (start < 0): start += self.len
                if (stop == None): stop = self.len
                elif (stop < 0): stop += self.len
                return self.array[start:stop:step, col]

            elif idx < 0:
                idx += self.len
            return self.array[idx, col]

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if (start == None): start = 0
            elif(start < -self.len): start = 0
            elif (start < 0): start += self.len
            if (stop == None): stop = self.len
            elif(stop < -self.len): stop = 0
            elif (stop < 0): stop += self.len

            sub = DataFrame(self.array[start:stop:step])
            sub.columns = self.columns.copy()
            return sub
        else:
            if (key < 0): key += self.len
            return self.array[key, :]


    def __setitem__(self, key:"str \
                              |tuple[str, int]|tuple[int, int]|tuple[str, slice]|tuple[int, slice]",
                    value:"np.ndarray|float") -> None:
        if isinstance(key, str):
            self.array[:self.len, self.columns[key]] = value
            return

        if (isinstance(key, tuple)):
            col = self.columns[key[0]] if isinstance(key[0], str) else key[0]
            idx = key[1]
            if isinstance(idx, slice):
                start, stop, step = key.start, key.stop, key.step
                if (start == None): start = 0
                elif (start < 0): start += self.len
                if (stop == None): stop = self.len
                elif (stop < 0): stop += self.len
                self.array[start:stop:step, col] = value
                return

            elif idx < 0:
                idx += self.len
            self.array[idx, col] = value
            return

        raise IndexError("Invalid key type")

