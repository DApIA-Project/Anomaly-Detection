
from typing import Generic, TypeVar, overload

V = TypeVar("V")


class OrderedDict(Generic[V]):

    def __init__(self)->None:
        self.k = []
        self.v = []

        self.__get_key_loc_cache_key__ = None
        self.__get_key_loc_cache_loc__ = None

    def __get_key_loc__(self, key:int) -> int:
        if (self.__get_key_loc_cache_key__ == key):
            return self.__get_key_loc_cache_loc__

        a = 0
        b = len(self.k) - 1
        while (a <= b):
            m = (a + b) // 2
            if (self.k[m] == key):
                return m
            elif (self.k[m] < key):
                a = m + 1
            else:
                b = m - 1
        self.__get_key_loc_cache_key__ = key
        self.__get_key_loc_cache_loc__ = a
        return a

    def __setitem__(self, key:int, value) -> None:
        loc = self.__get_key_loc__(key)
        if (loc < len(self.k) and self.k[loc] == key):
            self.v[loc] = value
        else:
            self.k.insert(loc, key)
            self.v.insert(loc, value)

    def __getitem__(self, key:int) -> V:
        loc = self.__get_key_loc__(key)
        if (loc < len(self.k) and self.k[loc] == key):
            return self.v[loc]
        return None

    def __delitem__(self, key:int) -> None:
        loc = self.__get_key_loc__(key)
        if (loc < len(self.k) and self.k[loc] == key):
            self.k.pop(loc)
            self.v.pop(loc)

    def __contains__(self, key:int) -> bool:
        loc = self.__get_key_loc__(key)
        return loc < len(self.k) and self.k[loc] == key


    def keys(self) -> list:
        return self.k

    def __repr__(self):
        return str(list(zip(self.k, self.v)))

    def get(self, key:int, default=None):
        loc = self.__get_key_loc__(key)
        if (loc < len(self.k) and self.k[loc] == key):
            return self.v[loc]
        return default

