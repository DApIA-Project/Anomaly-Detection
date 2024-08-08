
from typing import Generic, TypeVar, overload

V = TypeVar("V")


class OrderedDictInt(Generic[V]):
    MAX_BUCKET_LENGTH = 32

    def __init__(self) -> None:
        self.buckets = {}
        self.__keys__ = []
        self.__keys_dict__ = {}
        self.__iter_index__ = 0

    def __get_bucket__(self, key:int) -> "list":
        return self.buckets.get(key//self.MAX_BUCKET_LENGTH, None)

    def __setitem__(self, key:int, value:V):
        bucket = self.__get_bucket__(key)
        if (bucket is None):
            bucket = [None] * self.MAX_BUCKET_LENGTH
            self.buckets[key//self.MAX_BUCKET_LENGTH] = bucket

        i = key%self.MAX_BUCKET_LENGTH
        if (bucket[i] is None):
            bucket[i] = value
            self.__keys__.append(key)

    def __getitem__(self, key:int) -> V:
        bucket = self.__get_bucket__(key)
        if (bucket is None):
            return None
        return bucket[key%self.MAX_BUCKET_LENGTH]

    def __delitem__(self, key:int) -> None:
        bucket = self.__get_bucket__(key)
        if (bucket is None):
            return
        bucket[key%self.MAX_BUCKET_LENGTH] = None
        self.__keys__.remove(key)

    def __contains__(self, key:int) -> bool:
        bucket = self.__get_bucket__(key)
        if (bucket is None):
            return False
        return bucket[key%self.MAX_BUCKET_LENGTH] is not None

    def keys(self) -> "set[int]":
        return self.__keys__

    def __repr__(self) -> str:
        return str(self.buckets)

