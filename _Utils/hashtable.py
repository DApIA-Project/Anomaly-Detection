import os
import random as r

class str_(str):

    def __init__(self, s):
        self = s

    def to_bytes(self, size, *args, **kwargs):
        return super().encode()[0:size]

    def from_bytes(b, *args, **kwargs):
        return str_(b.decode()).rstrip("\x00")


class FileChainedLists:

    KEY_SIZE = 4
    CONTENT_SIZE = 4
    PATH = "test"
    space = 0
    dtype=int

    def __init__(self, path, dtype=int, content_size=4) -> None:
        self.PATH = path

        if not os.path.exists(f"{self.PATH}"):
            self.file = open(f"{self.PATH}", "wb+")
            self.file.write((0).to_bytes(self.KEY_SIZE, "big"))
        else:
            self.file = open(f"{self.PATH}", "rb+")
            self.file.seek(0)
            self.space = int.from_bytes(self.file.read(self.KEY_SIZE), "big")

        self.dtype = dtype
        if (self.dtype == str):
            self.dtype = str_
        self.CONTENT_SIZE = content_size

    def __addr__(self, it):
        return it * (self.CONTENT_SIZE + self.KEY_SIZE)

    def __get_it__(self, it, n=None):
        self.file.seek(self.__addr__(it) + self.CONTENT_SIZE)
        next = int.from_bytes(self.file.read(self.KEY_SIZE), "big")
        if (n == 0):
            return next

        if (next == 0):
            if (n == None):
                return it
            return None

        if (n == None):
            return self.__get_it__(next, n)
        return self.__get_it__(next, n - 1)


    def get_used_space(self):
        return self.space

    def close(self):
        self.file.close()

# |====================================================================================================================
# | append
# |====================================================================================================================

    def append(self, it, value):

        it = self.__get_it__(it)

        self.space = max(it, self.space) + 1

        self.file.seek(self.__addr__(it) + self.CONTENT_SIZE)
        self.file.write((self.space).to_bytes(self.KEY_SIZE, "big"))

        self.file.seek(self.__addr__(self.space))
        self.file.write(self.dtype(value).to_bytes(self.CONTENT_SIZE, "big"))

        self.file.seek(0)
        self.file.write((self.space).to_bytes(self.KEY_SIZE, "big"))

# |====================================================================================================================
# | getitem
# |====================================================================================================================

    # getitem
    def __getitem__(self, itn):
        it, n = itn
        it = self.__get_it__(it, n)
        if (it == 0):
            return None
        self.file.seek(self.__addr__(it))
        return self.dtype.from_bytes(self.file.read(self.CONTENT_SIZE), "big")


# |====================================================================================================================
# | To list
# |====================================================================================================================

    def __to_list__(self, it):
        if (it == 0):
            return []
        else:
            self.file.seek(self.__addr__(it))
            v = self.dtype.from_bytes(self.file.read(self.CONTENT_SIZE), "big")
            n = int.from_bytes(self.file.read(self.KEY_SIZE), "big")
            return [v] + self.__to_list__(n)

    def to_list(self, it):
        self.file.seek(self.__addr__(it) + self.CONTENT_SIZE)
        n = int.from_bytes(self.file.read(self.KEY_SIZE), "big")
        lst = self.__to_list__(n)
        return lst

    def clear(self):
        self.space = 0
        self.file.close()
        self.file = open(f"{self.PATH}", "wb+")
        self.file.write((0).to_bytes(self.KEY_SIZE, "big"))

    def close(self):
        self.file.close()




class FileMultiHashTable:

    KEY_SIZE = 4
    VALUE_SIZE = 4
    MAX_KEY = 2**(KEY_SIZE*8)
    MID_KEY = 2**(KEY_SIZE//2*8)
    HEADER = 4
    FILE_LENGTH = 2**12
    FILE_COUNT = 2**20
    content = None
    folder = None

    def __init__(self, folder, dtype=int, content_size=4) -> None:
        self.folder = folder
        # check if folder exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.content = FileChainedLists(os.path.join(self.folder, "values"), dtype, content_size)
        self.keys = FileChainedLists(os.path.join(self.folder, "values"), dtype, content_size)



    def __setitem__(self, key, value):
        key = int(key)
        key %= self.MAX_KEY

        filename = key // self.MID_KEY
        indice = key % self.MID_KEY
        byte = indice * self.VALUE_SIZE

        if not os.path.exists(os.path.join(self.folder, f"{filename}")):
            file = open(os.path.join(self.folder, f"{filename}"), "wb+")
            shift = byte
            file.write((shift).to_bytes(self.KEY_SIZE, "big"))
        else:
            file = open(os.path.join(self.folder, f"{filename}"), "rb+")
            shift = int.from_bytes(file.read(self.KEY_SIZE), "big")


        if (shift > byte):
            delta = shift - byte
            bucket = self.content.get_used_space() + 1

            # add header :
            content = byte.to_bytes(self.HEADER, "big")
            # add bucket
            content += bucket.to_bytes(self.KEY_SIZE, "big")
            # add space
            content += b"\x00" * (delta-self.KEY_SIZE)
            # add content from last file
            content += file.read()
            # overwrite file
            file.seek(0)
            file.write(content)

        else:
            file.seek(byte-shift + self.HEADER)
            bucket = int.from_bytes(file.read(self.KEY_SIZE), "big")

            if (bucket == 0):

                bucket = self.content.get_used_space() + 1

                file.seek(byte-shift + self.HEADER)
                file.write(bucket.to_bytes(self.KEY_SIZE, "big"))

        self.content.append(bucket, value)

        file.close()

    def __getitem__(self, key):
        key = int(key)
        key %= self.MAX_KEY

        filename = key // self.MID_KEY
        indice = key % self.MID_KEY
        byte = indice * self.VALUE_SIZE

        if not os.path.exists(os.path.join(self.folder, f"{filename}")):
            return []

        file = open(os.path.join(self.folder, f"{filename}"), "rb")
        shift = int.from_bytes(file.read(self.KEY_SIZE), "big")
        if (byte-shift < 0):
            return []
        file.seek(byte-shift + self.HEADER)

        bucket = int.from_bytes(file.read(self.KEY_SIZE), "big")
        if (bucket == 0):
            return []

        return self.content.to_list(bucket)

    # in operator
    def __contains__(self, key):
        return self.get(key) != None


    def get(self, key, default=None):
        res = self.__getitem__(key)
        if (len(res) == 0):
            return default
        return res

    def clear(self):
        self.content.close()
        os.system(f"rm -r {self.folder}")
        os.makedirs(self.folder)
        self.content = FileChainedLists(os.path.join(self.folder, "values"), self.content.dtype, self.content.CONTENT_SIZE)





