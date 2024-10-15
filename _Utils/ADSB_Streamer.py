
from numpy_typing import np, ax

from   _Utils.DataFrame import DataFrame
from   _Utils.Color import prntC
import _Utils.Color as C
from   _Utils.utils import *


# |====================================================================================================================
# | CACHE system
# |====================================================================================================================

# |--------------------------------------------------------------------------------------------------------------------
# | Generic Cache system
# |--------------------------------------------------------------------------------------------------------------------


class CacheElement:
    icao:str
    tag:str
    data:object

    def __init__(self, icao:str, tag:str, data:object) -> None:
        self.icao = icao
        self.tag = tag
        self.data = data

    def subset(self, start:int, end:int) -> object:
        raise NotImplementedError

class Cache:
    data:"dict[str, CacheElement]"

    def __init__(self) -> None:
        streamer.add_cache(self)
        self.data:dict[str, CacheElement] = {}

    def get(self, icao:str, tag:str) -> object:
        return self.data.get(icao+"_"+tag, None).data

    def set(self, icao:str, tag:str, data:object) -> None:
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            self.data[i_tag] = CacheElement(icao, tag, data)
        else:
            el.data = data

    def subset(self, icao:str, tag:str, start:int, end:int) -> object:
        raise NotImplementedError("Cache.subset() is not implemented for abstract class Cache")

    def open_cache_for(self, icao:str, tag:str) -> CacheElement:
        raise NotImplementedError("Cache.open_cache_for() is not implemented for abstract class Cache")
    def close_cache_for(self, icao:str, tag:str) -> None:
        raise NotImplementedError("Cache.close_cache_for() is not implemented for abstract class Cache")


# |--------------------------------------------------------------------------------------------------------------------
# | Cache for 2D arrays [time, feature]
# |--------------------------------------------------------------------------------------------------------------------

class CacheArray2DElement(CacheElement):
    data:np.float64_2d[ax.time, ax.feature]
    indexing:"list[int]"

    def __init__(self, icao:str, tag:str, feature_size:int, dtype=np.float64) -> None:
        super().__init__(icao, tag, np.empty((0, feature_size), dtype=dtype))
        self.indexing = []


class CacheArray2D(Cache):
    data:"dict[str, CacheArray2DElement]"
    feature_size:int
    dtype:np.dtype

    def __init__(self, dtype:np.dtype=np.float64) -> None:
        """
        if fiexed size, array length is physically always the same (size may vary)
        """
        super().__init__()
        self.data:dict[str, CacheArray2DElement] = {}
        self.dtype = dtype



    def set_feature_size(self, feature_size:int) -> None:
        self.feature_size = feature_size


    def open_cache_for(self, icao:str, tag:str) -> CacheArray2DElement:
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            el = CacheArray2DElement(icao, tag, self.feature_size, self.dtype)
            self.data[i_tag] = el
        else:
            prntC(C.ERROR, f"Cache for {i_tag} already exists")
        return el


    def close_cache_for(self, icao:str, tag:str) -> None:
        i_tag = icao+"_"+tag
        if (i_tag in self.data):
            del self.data[i_tag]


    def get(self, icao:str, tag:str) -> np.float64_2d[ax.time, ax.feature]:
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            return None
        return el.data


    def set(self, icao:str, tag:str, data:np.float64_2d[ax.time, ax.feature], indexing:"list[int]"=None) \
            -> "np.float64_2d[ax.time, ax.feature]|None":
        i_tag = icao+"_"+tag
        if (len(data) == 0):
            if (i_tag in self.data):
                del self.data[i_tag]
            return None

        if (indexing is None):
            indexing = list(range(len(data)))

        el = self.data.get(i_tag, None)
        if (el is None):
            el = CacheArray2DElement(icao, tag, self.feature_size, self.dtype)
            el.data = data
            el.indexing = indexing
            self.data[i_tag] = el
        else:
            el.data = data
            el.indexing = indexing

        return el.data



    def append(self, icao:str, tag:str, data:np.float64_1d[ax.feature], index:int=None) \
            -> np.float64_2d[ax.time, ax.feature]:

        # convert to np array data
        data = np.array(data, dtype=self.dtype)
        return self.extend(icao, tag, data.reshape(1, self.feature_size), index)


    def extend(self, icao:str, tag:str, data:"np.float64_2d[ax.sample, ax.feature]",
               indexs:"list[int]"=None) -> np.float64_2d[ax.time, ax.feature]:

        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            return self.set(icao, tag, data, indexs)
        else:
            if (indexs is None):
                indexs = list(range(len(el.data), len(el.data)+len(data)))
            el.data = np.vstack([el.data, data])
            el.indexing.extend(indexs)
        return el.data

    def subset(self, icao:str, tag:str, start:int, end:int) -> np.float64_2d[ax.time, ax.feature]:
        el = self.data.get(icao+"_"+tag, None)
        if (el is None):
            return np.empty((0, self.feature_size), dtype=self.dtype)
        a = 0
        while (a < len(el.indexing) and el.indexing[a] < start):
            a += 1
        b = a
        while (b < len(el.indexing) and el.indexing[b] < end):
            b += 1
        return el.data[a:b]



# |--------------------------------------------------------------------------------------------------------------------
# | Cache of list of Any
# |--------------------------------------------------------------------------------------------------------------------

class CacheListElement(CacheElement):
    data:"list[object]"

    def __init__(self, icao:str, tag:str) -> None:
        super().__init__(icao, tag, [])
        self.indexing = []

class CacheList(Cache):
    data:"dict[str, CacheListElement]"

    def __init__(self) -> None:
        super().__init__()
        self.data:dict[str, CacheListElement] = {}


    def open_cache_for(self, icao:str, tag:str) -> CacheListElement:
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            el = CacheListElement(icao, tag)
            self.data[i_tag] = el
        else:
            prntC(C.ERROR, f"Cache for {i_tag} already exists")
        return el


    def close_cache_for(self, icao:str, tag:str) -> None:
        i_tag = icao+"_"+tag
        if (i_tag in self.data):
            del self.data[i_tag]


    def get(self, icao:str, tag:str) -> "list[object]":
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            return None
        return el.data


    def set(self, icao:str, tag:str, data:"list[object]", indexing:"list[int]"=None) -> "list[object]|None":
        i_tag = icao+"_"+tag
        if (len(data) == 0):
            if (i_tag in self.data):
                del self.data[i_tag]
            return None

        if (indexing is None):
            indexing = list(range(len(data)))

        el = self.data.get(i_tag, None)
        if (el is None):
            el = CacheListElement(icao, tag)
            el.data = data
            el.indexing = indexing
            self.data[i_tag] = el
        else:
            el.data = data
            el.indexing = indexing

        return el.data


    def append(self, icao:str, tag:str, data:object, index:int=None) -> "list[object]":
        return self.extend(icao, tag, [data], [index])


    def extend(self, icao:str, tag:str, data:"list[object]", indexs:"list[int]"=None) -> "list[object]":
        i_tag = icao+"_"+tag
        el = self.data.get(i_tag, None)
        if (el is None):
            return self.set(icao, tag, data, indexs)
        else:
            if (indexs is None):
                indexs = list(range(len(el.data), len(el.data)+len(data)))
            el.data.extend(data)
            el.indexing.extend(indexs)
        return el.data

    def subset(self, icao:str, tag:str, start:int, end:int) -> "list[object]":
        el = self.data.get(icao+"_"+tag, None)
        if (el is None):
            return []
        a = 0
        while (a < len(el.indexing) and el.indexing[a] < start):
            a += 1
        b = a
        while (b < len(el.indexing) and el.indexing[b] < end):
            b += 1
        return el.data[a:b]



# |====================================================================================================================
# | STREAMER
# |====================================================================================================================

# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

__FEATURES__ = [
    "timestamp",
    "latitude", "longitude",
    "groundspeed", "track",
    "vertical_rate", "onground",
    "alert", "spi", "squawk",
    "altitude", "geoaltitude"
]

__FEATURE_MAP__ = dict([[__FEATURES__[i], i] for i in range(len(__FEATURES__))])

# |====================================================================================================================
# | UTILS
# |====================================================================================================================


def cast_feature(col:str, msg:object) -> float:
    if (msg is np.nan or msg == None or msg == ""):
        return np.nan

    elif (col == "icao24" or col == "callsign" or col == "tag"):
        return msg
    elif (col == "onground" or col == "alert" or col == "spi"):
        return float(msg == "True")
    elif (col == "timestamp"):
        return int(msg)
    elif (col == "latitude" or col == "longitude" or col == "altitude" or col == "geoaltitude" \
            or col == "groundspeed" or col == "vertical_rate" or col == "track" or col == "squawk"):
        return float(msg)
    else:
        return msg



def cast_msg(msg:"dict[str, object]") -> "dict[str, float]":
    return {col:cast_feature(col, value) for col, value in msg.items()}


# |====================================================================================================================
# | TRAJECTORY
# |====================================================================================================================

class Trajectory:
    def __init__(self, icao:str, tag:str) -> None:
        self.icao = icao
        self.tag = tag
        self.data:DataFrame = DataFrame(len(__FEATURES__))
        self.data.rename_columns(__FEATURES__)
        self.childs:set[str] = set()
        self.parent:str = None
        self.__cache_items__:list[CacheElement] = []

    def i_tag(self) -> str:
        return self.icao+"_"+self.tag

# |====================================================================================================================
# | STREAMING ADSB DATA
# |====================================================================================================================


class Streamer:
    def __init__(self) -> None:
        self.trajectories:dict[str, Trajectory] = {}
        self.__cache__:list[Cache] = []

    def add_cache(self, cache:Cache) -> None:
        self.__cache__.append(cache)


    def __create_trajectory__(self, icao:str, tag:str, start_time:int) -> Trajectory:
        """
        Create a new trajectory and add it to the list of trajectories
        Return a reference to the new trajectory
        """
        i_tag = icao+"_"+tag
        traj = Trajectory(icao, tag)
        self.trajectories[i_tag] = traj

        for cache in self.__cache__:
            cache_el = cache.open_cache_for(icao, tag)
            traj.__cache_items__.append(cache_el)

        if (tag == "0"):
            traj.parent = None
            prntC(C.INFO, f"New trajectory {i_tag} created")

        else:
            parent_trajectory:Trajectory = self.trajectories.get(icao+"_0", None)
            if (parent_trajectory == None):
                prntC(C.WARNING, f"New trajectory {i_tag} created but parent trajectory {icao}_0 does not exist")
            else:
                parent_trajectory.childs.add(traj.i_tag())
                traj.parent = parent_trajectory.i_tag()

                # transfer the history of the parent trajectory to the child
                i = parent_trajectory.data.get_relative_loc(start_time)
                parent_trajectory.data = parent_trajectory.data[:i]
                for cache in self.__cache__:
                    cache.set(icao, tag, cache.subset(icao, "0", 0, i))

                prntC(C.INFO, f"New trajectory {i_tag} created as child of {icao}_0")

        return traj

    def __is_a_new_trajectory__(self, trajectory:Trajectory, next_message:"dict[str, object]") -> bool:
        MAX_GAP = 30 * 60 # 30 min gap
        actual_time = next_message['timestamp']
        if (len(trajectory.data) == 0): return False
        last_update = trajectory.data["timestamp", -1]
        return (actual_time - last_update > MAX_GAP)

    def __remove_trajectory__(self, trajectory:Trajectory) -> None:
        childs = trajectory.childs
        parent = trajectory.parent

        prntC(C.INFO, f"Removing trajectory {trajectory.icao}_{trajectory.tag}")

        for cache in self.__cache__:
            cache.close_cache_for(trajectory.icao, trajectory.tag)

        if (parent is not None):
            self.trajectories[parent].childs.remove(trajectory.i_tag())

        for child in childs:
            self.__remove_trajectory__(self.trajectories[child])

        del self.trajectories[trajectory.icao+"_"+trajectory.tag]


    def add(self, x:"dict[str, object]") -> DataFrame:
        tag = x.get("tag", "0")
        icao = x["icao24"]
        i_tag = icao+"_"+tag
        actual_time = x["timestamp"]

        # create a new trajectory if it doesn't exist
        trajectory = self.trajectories.get(i_tag, None)
        if trajectory is None:
            trajectory = self.__create_trajectory__(icao, tag, start_time=actual_time)

        if self.__is_a_new_trajectory__(trajectory, x):
            self.__remove_trajectory__(trajectory)
            trajectory = self.__create_trajectory__(icao, tag, start_time=actual_time)


        # check if the message doesn't go back in time
        i = trajectory.data.get_relative_loc(actual_time)
        if (i < len(trajectory.data)):
            prntC(C.WARNING, f"Duplicate message for {i_tag} at timestamp {actual_time}")

            self.__remove_trajectory__(trajectory)
            trajectory = self.__create_trajectory__(icao, tag, start_time=actual_time)

        # append the message to the trajectory
        x = [x.get(col, np.nan) for col in __FEATURES__]
        trajectory.data.__append__(x)

        return trajectory.data


    def get(self, icao:str, tag:str) -> "DataFrame|None":
        trajectory = self.trajectories.get(icao+"_"+tag, None)
        if (trajectory is None):
            return None
        return trajectory.data


streamer = Streamer()

