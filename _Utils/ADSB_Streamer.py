
from numpy_typing import np, ax

from   _Utils.DataFrame import DataFrame
from   _Utils.Color import prntC
import _Utils.Color as C
from   _Utils.utils import *
import _Utils.geographic_maths as GEO


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
    def get_feature_size(self) -> int:
        return self.feature_size


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
        if index is not None:
            index = [index]
        return self.extend(icao, tag, [data], index)


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

class FloodingData:
    timestamp:int
    index:int

    def __init__(self, timestamp:int, index:int) -> None:
        self.timestamp = timestamp
        self.index = index

class Trajectory:
    icao:str
    tag:str
    data:DataFrame
    childs:"set[str]"
    parent:str
    __cache_items__:"list[CacheElement]"
    flooding:"list[FloodingData]"
    abnormal:int

    def __init__(self, icao:str, tag:str) -> None:
        self.icao = icao
        self.tag = tag
        self.data = DataFrame(len(__FEATURES__))
        self.data.rename_columns(__FEATURES__)
        self.childs = set()
        self.parent = None
        self.__cache_items__ = []
        self.flooding=[]
        self.abnormal = -1

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
        split = False

        for cache in self.__cache__:
            cache_el = cache.open_cache_for(icao, tag)
            traj.__cache_items__.append(cache_el)

        if (tag == "0"):
            traj.parent = None
            prntC(C.INFO, f"New trajectory {i_tag} created")
            return traj, split

        parent_trajectory:Trajectory = self.trajectories.get(icao+"_0", None)
        if (parent_trajectory == None):
            prntC(C.WARNING, f"New trajectory {i_tag} created but parent trajectory {icao}_0 does not exist")
        else:
            parent_trajectory.childs.add(traj.i_tag())
            traj.parent = parent_trajectory.i_tag()

            # transfer the history of the parent trajectory to the child
            i = parent_trajectory.data.get_relative_loc(start_time)
            traj.data = parent_trajectory.data[:i]



            parent_trajectory.flooding.append(FloodingData(start_time, i))
            traj.flooding.append(FloodingData(start_time, i))

            split = True


        return traj, split

    def split_cache(self, message:"dict[str, object]") -> None:
        icao = message["icao24"]
        tag = message.get("tag", "0")

        traj = self.get(icao, tag)
        i = traj.flooding[-1].index

        for cache in self.__cache__:
            cache.set(icao, tag, cache.subset(icao, "0", 0, i))
            subset = cache.subset(icao, "0", 0, i)


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

        for child in list(childs):
            self.__remove_trajectory__(self.trajectories[child])

        del self.trajectories[trajectory.icao+"_"+trajectory.tag]


    def add(self, x:"dict[str, object]") -> DataFrame:
        tag = x.get("tag", "0")
        icao = x["icao24"]
        i_tag = icao+"_"+tag
        actual_time = x["timestamp"]
        split = False

        # create a new trajectory if it doesn't exist
        trajectory = self.trajectories.get(i_tag, None)
        if trajectory is None:
            trajectory, split = self.__create_trajectory__(icao, tag, start_time=actual_time)

        elif self.__is_a_new_trajectory__(trajectory, x):
            self.__remove_trajectory__(trajectory)
            trajectory, split = self.__create_trajectory__(icao, tag, start_time=actual_time)

        # check if the message doesn't go back in time
        i = trajectory.data.get_relative_loc(actual_time)
        if (i < len(trajectory.data)):
            prntC(C.WARNING, f"Duplicate message for {i_tag} at timestamp {actual_time}")

            self.__remove_trajectory__(trajectory)
            trajectory, split = self.__create_trajectory__(icao, tag, start_time=actual_time)

        # check if the message dosen't teleport
        lats, lons, ts = trajectory.data["latitude"], trajectory.data["longitude"], trajectory.data["timestamp"]
        if (len(ts) >= 2):
            t = len(lats) - 1
            while (t >= 1 and ((lats[t] == 0 and lons[t] == 0) or (lats[t-1] == lats[t] and lons[t-1] == lons[t]))):
                t -= 1
            llat, llon, lt = lats[t], lons[t], ts[t]
            nlat, nlon, nt = x.get("latitude", np.nan), x.get("longitude", np.nan), x.get("timestamp", np.nan)
            if (GEO.distance(llat, llon, nlat, nlon) / (nt - lt) > 686): # mach 2
                prntC(C.WARNING, f"Teleportation detected for {i_tag} at timestamp {actual_time}")
                self.__remove_trajectory__(trajectory)
                trajectory, split = self.__create_trajectory__(icao, tag, start_time=actual_time)

        # append the message to the trajectory
        x = [x.get(col, np.nan) for col in __FEATURES__]
        trajectory.data.__append__(x)

        return split


    def get(self, icao:str, tag:str) -> "Trajectory|None":
        return self.trajectories.get(icao+"_"+tag, None)

    def is_flooding(self, traj:Trajectory, timestamp:int, delay:int) -> bool:
        for flood in traj.flooding:
            if (flood.timestamp <= timestamp and timestamp < flood.timestamp + delay):
                return True
        return False

    def ended_flooding(self, traj:Trajectory, timestamp:int, delay:int) -> bool:
        t = traj.data.get_relative_loc(timestamp)
        last_timestamp = traj.data["timestamp", t-1]
        for flood in traj.flooding:
            if (last_timestamp <=flood.timestamp + delay and timestamp >= flood.timestamp + delay):
                return True
        return False

    def setAbnormal(self, icao:str, tag:str, timestamp:int) -> None:
        traj = self.get(icao, tag)
        if (traj is None): return
        traj.abnormal = timestamp

    def isAbnormal(self, icao:str, tag:str, timestamp:int) -> bool:
        traj = self.get(icao, tag)
        if (traj is None): return False
        if (traj.abnormal == -1): return False
        return traj.abnormal < timestamp





streamer = Streamer()
