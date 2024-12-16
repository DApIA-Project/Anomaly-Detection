from numpy_typing import np, ax
import pandas as pd
from typing import overload, TypeVar

class FeatureGetter:

    FEATURE_MAP = None
    LAT = None
    LON = None
    TRACK = None
    TIMESTAMP = None
    ICAO = None
    CALLSIGN = None
    BARO_ALT = None
    GEO_ALT = None
    VELOCITY = None

    def init(self, CTX:dict)->None:
        """Init the feature getter with the context"""
        self.FEATURE_MAP = CTX["FEATURE_MAP"]
        self.LAT       = self.FEATURE_MAP.get("latitude", None)
        self.LON       = self.FEATURE_MAP.get("longitude", None)
        self.TRACK     = self.FEATURE_MAP.get("track", None)
        self.TIMESTAMP = self.FEATURE_MAP.get("timestamp", None)
        self.ICAO      = self.FEATURE_MAP.get("icao24", None)
        self.CALLSIGN  = self.FEATURE_MAP.get("callsign", None)
        self.BARO_ALT  = self.FEATURE_MAP.get("altitude", None)
        self.GEO_ALT   = self.FEATURE_MAP.get("geoaltitude", None)
        self.VELOCITY  = self.FEATURE_MAP.get("groundspeed", None)
        CTX["FG"] = self

    T  = TypeVar('T')
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    T3 = TypeVar('T3')
    T4 = TypeVar('T4')

    # |====================================================================================================================
    # | LAT GETTER
    # |====================================================================================================================
    @overload
    def lat(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def lat(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def lat(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def lat(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def lat(self) -> int: ...

    def lat(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.LAT
        return slf.take(self.LAT, axis)


    # |====================================================================================================================
    # | LON GETTER
    # |====================================================================================================================
    @overload
    def lon(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def lon(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def lon(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def lon(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def lon(self) -> int: ...


    def lon(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.LON
        return slf.take(self.LON, axis)


    # |====================================================================================================================
    # | LAT_LON GETTER
    # |====================================================================================================================
    @overload
    def lat_lon(self, slf:T, axis:int=-1) -> T: ...
    @overload
    def lat_lon(self) -> "tuple[int, int]": ...


    def lat_lon(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return (self.LAT, self.LON)
        return slf.take([self.LAT, self.LON], axis)


    # |====================================================================================================================
    # | TRACK GETTER
    # |====================================================================================================================
    @overload
    def track(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def track(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def track(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def track(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def track(self) -> int: ...


    def track(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.TRACK
        if self.TRACK is None: return None
        return slf.take(self.TRACK, axis)

    # |====================================================================================================================
    # | TIMESTAMP GETTER
    # |====================================================================================================================
    @overload
    def timestamp(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def timestamp(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def timestamp(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def timestamp(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def timestamp(self) -> int: ...

    def timestamp(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.TIMESTAMP
        return slf.take(self.TIMESTAMP, axis)

    # |====================================================================================================================
    # | BARMOETRIC ALTITUDE GETTER
    # |====================================================================================================================
    @overload
    def baroAlt(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def baroAlt(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def baroAlt(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def baroAlt(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def baroAlt(self) -> int: ...


    def baroAlt(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.BARO_ALT
        return slf.take(self.BARO_ALT, axis)

    # |====================================================================================================================
    # | GPS ALTITUDE GETTER
    # |====================================================================================================================
    @overload
    def geoAlt(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def geoAlt(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def geoAlt(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def geoAlt(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def geoAlt(self) -> int: ...


    def geoAlt(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.GEO_ALT
        return slf.take(self.GEO_ALT, axis)

    # |====================================================================================================================
    # | VELOCITY GETTER
    # |====================================================================================================================
    @overload
    def velocity(self, slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
    @overload
    def velocity(self, slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def velocity(self, slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def velocity(self, slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def velocity(self) -> int: ...

    def velocity(self, slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
        if not(isinstance(slf, np.ndarray)): return self.VELOCITY
        return slf.take(self.VELOCITY, axis)

    # |====================================================================================================================
    # | GENERIC GETTER
    # |====================================================================================================================

    @overload
    def get(self, slf:np.array_1d[T, T1], feature:str, axis:int=-1) -> T: ...
    @overload
    def get(self, slf:np.array_2d[T, T1, T2], feature:str, axis:int=-1) -> np.array_1d[T, T1]: ...
    @overload
    def get(self, slf:np.array_3d[T, T1, T2, T3], feature:str, axis:int=-1) -> np.array_2d[T, T1, T2]: ...
    @overload
    def get(self, slf:np.array_4d[T, T1, T2, T3, T4], feature:str, axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
    @overload
    def get(self, feature:str) -> int: ...


    def __get__(self, slf:np.ndarray, feature:str, axis:int=-1) -> np.ndarray:
        return slf.take(self.FEATURE_MAP[feature], axis)

    def get(self, *args) -> np.ndarray:
        if (isinstance(args[0], str)): return self.FEATURE_MAP[args[0]]
        return self.__get__(*args)


    def get_not(self, slf:np.ndarray, feature:str, axis:int=-1) -> np.ndarray:
        return slf.take([i for i in range(slf.shape[axis]) if i != self.FEATURE_MAP[feature]], axis)

    # |====================================================================================================================
    # | GENERIC SETTER
    # |====================================================================================================================

    def __axis_i__(self, arr:np.ndarray, ind:slice, axis:int) -> "tuple[slice]":
        # return (:, :, ..., axis, :, :,)
        if (axis == -1): return tuple([slice(None) for _ in range(arr.ndim-1)] + [ind])
        if (axis < 0): axis = arr.ndim + axis
        return tuple([slice(None) if i != axis else ind for i in range(arr.ndim)])

    def set(self, slf:np.ndarray, feature:str, value:"float|np.ndarray", axis:int=-1) -> np.ndarray:
        slf[self.__axis_i__(slf, self.FEATURE_MAP[feature], axis)] = value
        return slf

    # |====================================================================================================================
    # | check if a feature exists
    # |====================================================================================================================

    def has(self, feature:str) -> bool:
        return feature in self.FEATURE_MAP


FG_spoofing = FeatureGetter()
FG_flooding = FeatureGetter()
FG_replay = FeatureGetter()
FG_separator = FeatureGetter()
FG_interp = FeatureGetter()