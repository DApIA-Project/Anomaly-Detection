from _Utils.numpy import np, ax
import pandas as pd
from typing import overload, TypeVar

__FEATURE_MAP__ = None
__LAT__ = None
__LON__ = None
__TRACK__ = None
__TIMESTAMP__ = None
__ICAO__ = None
__CALLSIGN__ = None
__BARO_ALT__ = None
__GEO_ALT__ = None
__VELOCITY__ = None

def init(CTX:dict)->None:
    """Init the feature getter with the context"""
    global __FEATURE_MAP__,\
           __LAT__, __LON__, __TRACK__,\
           __TIMESTAMP__, __ICAO__, __CALLSIGN__,\
           __BARO_ALT__, __GEO_ALT__, __VELOCITY__
    __FEATURE_MAP__ = CTX["FEATURE_MAP"]
    __LAT__       = __FEATURE_MAP__.get("latitude", None)
    __LON__       = __FEATURE_MAP__.get("longitude", None)
    __TRACK__     = __FEATURE_MAP__.get("track", None)
    __TIMESTAMP__ = __FEATURE_MAP__.get("timestamp", None)
    __ICAO__      = __FEATURE_MAP__.get("icao24", None)
    __CALLSIGN__  = __FEATURE_MAP__.get("callsign", None)
    __BARO_ALT__  = __FEATURE_MAP__.get("altitude", None)
    __GEO_ALT__   = __FEATURE_MAP__.get("geoaltitude", None)
    __VELOCITY__  = __FEATURE_MAP__.get("groundspeed", None)

T  = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')

# |====================================================================================================================
# | LAT GETTER
# |====================================================================================================================
@overload
def lat(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def lat(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def lat(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def lat(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def lat() -> int: ...

def lat(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __LAT__
    return slf.take(__LAT__, axis)


# |====================================================================================================================
# | LON GETTER
# |====================================================================================================================
@overload
def lon(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def lon(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def lon(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def lon(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def lon() -> int: ...


def lon(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __LON__
    return slf.take(__LON__, axis)


# |====================================================================================================================
# | LAT_LON GETTER
# |====================================================================================================================
@overload
def lat_lon(slf:T, axis:int=-1) -> T: ...
@overload
def lat_lon() -> "tuple[int, int]": ...


def lat_lon(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return (__LAT__, __LON__)
    return slf.take([__LAT__, __LON__], axis)


# |====================================================================================================================
# | TRACK GETTER
# |====================================================================================================================
@overload
def track(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def track(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def track(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def track(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def track() -> int: ...


def track(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __TRACK__
    return slf.take(__TRACK__, axis)

# |====================================================================================================================
# | TIMESTAMP GETTER
# |====================================================================================================================
@overload
def timestamp(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def timestamp(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def timestamp(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def timestamp(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def timestamp() -> int: ...

def timestamp(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __TIMESTAMP__
    return slf.take(__TIMESTAMP__, axis)

# |====================================================================================================================
# | BARMOETRIC ALTITUDE GETTER
# |====================================================================================================================
@overload
def baroAlt(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def baroAlt(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def baroAlt(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def baroAlt(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def baroAlt() -> int: ...


def baroAlt(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __BARO_ALT__
    return slf.take(__BARO_ALT__, axis)

# |====================================================================================================================
# | GPS ALTITUDE GETTER
# |====================================================================================================================
@overload
def geoAlt(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def geoAlt(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def geoAlt(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def geoAlt(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def geoAlt() -> int: ...


def geoAlt(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __GEO_ALT__
    return slf.take(__GEO_ALT__, axis)

# |====================================================================================================================
# | VELOCITY GETTER
# |====================================================================================================================
@overload
def velocity(slf:np.array_1d[T, T1], axis:int=-1) -> T: ...
@overload
def velocity(slf:np.array_2d[T, T1, T2], axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def velocity(slf:np.array_3d[T, T1, T2, T3], axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def velocity(slf:np.array_4d[T, T1, T2, T3, T4], axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def velocity() -> int: ...

def velocity(slf:np.ndarray = None, axis:int=-1) -> np.ndarray:
    if not(isinstance(slf, np.ndarray)): return __VELOCITY__
    return slf.take(__VELOCITY__, axis)

# |====================================================================================================================
# | GENERIC GETTER
# |====================================================================================================================

@overload
def get(slf:np.array_1d[T, T1], feature:str, axis:int=-1) -> T: ...
@overload
def get(slf:np.array_2d[T, T1, T2], feature:str, axis:int=-1) -> np.array_1d[T, T1]: ...
@overload
def get(slf:np.array_3d[T, T1, T2, T3], feature:str, axis:int=-1) -> np.array_2d[T, T1, T2]: ...
@overload
def get(slf:np.array_4d[T, T1, T2, T3, T4], feature:str, axis:int=-1) -> np.array_3d[T, T1, T2, T3]: ...
@overload
def get(feature:str) -> int: ...


def __get__(slf:np.ndarray, feature:str, axis:int=-1) -> np.ndarray:
    return slf.take(__FEATURE_MAP__[feature], axis)

def get(args:any=None) -> np.ndarray:
    if (isinstance(args, str)): return __FEATURE_MAP__[args]
    return __get__(*args)

# |====================================================================================================================
# | GENERIC SETTER
# |====================================================================================================================

def __axis_i__(arr:np.ndarray, ind:slice, axis:int) -> "tuple[slice]":
    # return (:, :, ..., axis, :, :,)
    if (axis == -1): return tuple([slice(None) for _ in range(arr.ndim-1)] + [ind])
    if (axis < 0): axis = arr.ndim + axis
    return tuple([slice(None) if i != axis else ind for i in range(arr.ndim)])

def set(slf:np.ndarray, feature:str, value:"float|np.ndarray", axis:int=-1) -> np.ndarray:
    slf[__axis_i__(slf, __FEATURE_MAP__[feature], axis)] = value
    return slf

# |====================================================================================================================
# | check if a feature exists
# |====================================================================================================================

def has(feature:str) -> bool:
    return feature in __FEATURE_MAP__


