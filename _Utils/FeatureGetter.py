import numpy as np
import pandas as pd

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

def init(CTX):
    global __FEATURE_MAP__, __LAT__, __LON__, __TRACK__, __TIMESTAMP__, __ICAO__, __CALLSIGN__, __BARO_ALT__, __GEO_ALT__, __VELOCITY__
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
    
def __axis_i__(arr, ind, axis):
    # return (:, :, ..., axis, :, :,)
    if (axis == -1): return tuple([slice(None) for _ in range(arr.ndim-1)] + [ind])
    if (axis < 0): axis = arr.ndim + axis
    return tuple([slice(None) if i != axis else ind for i in range(arr.ndim)])

def lat(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __LAT__
    return slf.take(__LAT__, axis)
def lon(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __LON__
    return slf.take(__LON__, axis)
def track(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __TRACK__
    return slf.take(__TRACK__, axis)
def timestamp(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __TIMESTAMP__
    return slf.take(__TIMESTAMP__, axis)
def icao(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __ICAO__
    return slf.take(__ICAO__, axis)
def callsign(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __CALLSIGN__
    return slf.take(__CALLSIGN__, axis)
def baroAlt(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __BARO_ALT__
    return slf.take(__BARO_ALT__, axis)
def geoAlt(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __GEO_ALT__
    return slf.take(__GEO_ALT__, axis)
def velocity(slf:np.ndarray = None, axis=-1):
    if not(isinstance(slf, np.ndarray)): return __VELOCITY__
    return slf.take(__VELOCITY__, axis)

def df_icao(flight:pd.DataFrame):
    return flight["icao24"].iloc[0]
def df_callsign(flight:pd.DataFrame):
    return flight["callsign"].iloc[0]


def __get__(slf:np.ndarray, feature, axis=-1): return slf.take(__FEATURE_MAP__[feature], axis)
def get(args=None):
    if (isinstance(args, str)): return __FEATURE_MAP__[args]
    return __get__(*args)
def set(slf, feature, value, axis=-1):
    slf[__axis_i__(slf, __FEATURE_MAP__[feature], axis)] = value

def has(feature):
    return feature in __FEATURE_MAP__


