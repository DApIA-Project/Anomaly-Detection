
import numpy as np

from _Utils.DataFrame import DataFrame
from _Utils.Color import prntC
import _Utils.Color as C


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


def cast_msg(col, msg):
    if (msg == np.nan or msg == None or msg == ""):
        return np.nan
    elif (col == "icao24" or col == "callsign"):
        return msg
    elif (col == "onground" or col == "alert" or col == "spi"):
        return float(msg == "True")
    elif (col == "timestamp"):
        return int(msg)
    else:
        return float(msg)

# |====================================================================================================================
# | ADSBStreamer : Replay ADS-B messages and store trajectories
# |====================================================================================================================
class _ADSBStreamer:
    def __init__(self):
        self.trajectories:dict[str, DataFrame] = {}
        self.__cache__:dict[str, dict[str, np.ndarray]] = {}

    def add(self, x:dict) -> DataFrame:
        icao24 = x['icao24']
        if icao24 not in self.trajectories:
            self.trajectories[icao24] = DataFrame(len(__FEATURES__))
            self.trajectories[icao24].setColums(__FEATURES__)

        x = [cast_msg(col, x.get(col, np.nan)) for col in __FEATURES__]

        if(not(self.trajectories[icao24].set(x))):
            prntC(C.WARNING, f"Duplicate message for {icao24} at timestamp {x[__FEATURE_MAP__['timestamp']]}")

        return self.trajectories[icao24]

    def get(self, icao24:str) -> DataFrame:
        return self.trajectories.get(icao24, None)

    def cache(self, tag:str, icao24:str, data:np.ndarray=None)->"np.ndarray|None":
        if (data is None):
            return self.__get_cache__(tag, icao24)

        if (tag not in self.__cache__):
            self.__cache__[tag] = {}

        self.__cache__[tag][icao24] = data

    def __get_cache__(self, tag, icao24):
        emp = self.__cache__.get(tag, None)
        if (emp == None):
            return None
        return emp.get(icao24, None)


Streamer = _ADSBStreamer()