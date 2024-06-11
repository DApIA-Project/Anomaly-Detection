
from _Utils.numpy import np, ax

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


def cast_msg(col:str, msg:object) -> float:

    if (msg is np.nan or msg == None or msg == ""):
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
class Streamer:
    def __init__(self) -> None:
        self.trajectories:dict[str, DataFrame] = {}
        self.__cache__:dict[str, dict[str, object]] = {}
        self.__icao_to_tag__:dict[str, set] = {}
        self.__tag_to_icao__:dict[str, str] = {}

    def clear(self) -> None:
        self.trajectories.clear()
        self.__cache__.clear()
        self.__icao_to_tag__.clear()
        self.__tag_to_icao__.clear()

    def add(self, x:"dict[str, object]", tag:str=None) -> DataFrame:
        if (tag == None):
            tag = x['icao24']

        if (x['icao24'] not in self.__icao_to_tag__):
            self.__icao_to_tag__[x['icao24']] = set()
        self.__icao_to_tag__[x['icao24']].add(tag)
        self.__tag_to_icao__[tag] = x["icao24"]

        if tag not in self.trajectories:
            self.trajectories[tag] = DataFrame(len(__FEATURES__))
            self.trajectories[tag].setColums(__FEATURES__)

        x = [cast_msg(col, x.get(col, np.nan)) for col in __FEATURES__]

        last_timestamp = self.trajectories[tag].array[-1][__FEATURE_MAP__['timestamp']]
        timestamp = x[__FEATURE_MAP__['timestamp']]
        MAX_GAP = 30 * 60
        if (last_timestamp > 0 and timestamp - last_timestamp > MAX_GAP):
            prntC(C.WARNING, f"Gap of {timestamp - last_timestamp} seconds for {tag} at timestamp {x[__FEATURE_MAP__['timestamp']]}.")
            self.trajectories[tag].clear()
            self.__cache__[tag] = {}

        if(not(self.trajectories[tag].set(x))):
            prntC(C.WARNING, f"Duplicate message for {tag} at timestamp {x[__FEATURE_MAP__['timestamp']]}")

        return self.trajectories[tag]

    def set(self, x:"dict[str, object]", tag:str) -> DataFrame:
        if (tag not in self.trajectories):
            self.add(x, tag)

        x = [cast_msg(col, x.get(col, np.nan)) for col in __FEATURES__]


    # def remove(self, tag:str) -> DataFrame:
    #     if (tag in self.trajectories):
    #         self.trajectories.pop(tag)
    #         self.__icao_to_tag__[self.__tag_to_icao__[tag]].remove(tag)
            # self.__tag_to_icao__.pop(tag)

    def get(self, tag:str) -> DataFrame:
        return self.trajectories.get(tag, None)

    def cache(self, label:str, tag:str, data:object=None)->"object|None":
        if (data is None):
            return self.__get_cache__(tag, label)

        if (tag not in self.__cache__):
            self.__cache__[tag] = {}

        self.__cache__[tag][label] = data

    def __get_cache__(self, tag:str, label:str) -> "object|None":
        tmp = self.__cache__.get(tag, None)
        if (tmp == None):
            return None
        return tmp.get(label, None)

    def get_tags_for_icao(self, icao:str) -> "set[str]":
        return self.__icao_to_tag__.get(icao, set())
    def get_icao_for_tag(self, tag:str) -> str:
        return self.__tag_to_icao__.get(tag, None)
