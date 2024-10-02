

import math
from numpy_typing import np as np_, ax

from typing import overload, TypeVar



def predict(lat:float, lon:float, bearing:float, distance:float) -> "tuple[float, float]":
    """
    Predict the next position of an aircraft
    """
    R = 6371*1000 # m
    lat = math.radians(lat)
    lon = math.radians(lon)
    bearing = math.radians(bearing)

    lat2 = math.asin(math.sin(lat)*math.cos(distance/R) + math.cos(lat)*math.sin(distance/R)*math.cos(bearing))
    lon2 = lon + math.atan2(
        math.sin(bearing)*math.sin(distance/R)*math.cos(lat),
        math.cos(distance/R)-math.sin(lat)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def bearing(lat1:float, lon1:float, lat2:float, lon2:float) -> float:
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
    return math.degrees(math.atan2(y, x))


def distance(lat1:float, lon1:float, lat2:float, lon2:float) -> float:
    """Compute the distance between two points on earth in meters"""
    # Radius of earth in KM
    R = 6378.137
    d_lat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    d_lon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(d_lat/2.0)**2 + \
        math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(d_lon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    return distance

def bearing_diff(a:float, b:float)->float:
    a = a % 360
    b = b % 360

    # compute relative angle
    diff = b - a

    if (diff > 180):
        diff -= 360
    elif (diff < -180):
        diff += 360
    return diff


T  = TypeVar('T', float, np_.ndarray)
class np:

    @overload
    @staticmethod
    def bearing(lat1:T,lon1:T,lat2:T,lon2:T) -> T: ...

    @staticmethod
    def bearing(lat1:np_.ndarray, lon1:np_.ndarray, lat2:np_.ndarray, lon2:np_.ndarray) -> np_.ndarray:
        lat1 = np_.radians(lat1)
        lon1 = np_.radians(lon1)
        lat2 = np_.radians(lat2)
        lon2 = np_.radians(lon2)

        y = np_.sin(lon2-lon1) * np_.cos(lat2)
        x = np_.cos(lat1)*np_.sin(lat2) - np_.sin(lat1)*np_.cos(lat2)*np_.cos(lon2-lon1)
        return np_.degrees(np_.arctan2(y, x))

    @overload
    @staticmethod
    def distance(lat1:T, lon1:T, lat2:T, lon2:T) -> T: ...

    @staticmethod
    def distance(lat1:np_.ndarray, lon1:np_.ndarray, lat2:np_.ndarray, lon2:np_.ndarray) -> np_.ndarray:
        # Radius of earth in KM
        R = 6378.137
        d_lat = lat2 * math.pi / 180 - lat1 * math.pi / 180
        d_lon = lon2 * math.pi / 180 - lon1 * math.pi / 180
        a = np_.sin(d_lat/2.0)**2 + \
            np_.cos(lat1 * math.pi / 180) * np_.cos(lat2 * math.pi / 180) * np_.sin(d_lon/2.0)**2
        c = 2 * np_.arctan2(np_.sqrt(a), np_.sqrt(1-a))
        distance = R * c * 1000
        return distance

    @staticmethod
    def bearing_diff(a:np_.float64_1d[ax.time], b:np_.float64_1d[ax.time]) -> np_.float64_1d[ax.time]:
        # apply bearing diff to all elements
        return np_.array([bearing_diff(a[i], b[i]) for i in range(len(a))])
