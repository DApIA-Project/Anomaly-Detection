

import math



def predict(lat, lon, bearing, distance):
    """
    Predict the next position of an aircraft
    """
    R = 6371*1000 # m
    lat = math.radians(lat)
    lon = math.radians(lon)
    bearing = math.radians(bearing)

    lat2 = math.asin(math.sin(lat)*math.cos(distance/R) + math.cos(lat)*math.sin(distance/R)*math.cos(bearing))
    lon2 = lon + math.atan2(math.sin(bearing)*math.sin(distance/R)*math.cos(lat), math.cos(distance/R)-math.sin(lat)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
    return math.degrees(math.atan2(y, x))

# compute distance based on lat, lon
def distance(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2.0)**2 + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    return distance

def bearing_diff(a, b):
    a = a % 360
    b = b % 360

    # compute relative angle
    diff = b - a

    if (diff > 180):
        diff -= 360
    elif (diff < -180):
        diff += 360
    return diff