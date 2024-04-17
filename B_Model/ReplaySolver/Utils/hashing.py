import numpy as np
import D_DataLoader.Utils as U


def serialize_lat_lon(lat, lon, CTX):
    _x, _y, _z = np.cos(np.radians(lon)) * np.cos(np.radians(lat)), np.sin(np.radians(lon)) * np.cos(np.radians(lat)), np.sin(np.radians(lat))
    x, y, z = _x.copy(), _y.copy(), _z.copy()
    for t in range(0, len(lat)):
        if (t == 0):
            x[t], y[t], z[t] = (1, 0, 0)

        x[t], y[t], z[t] = U.Zrotation(x[t], y[t], z[t], np.radians(-lon[t-1]))
        x[t], y[t], z[t] = U.Yrotation(x[t], y[t], z[t], np.radians(-lat[t-1]))

        if (t >= 2):
            lx, ly, lz = _x[t-2], _y[t-2], _z[t-2]
            lx, ly, lz = U.Zrotation(lx, ly, lz, np.radians(-lon[t-1]))
            lx, ly, lz = U.Yrotation(lx, ly, lz, np.radians(-lat[t-1]))
            R = -np.arctan2(-lz, -ly)

        else:
            R = -np.arctan2(z[t], y[t])

        x[t], y[t], z[t] = U.Xrotation(x[t], y[t], z[t], -R)

    x = y
    y = z
    return x[1:], y[1:]

def make_fingerprint(x, y, CTX):
    """
    Compute the fingerprint of a trajectory
    fingerprint is a string of L, R, N
    - L : left turn
    - R : right turn
    - N : netral (cannot be determined)
    """
        
    a = np.arctan2(y, x)
    d = np.sqrt(x**2 + y**2)

    MARGIN = 0.002

    res = ""
    for i in range(len(x)):
        if (d[i] < 0.000001):
            res+="N"

        else:
            if (a[i] < -MARGIN and a[i] > -np.pi + MARGIN):
                res += "L"
            elif (a[i] > MARGIN and a[i] < np.pi - MARGIN):
                res += "R"
            else:
                res += "N"
    return res
    
    
def sub_fp(fp):
    """
    List all the possible fingerprint variants when there is the N placeholder (same as . from regex)
    """
    # remplace each N by R and L variants
    res = [""]
    m_count = sum([1 for c in fp if c == "N"])
    if (m_count == 0):
        return [fp]
    if (m_count > 5):
        return ["N"*len(fp)]
    
    for c in range(len(fp)):
        if (fp[c] == "N"):
            l = len(res)
            res = res + res
            for i in range(l):
                res[i+l*0] += "R"
                res[i+l*1] += "L"
        else:
            for i in range(len(res)):
                res[i] += fp[c]
    return res
      

def compute_hash(fp):
    """
    Compute the hash of a fingerprint
    """
    hash = 0
    for i in range(len(fp)):
        v = 0
        if (fp[i] == "L"):
            v = 1
        elif (fp[i] == "R"):
            v= -1

        hash += v * (3 ** i)
    hash = abs(hash)
    return hash