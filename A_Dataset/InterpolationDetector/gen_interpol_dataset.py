
import os
import pandas as pd
from matplotlib import pyplot as plt
import improvedArrows 
import numpy as np

import warnings
warnings.filterwarnings("error")


IMPACT_AREA = 5 # desired seconds before and after used to compute the beziers vectors
debug = False


def prntD(*args, **kwargs):
    if (debug):
        print(*args, **kwargs)


def compute_score_at_r(i, tolerance, gaps, consecutives):
    """
    Compute the number of messages that can be used to interpolate the gap at the right of the index i
    """
    r_score = 1
    tolerates = 0
    shift = 1

    while (i+shift < len(gaps)):
        nb_missing = gaps[i+shift] - 1
        tolerates += nb_missing
        
        if (tolerates > tolerance):
            return r_score

        seq = consecutives[i+shift]
        r_score += seq
        shift += seq

    return len(gaps) - (i+1)


def compute_score_at_l(i, tolerance, gaps, consecutives):
    """
    Compute the number of messages that can be used to interpolate the gap at the left of the index i
    """
    l_score = 1
    tolerates = 0
    shift = -1
    
    while (i+shift >= 0):
        gab_b = gaps[i+shift] - 1
        tolerates += gab_b
        
        if (tolerates > tolerance):
            return l_score
    
        seq = consecutives[i+shift]
        l_score += seq
        shift -= seq
        
    return i+1



def best_loc_to_interpolate(gaps, consecutives):
    
    """search for the best location to interpolate a gap"""

    largest_gap = np.max(gaps) + 1

    found_tolerable = False
    tolerable = 0

    last_change_best_i = -1
    last_change_best_lr_score = (0, 0)
    last_change_tolerable = 0

    while not(found_tolerable):
        found_gap = 0

        best_i = -1
        best_score = -largest_gap
        last_change_best_score = -largest_gap
        best_lr_score = (0, 0)
    
        for i in range(0, len(gaps)-1):

            if (gaps[i] > 1):
                found_gap += 1

                r = compute_score_at_r(i, tolerable, gaps, consecutives)
                l = compute_score_at_l(i, tolerable, gaps, consecutives)
                score = min(r, l)
                if (score > 1):
                    score = score - (gaps[i] - 1) / 2.0

                    found_tolerable = True
                    if (score > best_score):
                        best_score = score
                        best_lr_score = (l, r)
                        best_i = i
                
                else:
                    if (score > last_change_best_score):
                        last_change_best_i = i
                        last_change_best_lr_score = (l, r)
                        last_change_best_score = score - (gaps[i] - 1) / 2.0
                        last_change_tolerable = tolerable
            
        prntD("found_gap", found_gap, "found_tolerable", found_tolerable, "tolerable", tolerable, largest_gap)

        if (found_gap == 0):
            return -1, (0, 0), -1, tolerable
        tolerable += 1


        if (tolerable > largest_gap):
            prntD("use last change whit gap", gaps[last_change_best_i], "score", last_change_best_score, "tolerated", last_change_tolerable)
            return last_change_best_i, last_change_best_lr_score, last_change_best_score, last_change_tolerable


    return best_i, best_lr_score, best_score, tolerable-1

def normalize_ts(ts):
    ts = ts - ts[0]
    return ts


def compute_gaps(ts):

    ts = normalize_ts(ts)

    gaps = ts.copy()
    consecutives = np.ones(len(ts), dtype=np.int32)

    consecutive_no_gap = 0
    for i in range(0, len(ts)-1):
        gaps[i] = ts[i+1] - ts[i]
    gaps[-1] = 1

    for i in range(0, len(ts)):
        if (gaps[i] == 1):
            consecutive_no_gap += 1
        else:
            consecutives[i-consecutive_no_gap:i] = consecutive_no_gap
            consecutive_no_gap = 0
    
    consecutives[-consecutive_no_gap:] = consecutive_no_gap
    #cap gap_plot at 25
    if (debug):
        gaps_plot = np.minimum(gaps, 25)
        plt.figure(figsize=(14, 4))
        plt.bar(np.arange(len(gaps))*3, gaps_plot, label="gaps size")
        plt.bar(np.arange(len(gaps))*3+1, consecutives, label="number of consecutives timestamps")
        # set x ticks
        x_ticks = np.arange(0, len(gaps)*3, 3)
        x_labels = [str(i)+"\n"+str(ts[i]) for i in range(len(ts))]
        plt.xticks(x_ticks, x_labels)
        plt.legend()
        plt.show()

    return gaps, consecutives


def intersect(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
    x_denom = ((Ax-Bx)*(Cy-Dy) - (Ay-By)*(Cx-Dx))
    if (x_denom == 0):
        return None, None
    y_denom = ((Ax-Bx)*(Cy-Dy) - (Ay-By)*(Cx-Dx))
    if (y_denom == 0):
        return None, None
    
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    x = ((Ax*By - Ay*Bx)*(Cx-Dx) - (Ax-Bx)*(Cx*Dy - Cy*Dx)) / x_denom
    y = ((Ax*By - Ay*Bx)*(Cy-Dy) - (Ay-By)*(Cx*Dy - Cy*Dx)) / y_denom
    return x, y


def angle_shift(a1, a2):
    diff = a2 - a1
    # prntD(diff)
    q = int(diff / (2 * np.pi))
    diff -= q * 2 * np.pi
    # prntD(diff)

    if (diff > np.pi):
        diff -= 2 * np.pi

    if (diff < -np.pi):
        diff += 2 * np.pi
        
    return diff


def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    b = (sum_y - a * sum_x) / n

    return a, b



def predict_next_pts(x, y, g):
    if (len(x) < 2):
        return x[-1], y[-1]
    
    if (len(x) == 2):
        vx = x[-1] - x[0]
        vy = y[-1] - y[0]
        return x[-1] + vx, y[-1] + vy
    
    d = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        d[i] = np.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2)
    d = np.mean(d)

    prntD("gaps", g)

    t = np.zeros(len(x))
    for i in range(len(x)-1):
        t[i+1] = t[i] + g[i]

    a = np.zeros(len(x)-1)
    a_x = np.zeros(len(x))
    for i in range(len(x)-1):
        vx = x[i+1] - x[i]
        vy = y[i+1] - y[i]
        if (vx == 0 and vy == 0 and i > 0):
            a[i] = a[i-1]
        else:
            a[i] = np.arctan2(vy, vx)
        a_x[i+1] = (t[i] + t[i+1]) / 2.0
    a_x = a_x[1:]
    a_x = a_x - a_x[0]


    rots = np.zeros(len(a)-1)
    rotx = np.zeros(len(a))
    for i in range(len(a)-1):
        rots[i] = angle_shift(a[i], a[i+1])
        rotx[i+1] = (a_x[i] + a_x[i+1]) / 2.0
    rotx = rotx[1:]
    rotx = rotx - rotx[0]

    if (len(rots) == 1):
        next_rot = rots[0]
    else:
        reg_a, reg_b = linear_regression(rotx, rots)
        next_rot = reg_a * (rotx[-1] + 1) + reg_b

    next_angle = a[-1] + next_rot / 2.0
    nx, ny = x[-1] + d * np.cos(next_angle), y[-1] + d * np.sin(next_angle)


    if (debug):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].plot(x, y, 'o-')
        ax[0].plot(nx, ny, 'o')
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_title("Predicted next point")

        ax[1].plot(a_x, a, 'o')
        ax[1].plot(a_x[-1]+1, next_angle, 'o')
        ax[1].set_title("Angles")
        
        ax[2].plot(rotx, rots, 'o')
        ax[2].plot(rotx[-1]+1, next_rot, 'o')
        if (len(rots) > 1):
            ax[2].plot(rotx, reg_a * rotx + reg_b)
        ax[2].set_title("Rotations")

        plt.show()

    return nx, ny


def cubic(t, p0, p1, p2, p3):
    return p0 + t * (-3.0 * p0 + 3.0 * p1) + t**2.0 * (3.0 * p0 - 6.0 * p1 + 3.0 * p2) + t**3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)

def largest_under(a, u):
    b, t = 0, len(a) - 1
    while (t - b > 1):
        m = ((t + b) +1) // 2

        if (a[m] > u):
            t = m
        else:
            b = m

    res = b
    # if (a[t] <= u):
    #     res = t
    return res


def cubic_bezier_eq(p, pts):
    
    if (p[0] == p[2] and p[2] == p[4] and p[4] == p[6] and\
        p[1] == p[3] and p[3] == p[5] and p[5] == p[7]):
        return np.full(pts, p[0]), np.full(pts, p[1])
    
    SAMPLES = 25
    # pre-compute arc-lenghts
    x, y = np.zeros(SAMPLES), np.zeros(SAMPLES)
    a = np.zeros(SAMPLES)
    for i in range(SAMPLES):
        t = i / (SAMPLES-1)
        x[i], y[i] = cubic(t, p[0], p[2], p[4], p[6]), cubic(t, p[1], p[3], p[5], p[7])
        if (i > 0):
            a[i] = a[i-1] + np.sqrt((x[i-1] - x[i]) ** 2 + (y[i-1] - y[i]) ** 2)

    lenght = a[-1]
    # a = a / lenght
    x, y = np.zeros(pts), np.zeros(pts)
    for i in range(pts):
        u = i / (pts-1) * lenght
        if (u > lenght): u = lenght
        ti = largest_under(a, u)
        a_before = a[ti]
        a_after = a[ti+1]
        l_fra = (u - a_before) / (a_after-a_before)
        t = (ti +  l_fra) / (SAMPLES-1)
        x[i], y[i] = cubic(t, p[0], p[2], p[4], p[6]), cubic(t, p[1], p[3], p[5], p[7])

    return x, y



def gap_filler(x, y, ts):
    global debug

    gaps, consecutives = compute_gaps(ts)

    best_i, best_lr_score, best_score, tolerable = best_loc_to_interpolate(gaps, consecutives)
    if (best_i == -1):
        return x, y, ts, False
    l_score = best_lr_score[0]
    r_score = best_lr_score[1]

    prntD("best_i", best_i, "tolerated :", tolerable, "with score l:", l_score, "r:", r_score, "tot:", min(l_score, r_score), "-", "("+str(gaps[best_i])+"-1)/2", "=", min(l_score, r_score) - (gaps[best_i] - 1)/2, "(verif:", best_score, ")")

    l_score = min(l_score, IMPACT_AREA)
    r_score = min(r_score, IMPACT_AREA)
        
    pts = int(gaps[best_i] + 1)

    lxs = x[best_i-l_score+1:best_i+1]
    lys = y[best_i-l_score+1:best_i+1]
    lgaps = gaps[best_i-l_score+1:best_i]

    rxs = x[best_i+1:best_i+1+r_score]
    rys = y[best_i+1:best_i+1+r_score]
    rgaps = gaps[best_i+1:best_i+r_score]


    lnx, lny = predict_next_pts(lxs, lys,  lgaps)
    rnx, rny = predict_next_pts(rxs[::-1], rys[::-1], rgaps[::-1])

    lxV, lyV = lnx - lxs[-1], lny - lys[-1]
    rxV, ryV = rnx - rxs[0], rny - rys[0]


    Ox, Oy = intersect(
        lxs[-1], lys[-1], lxs[-1] + lxV, lys[-1] + lyV,
        rxs[0], rys[0], rxs[0] + rxV, rys[0] + ryV)
    
    VABx, VABy = (rxs[0] - lxs[-1]), (rys[0] - lys[-1])
    VABxy2 = VABx * VABx + VABy * VABy
    
    if (Ox is None or VABxy2 <= 10e-7):
        scalar = 0
    else:   
        VAOx, VAOy = (Ox - lxs[-1]), (Oy - lys[-1])
        scalar = (VAOx * VABx + VAOy * VABy) / (VABxy2)

    if (scalar > 0 and scalar < 1):
        lxV, lyV = (Ox - lxs[-1]) * (2.0/3.0), (Oy - lys[-1]) * (2.0/3.0)
        rxV, ryV = (Ox - rxs[0]) * (2.0/3.0), (Oy - rys[0]) * (2.0/3.0)
        prntD("a", lxV, lyV, rxV, ryV)

    else:
        lVl = np.sqrt((lxV) ** 2 + (lyV) ** 2)
        rVl = np.sqrt((rxV) ** 2 + (ryV) ** 2)
        ABl = np.sqrt((rxs[0] - lxs[-1]) ** 2 + (rys[0] - lys[-1]) ** 2)

        # critical case where there is only one point on the left or one on the right
        # in this case connect the two points with a line
        if (ABl <= 10e-7):
            lxV, lyV = 0, 0
            rxV, ryV = 0, 0
            prntD(lxs, lys, rxs, rys)
            prntD(rxs[0], rys[0], lxs[-1], lys[-1])
        else:
            if (lVl == 0):
                lxV = rxs[0] - lxs[-1]
                lyV = rys[0] - lys[-1]
                lVl = np.sqrt((lxV) ** 2 + (lyV) ** 2)
            if (rVl == 0):
                rxV = lxs[-1] - rxs[0]
                ryV = lys[-1] - rys[0]
                rVl = np.sqrt((rxV) ** 2 + (ryV) ** 2)



            lxV, lyV = lxV / lVl * ABl * (1.0/3.0), lyV / lVl * ABl * (1.0/3.0)
            rxV, ryV = rxV / rVl * ABl * (1.0/3.0), ryV / rVl * ABl * (1.0/3.0)
    

    #param
    Ax = lxs[-1]
    Ay = lys[-1]
    Bx = lxs[-1] + lxV
    By = lys[-1] + lyV
    Cx = rxs[0] + rxV
    Cy = rys[0] + ryV
    Dx = rxs[0]
    Dy = rys[0]
    
    
    prntD(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy)

    interpX, interpY = cubic_bezier_eq((Ax, Ay, Bx, By, Cx, Cy, Dx, Dy), pts)
    interpX = interpX[1:-1]
    interpY = interpY[1:-1]

    prntD(interpY, interpX)
    
    inter_ts = np.arange(ts[best_i]+1, ts[best_i+1], 1)
    # prntD("inter_ts", ts[best_i], ts[best_i+1], inter_ts)
    

    x = np.concatenate([x[:best_i+1], interpX, x[best_i+1:]])
    y = np.concatenate([y[:best_i+1], interpY, y[best_i+1:]])
    ts = np.concatenate([ts[:best_i+1], inter_ts, ts[best_i+1:]])

    if (debug):
        plt.figure(figsize=(8, 8))
        plt.plot(lxs, lys, 'o')
        plt.plot(rxs, rys, 'o')
        plt.arrow(lxs[-1], lys[-1], lxV, lyV, "--", color="black", linewidth=1)
        prntD(rxs[0], rys[0], rxV, ryV)
        plt.arrow(rxs[0], rys[0], rxV, ryV, "--", color="black", linewidth=1)
        # plt.scatter(Ox, Oy, marker='o', color='red')
        plt.plot(interpX, interpY, 'o', markersize=2)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.ashow()

        prntD()
        


    return x, y, ts, True

def gen_interpolated_traj(df, interpolated_ratio, noise, icao24=None):
    
    indices = list(range(1, len(df)-1))
    np.random.shuffle(indices)
    to_remove = int(len(df) * interpolated_ratio)
    df_i = df.drop(indices[:to_remove]).reset_index(drop=True)
    
    
    lats, lons, timestamps = df_i["latitude"].values, df_i["longitude"].values, df_i["timestamp"].values

    fill_gaps = True
    while (fill_gaps):
        lats, lons, timestamps, fill_gaps = gap_filler(lats, lons, timestamps)
        
    interp_df = pd.DataFrame({"latitude":lats, "longitude":lons, "timestamp":timestamps})

    # merde on timestamp
    df_i = pd.merge(df, interp_df, on="timestamp", how="inner")
    df_i["latitude"] = df_i["latitude_y"]
    df_i["longitude"] = df_i["longitude_y"]
    df_i.drop(columns=["latitude_x", "longitude_x", "latitude_y", "longitude_y"], inplace=True)

    # df = df.fillna(method="ffill")
    with pd.option_context('future.no_silent_downcasting', True):
        out = df_i.ffill().infer_objects(copy=False)
        
    out["latitude"] = out["latitude"] + np.random.normal(0, noise, len(out))
    out["longitude"] = out["longitude"] + np.random.normal(0, noise, len(out))
    
    if (icao24 is not None):
        out["icao24"] = icao24
    
    return out
    

def gen_train():
    NOISE_LEVEL = [0, 0, 0, 0, 0, 0, 0.00004, 0.00004, 0.00004, 0.00004, 0.00008, 0.00008, 0.00015]
    BASE_PATH = "./Train/base/"
    INTERP_PATH = f"./Train/interp_{min(NOISE_LEVEL)}-{max(NOISE_LEVEL)}/"

    if not os.path.exists(INTERP_PATH):
        os.makedirs(INTERP_PATH)
    else:
        os.system("rm -rf "+INTERP_PATH+"*")

    files = os.listdir(BASE_PATH)

    # remove all files in interp folder
    os.system("rm -rf "+INTERP_PATH+"*")


    PERCENTAGE = 0.9

    for i in range(len(files)):
        file = files[i]
        noise = np.random.choice(NOISE_LEVEL)

        base_traj = pd.read_csv(BASE_PATH+file, dtype={"icao24":str, "callsign":str})
        traj = gen_interpolated_traj(base_traj, PERCENTAGE, noise)
        
        filename = file.split(".")[0]
        traj.to_csv(INTERP_PATH+filename+f"_{noise}.csv", index=False)
        
        print(i+1, "/", len(files), "done")
        
def gen_eval():
    BASE_PATH = "./Eval/base/"
    PERCENTAGE = 0.9
    files = os.listdir(BASE_PATH)
    file_i = 0
    
    os.system("rm -rf ./Eval/auto_*.csv")

    noises = np.concatenate([np.arange(0.00002, 0.00022, 0.00002), np.zeros(5)])
    for noise in noises:
        
        file = files[file_i]
        
        base_traj = pd.read_csv(BASE_PATH+file, dtype={"icao24":str, "callsign":str})
        traj = gen_interpolated_traj(base_traj, PERCENTAGE, noise, icao24=base_traj["icao24"].values[0]+"i")
        
        filename = file.split(".")[0]
        base_traj.to_csv(f"./Eval/auto_{filename}_{noise}.csv", index=False)
        traj.to_csv(f"./Eval/auto_{filename}_{noise}_interp.csv", index=False)
        
        file_i += 1
        

gen_eval()
