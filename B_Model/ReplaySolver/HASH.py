
import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from sklearn.cluster import KMeans
from B_Model.ReplaySolver.Utils.clusters import *

import numpy as np

import time

import os



def Xrotation(x, y, z, r):
    xx = x
    yx = y * np.cos(r) - z * np.sin(r)
    zx = y * np.sin(r) + z * np.cos(r)
    return xx, yx, zx

def Yrotation(x, y, z, r):
    xy = x * np.cos(r) + z * np.sin(r)
    yy = y
    zy = -x * np.sin(r) + z * np.cos(r)
    return xy, yy, zy

def Zrotation(x, y, z, r):
    xz = x * np.cos(r) - y * np.sin(r)
    yz = x * np.sin(r) + y * np.cos(r)
    zz = z
    return xz, yz, zz


def rotate(olat, olon, otrack, lat, lon):
    Y = olat
    Z = -olon
    R = otrack

    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))

    r = np.radians(Z)
    x, y, z = Zrotation(x, y, z, r)

    r = np.radians(Y)
    x, y, z = Yrotation(x, y, z, r)

    r = np.radians(R)
    x, y, z = Xrotation(x, y, z, r)

    # convert back cartesian to lat lon
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lat, lon


def serialize_lat_lon(lat, lon, CTX):
    _x, _y, _z = np.cos(np.radians(lon)) * np.cos(np.radians(lat)), np.sin(np.radians(lon)) * np.cos(np.radians(lat)), np.sin(np.radians(lat))
    x, y, z = _x.copy(), _y.copy(), _z.copy()
    for t in range(0, len(lat)):
        if (t == 0):
            x[t], y[t], z[t] = (1, 0, 0)


        x[t], y[t], z[t] = Zrotation(x[t], y[t], z[t], np.radians(-lon[t-1]))
        x[t], y[t], z[t] = Yrotation(x[t], y[t], z[t], np.radians(lat[t-1]))

        if (t >= 2):
            lx, ly, lz = _x[t-2], _y[t-2], _z[t-2]
            lx, ly, lz = Zrotation(lx, ly, lz, np.radians(-lon[t-1]))
            lx, ly, lz = Yrotation(lx, ly, lz, np.radians(lat[t-1]))
            R = np.arctan2(-lz, -ly)

        else:
            R = np.arctan2(z[t], y[t])


        x[t], y[t], z[t] = Xrotation(x[t], y[t], z[t], -R)


    x = y
    y = z
    # a = np.arctan2(y, x)
    # d = np.sqrt(x**2 + y**2) 
    return x[1:], y[1:]


    
def compare_ts_fp(timeserie_v, timeserie_o, fingerprint_v, fingerprint_o):
    # transform str and fingerprint into a set of tuples (digit, occurence)
    len_fp = sum([fingerprint_o[i] for i in range(len(fingerprint_o))])
    loc = 0
    maxloc = 0
    maxSimilarity = 0
    _len_fp = len(fingerprint_v)
    _len_ts = len(timeserie_v)

    for shift in range(len(timeserie_v) - len(fingerprint_v) + 1):
        
        # compare the two sets
        ts_i = shift
        fp_i = 0
        ts_i_occ = timeserie_o[ts_i] - min(timeserie_o[ts_i], fingerprint_o[fp_i])
        fp_i_occ = 0
        nb = 0
        max_backup_delta = ts_i_occ

        while True:

            timeserie_ts_i_1 = timeserie_o[ts_i]
            timeserie_fp_i_1 = fingerprint_o[fp_i]
            l = min(timeserie_ts_i_1 - ts_i_occ, timeserie_fp_i_1 - fp_i_occ)
            ts_i_occ += l
            fp_i_occ += l

            if (timeserie_v[ts_i] == fingerprint_v[fp_i]):
                nb += l

            # print(ts_i_occ, fp_i_occ)
            if fp_i_occ >= timeserie_fp_i_1:
                fp_i += 1
                fp_i_occ = 0
                if (fp_i == _len_fp):
                    break

            if ts_i_occ >= timeserie_ts_i_1:
                ts_i += 1
                ts_i_occ = 0
                if (ts_i == _len_ts):
                    break

        
        if nb > maxSimilarity:
            maxSimilarity = nb
            maxloc = loc + max_backup_delta

            if (maxSimilarity == len_fp):
                break

        loc += timeserie_o[shift]


    return maxSimilarity, maxloc


def fingerprint_from_ts(ts, t, min_length):
    nb_elements = 0
    wi = 0
    fp_v = [0] * min_length
    fp_o = [0] * min_length
    length = 0
    while (nb_elements < min_length and t < len(ts[0])):
        fp_v[wi] = ts[0][t]
        fp_o[wi] = ts[1][t]
        length += ts[1][t]
        nb_elements += ts[1][t]
        t += 1
        wi += 1

    if (nb_elements < min_length):
        return (), len(ts), 0

    return (fp_v[:wi], fp_o[:wi]), t, length

def compare_ts_ts(check_ts, true_ts, min_length):
    similarity = 0.0
    nb = 0
    t = 0
    while (t < len(check_ts[0])):

        fp, t, length = fingerprint_from_ts(check_ts, t, min_length)
        if (len(fp) == 0): # STOP !
            break

        if (len(fp[0]) >= 3):

            match, loc = compare_ts_fp(true_ts[0], true_ts[1], fp[0], fp[1])

            if (match == length): # perfect match !
                return match

            similarity += match
            nb += 1

        if (nb >= 5 and similarity / nb < 0.8 * min_length): 
            # print("break")
            break
        
    similarity /= nb
    return similarity


      

class Model(AbstactModel):
    """
    Convolutional neural network model for 
    aircraft classification based on 
    recordings of ADS-B data fragment.

    Parameters:
    ------------

    CTX: dict
        The hyperparameters context


    Attributes:
    ------------

    name: str (MENDATORY)
        The name of the model for mlflow logs
    
    Methods:
    ---------

    predict(x): (MENDATORY)
        return the prediction of the model

    compute_loss(x, y): (MENDATORY)
        return the loss and the prediction associated to x, y and y_

    training_step(x, y): (MENDATORY)
        do one training step.
        return the loss and the prediction of the model for this batch

    visualize(save_path):
        Generate a visualization of the model's architecture
    """

    name = "CPU"

    def __init__(self, CTX:dict):
        
        # load context
        self.CTX = CTX

        slice = 2
        levels = 1
        self.CLUSTERS = slice * levels + 1
        self.MIN_CHANGE = 3
        self.clustering = SliceCluster(slices=slice, levels=levels)
        self.ts = {}
        self.hashes = {}


    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]()=+-_.,:;!?@'\"#$%^&*~`|\\/<>°§£¤µ"
    def compute_hash(self, fp):
        hash = ""
        for i in range(len(fp)):
            hash += self.ALPHABET[fp[i]]
        return hash


        
    def predict(self, x):
        return self.compute_loss(x, [""]*len(x))[1]
        


    def compute_loss(self, x, y):
        """
        Make a prediction and compute the lossSequelize
        that will be used for training
        """
        """
        Make prediction for x 
        """
        start = time.time()
        serialized_x = []
        serialized_y = []
        for s in range(len(x)):
            sample = x[s]
            lat_i = self.CTX["FEATURE_MAP"]["latitude"]
            lon_i = self.CTX["FEATURE_MAP"]["longitude"]
            track_i = self.CTX["FEATURE_MAP"]["track"]
            groundspeed_i = self.CTX["FEATURE_MAP"]["groundspeed"]
            vertical_rate_i = self.CTX["FEATURE_MAP"]["vertical_rate"]
            altitude_i = self.CTX["FEATURE_MAP"]["altitude"]
            geoaltitude_i = self.CTX["FEATURE_MAP"]["geoaltitude"]

            sx, sy = serialize_lat_lon(sample[:, lat_i], sample[:, lon_i], self.CTX)
            d = np.sqrt(sx**2 + sy**2)
            sx[d > 0.0001] = 0
            sy[d > 0.0001] = 0

            fsx, fsy = serialize_lat_lon(sample[:, lat_i], -sample[:, lon_i], self.CTX)
            d = np.sqrt(sx**2 + sy**2)
            fsx[d > 0.0001] = 0
            fsy[d > 0.0001] = 0

            serialized_x.append(sx)
            serialized_y.append(sy)
            serialized_x.append(fsx)
            serialized_y.append(fsy)

        # convert files to fingerprint
        print("comparing !")
        ts = []
        for i in range(len(serialized_x)):
            X = np.array([serialized_x[i], serialized_y[i]]).T.astype(np.double)
            fingerprint = self.clustering.predict(X)
            ts.append(fingerprint)

    
            
        # plot ts
        plt_ts = []
        plot_labels = []
        max_len = 0
        for i in range(int(len(ts)/2)):
            # show test timeseries
            plt_ts.append([])
            for j in range(len(ts[i*2])):
                plt_ts[-1].append(ts[i*2][j])

                if (len(plt_ts[-1]) > 128):
                    if (len(plt_ts[-1]) > max_len):
                        max_len = len(plt_ts[-1])
                    break
            plot_labels.append(y[i])

            # show true timeseries
            if (y[i] in self.ts):
                plt_ts.append([])
                for j in range(len(self.ts[y[i]])):
                    plt_ts[-1].append(self.ts[y[i]][j])

                    if (len(plt_ts[-1]) > 128):
                        if (len(plt_ts[-1]) > max_len):
                            max_len = len(plt_ts[-1])
                        break
                plot_labels.append("TRUE")


        import matplotlib.pyplot as plt
        # on each line, plot dot with color corresponding to the hash
        colors = ['#cbff00', '#00ffb2', '#3200ff', '#ff004c', '#0cff00', '#008cff', '#f200ff', '#ff7200']
        fig, ax = plt.subplots(figsize=(20, len(plt_ts)/max_len*20))
        for i in range(len(plt_ts)):
            for j in range(len(plt_ts[i])):
                # make rectangle
                ax.add_patch(plt.Rectangle((j+0.1, i+0.1), 0.8, 0.8, color=colors[plt_ts[i][j]%len(colors)]))

        # set axis
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, len(plt_ts))
        # yticks = filenamesSequelize
        ax.set_yticks(np.array(range(len(plt_ts))) + 0.5)
        ax.set_yticklabels(plot_labels)
        # if history = 32
        # xticks = 0, 32, 64, 96, 128
        ax.set_xticks(np.array(range(0, max_len, 32)) + 0.5)
        ax.set_xticklabels(range(0, max_len, 32))

        ax.invert_yaxis()

        plt.title("Hashed timeseries")
        plt.savefig("./_Artifacts/hashed_timeseries.png",bbox_inches='tight', dpi=300)
        plt.clf()


        # compute hashes for each timeseries
        res = []
        acc = 0
        for i in range(int(len(ts)/2)):

            hashes = []
            for j in range(len(ts[i]) - self.CTX["HISTORY"] + 1):
                hashes.append(self.compute_hash(ts[i*2+0][j:j+self.CTX["HISTORY"]]))
                hashes.append(self.compute_hash(ts[i*2+1][j:j+self.CTX["HISTORY"]]))

            # find matches
            matches = []
            for hash in hashes:
                get = self.hashes.get(hash, [])

                for match in get:
                    matches.append(match)

            # occ
            occ = {}
            for match in matches:
                occ[match[0]] = occ.get(match[0], 0) + 1
            
            # sort
            occ = list(occ.items())
            occ.sort(key=lambda x: x[1], reverse=True)

            # print the 3 bests with their occurence
            # print("Best matches : ", occ[:3])
            if (len(occ) > 0):
                if (occ[0][1] >= 20):
                    res.append(occ[0][0])
                else:
                    res.append("unknown")
            else:
                res.append("unknown")

            print("pred : ", res[-1], " true : ", y[i], " similarity : ", occ[0][1] if len(occ) > 0 else 0, "matches count:", len(occ))
            acc += res[-1] == y[i]

        print("elapsed time : ", time.time() - start)
        print()

        return acc / len(res), res

    def training_step(self, x, y):
        """
        Fit the model, add new data !
        """

        serialized_x = []
        serialized_y = []
        files = []
        for s in range(len(x)):
            print("\rSerializeation:", s+1, "/", len(x), end="")
            sample = x[s]
            match = y[s]
            lat_i = self.CTX["FEATURE_MAP"]["latitude"]
            lon_i = self.CTX["FEATURE_MAP"]["longitude"]
            track_i = self.CTX["FEATURE_MAP"]["track"]
            groundspeed_i = self.CTX["FEATURE_MAP"]["groundspeed"]
            vertical_rate_i = self.CTX["FEATURE_MAP"]["vertical_rate"]
            altitude_i = self.CTX["FEATURE_MAP"]["altitude"]
            geoaltitude_i = self.CTX["FEATURE_MAP"]["geoaltitude"]

            sx, sy = serialize_lat_lon(sample[:, lat_i], sample[:, lon_i], self.CTX)
            d = np.sqrt(sx**2 + sy**2)
            sx[d > 0.0001] = 0
            sy[d > 0.0001] = 0

            # concat
            serialized_x = np.concatenate((serialized_x, sx))
            serialized_y = np.concatenate((serialized_y, sy))
            files += [match] * len(sx)
        print()


        serialized_x = np.array(serialized_x)
        serialized_y = np.array(serialized_y)

        sx = serialized_x[:]
        sy = serialized_y[:]

        X = np.array([sx, sy]).T
        self.clustering.fit(X)

        X = np.array([serialized_x, serialized_y]).T
        labels = self.clustering.predict(X)

        # plot clusters
        import matplotlib.pyplot as plt
        colors = ['#cbff00', '#00ffb2', '#3200ff', '#ff004c', '#0cff00', '#008cff', '#f200ff', '#ff7200']
        #square figure
        plt.figure(figsize=(10,10))

        plt.scatter(serialized_x[:50000], serialized_y[:50000], c=[colors[i%len(colors)] for i in labels[:50000]], s=0.35)
        plt.title("Clusters")
        plt.axis('equal')
        plt.savefig("./_Artifacts/clusters.png")
        plt.clf()

        
        # remove each files that already have been fingerprinted
        to_drop = []
        for i in range(len(files)):
            if (files[i] in self.ts):
                to_drop.append(i)

        for f in to_drop:
            self.ts.pop(files[f], None)

        # for each new file generate the fingerprint
        for i in range(len(labels)):
            if (files[i] not in self.ts):
                self.ts[files[i]] = []
            
            self.ts[files[i]].append(labels[i])   

        lens = [len(self.ts[f]) for f in self.ts]
        mean = sum(lens) / len(lens)
        print("Mean length : ", mean)    
        print("Max length : ", max(lens)) 

        # for each new files add new hashes
        self.hashes = {}
        hash_count = 0 
        ii = 0
        print()
        for file in self.ts:
            print("\rHashing", ii+1, "/", len(self.ts), end="")
            # split each file into fingerprints
            for w in range(len(self.ts[file]) - self.CTX["HISTORY"] + 1):
                fp = self.ts[file][w:w+self.CTX["HISTORY"]]

                # check if the fingerprint is interesting
                changes = 1
                last = fp[0]
                for i in range(1, len(fp)):
                    if (fp[i] != last):
                        changes += 1
                        last = fp[i]
                
                if (changes >= self.MIN_CHANGE):
                    # compute hash
                    hash = self.compute_hash(fp)
                    hash_count += 1

                    # add hash to the list
                    if (hash not in self.hashes):
                        self.hashes[hash] = []
                    self.hashes[hash].append((file, w))

            ii += 1
        print()
        print("Hash count : ", hash_count)

        # stat : count the hash that has collisions
        collisions = 0
        to_del = []
        for hash in self.hashes:
            if (len(self.hashes[hash]) > 1):
                collisions += 1
                to_del.append(hash)

        for hash in to_del:
            self.hashes.pop(hash, None)
        
        print("Collisions : ", collisions, "/", len(self.hashes))
        print(list(self.hashes.keys())[:100])
        return 0, 0





    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """
        

    def getVariables(self):
        """
        Return the variables of the model
        """
        return self.hashes, self.ts, self.clustering.getVariables()


    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        self.hashes = variables[0]
        self.ts = variables[1]
        self.clustering.setVariables(variables[2])
