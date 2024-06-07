from _Utils.numpy import np, ax
import time
from _Utils.os_wrapper import os

from B_Model.AbstractModel import Model as AbstactModel


import _Utils.FeatureGetter as FG
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.ProgressBar import ProgressBar
from B_Model.ReplaySolver.Utils import hashing


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================


PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
BAR = ProgressBar()



class Model(AbstactModel):

    name = "HASH"

# |====================================================================================================================
# |    INITIALIZATION
# |====================================================================================================================

    def __init__(self, CTX:dict) -> None:

        # Public attributes
        self.CTX = CTX
        self.MIN_CHANGE = 3
        self.ts = {}
        self.hashes = {}



# |====================================================================================================================
# |    PREDICTION
# |====================================================================================================================


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

            sx, sy = hashing.serialize_lat_lon(FG.lat(sample), FG.lon(sample), self.CTX)
            d = np.sqrt(sx**2 + sy**2)
            sx[d > 0.0001] = 0
            sy[d > 0.0001] = 0

            serialized_x.append(sx)
            serialized_y.append(sy)

        # convert files to fingerprint
        print("comparing !")
        ts = []
        for i in range(len(serialized_x)):
            ts.append(hashing.make_fingerprint(serialized_x[i], serialized_y[i], self.CTX))



        # plot ts
        plt_ts = []
        plot_labels = []
        max_len = 0
        for i in range(int(len(ts))):
            # show test timeseries
            plt_ts.append([])
            for j in range(len(ts[i])):
                plt_ts[-1].append(ts[i][j])

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
        colors = {"L":"#e74c3c", "R":"#3498db", "N":"#2ecc71"}
        fig, ax = plt.subplots(figsize=(20, len(plt_ts)/max_len*20))
        for i in range(len(plt_ts)):
            for j in range(len(plt_ts[i])):
                # make rectangle
                ax.add_patch(plt.Rectangle((j+0.1, i+0.1), 0.8, 0.8, color=colors[plt_ts[i][j]]))

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
        plt.savefig(self.ARTIFACTS+"/hashed_timeseries.png",bbox_inches='tight', dpi=300)
        plt.clf()


        # compute hashes for each timeseries
        res = []
        acc = 0
        for i in range(int(len(ts))):

            hashes = []
            for j in range(len(ts[i]) - self.CTX["HISTORY"] + 1):
                sub_fp = hashing.sub_fp(ts[i][j:j+self.CTX["HISTORY"]])

                for k in range(len(sub_fp)):
                    hashes.append(hashing.compute_hash(sub_fp[k]))

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
                    res.append("Unknown-flight")
            else:
                res.append("Unknown-flight")

            print("pred : ", res[-1], " true : ", y[i], " similarity : ", occ[0][1] if len(occ) > 0 else 0, "matches count:", len(occ))
            acc += res[-1] == y[i]

        print("elapsed time : ", time.time() - start)
        print()

        return acc / len(res), res

    def training_step(self, x, y):
        """
        Fit the model, add new data !
        """


        prntC(C.INFO, "Serializing data :")
        BAR.reset(max=len(x))

        serialized_x = np.zeros((0,))
        serialized_y = np.zeros((0,))
        files = []

        for s in range(len(x)):
            sample = x[s]
            match = y[s]

            sx, sy = hashing.serialize_lat_lon(FG.lat(sample), FG.lon(sample), self.CTX)

            d = np.sqrt(sx**2 + sy**2)
            sx[d > 0.0001] = 0
            sy[d > 0.0001] = 0

            # concat
            serialized_x = np.concatenate((serialized_x, sx))
            serialized_y = np.concatenate((serialized_y, sy))
            files += [match] * len(sx)

            BAR.update(s+1)


        serialized_x = np.array(serialized_x)
        serialized_y = np.array(serialized_y)

        labels = hashing.make_fingerprint(serialized_x, serialized_y, self.CTX)

        # # plot clusters
        # import matplotlib.pyplot as plt
        # colors = {"L":"#e74c3c", "R":"#3498db", "N":"#2ecc71"}
        # #square figure
        # plt.figure(figsize=(10,10))
        # plt.scatter(serialized_x[:50000], serialized_y[:50000], c=[colors[i] for i in labels[:50000]], s=0.35)
        # plt.title("Clusters")
        # plt.axis('equal')
        # plt.savefig(self.ARTIFACTS+"/clusters.png",bbox_inches='tight', dpi=300)
        # plt.clf()


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
        prntC(C.INFO, "Mean length : ", C.BLUE, mean)
        prntC(C.INFO, "Max length : ", C.BLUE, max(lens), "\n")

        # for each new files add new hashes
        prntC(C.INFO, "Hashing : ")
        BAR.reset(max=len(x))

        self.hashes = {}
        hash_count = 0
        fn = 0

        for file in self.ts:
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
                    fps = hashing.sub_fp(fp)
                    for f in range(len(fps)):
                        # compute hash
                        hash = hashing.compute_hash(fps[f])
                        hash_count += 1

                        # add hash to the list
                        if (hash not in self.hashes):
                            self.hashes[hash] = []
                        self.hashes[hash].append((file, w))

            fn += 1
            BAR.update(fn)

        prntC(C.INFO, "Hash count : ", C.BLUE, hash_count)

        # stat : count the hash that has collisions
        collisions = 0
        to_del = []
        for hash in self.hashes:
            if (len(self.hashes[hash]) > 1):
                collisions += 1
                to_del.append(hash)

        for hash in to_del:
            self.hashes.pop(hash, None)

        prntC(C.INFO, "Collisions : ", C.BLUE, collisions, C.RESET, "/", C.BLUE, len(self.hashes))
        prntC()
        return 0, 0





    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """


    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.hashes, self.ts


    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        self.hashes = variables[0]
        self.ts = variables[1]
