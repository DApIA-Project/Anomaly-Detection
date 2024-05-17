
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer
from D_DataLoader.TrajectorySeparator.DataLoader import DataLoader
import D_DataLoader.TrajectorySeparator.Utils as SU
import D_DataLoader.Utils as U

from _Utils.save import write, load
import _Utils.Color as C
from _Utils.Color import prntC
from _Utils.ProgressBar import ProgressBar
from _Utils.DebugGui import GUI
import _Utils.geographic_maths as GEO
import _Utils.Limits as Limits
import _Utils.FeatureGetter as FG



# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
ARTIFACTS = "./_Artifacts/"

EVAL_FOLDER = "./A_Dataset/TrajectorySeparator/"

BAR = ProgressBar(max = 100)

MAX_PLOT = 10
NB_PLOT = {}

class Trainer(AbstractTrainer):


# |====================================================================================================================
# |     INITIALIZATION
# |====================================================================================================================

    def __init__(self, CTX:dict, Model:"type[_Model_]") -> None:
        super().__init__(CTX, Model)
        self.CTX = CTX

        self.model:_Model_ = Model(CTX)
        self.__makes_artifacts__()
        self.__init_GUI__()
        self.dl = DataLoader(CTX)


    def __makes_artifacts__(self) -> None:
        self.ARTIFACTS = ARTIFACTS+PBM_NAME+"/"+self.model.name
        if not os.path.exists(ARTIFACTS):
            os.makedirs(ARTIFACTS)
        if not os.path.exists(ARTIFACTS+PBM_NAME):
            os.makedirs(ARTIFACTS+PBM_NAME)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)

        os.system("rm -rf "+self.ARTIFACTS+"/*")

        if not os.path.exists(self.ARTIFACTS+"/Predictions_plot"):
            os.makedirs(self.ARTIFACTS+"/Predictions_plot")


    def __init_GUI__(self) -> None:
        GUI.visualize("/Training", GUI.TEXT, "Loading...")

# |====================================================================================================================
# |     SAVE & LOAD MODEL'S VARIABLES
# |====================================================================================================================

    def save(self) -> None:
        pass

    def load(self, path:str=None) -> None:
        pass

# |====================================================================================================================
# |     TRAINING FUNCTIONS (NO TRAINING NEEDED FOR THIS)
# |====================================================================================================================

    def train(self):
        pass

# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================

    def __loss_matrix__(self, y, y_):
        mat = np.zeros((len(y), len(y_)), dtype=np.float32)
        for i in range(len(y)):
            for j in range(len(y_)):
                mat[i][j] = GEO.distance(y[i][0], y[i][1], y_[j][0], y_[j][1])
        return mat

    def __associate__(self, msgs, y_, tags):

        y = np.array([[
            msgs[i]["latitude"],
            msgs[i]["longitude"]]
                for i in range(len(msgs))], dtype=np.float32)

        mat = self.__loss_matrix__(y, y_)
        assoc = np.full((len(y),), -1, dtype=int)

        for _ in range(min(len(y), len(y_))):
            min_i = np.argmin(mat)
            min_msg, min_pred = min_i // len(y_), min_i % len(y_)


            if (mat[min_msg, min_pred] < 1000000000):
                assoc[min_msg] = min_pred

            mat[min_msg, :] = Limits.INT_MAX
            mat[:, min_pred] = Limits.INT_MAX

        icao = msgs[0]["icao24"]
        msg_tags = []
        nb_new = 0
        for i in range(len(msgs)):
            if (assoc[i] != -1):
                msg_tags.append(tags[assoc[i]])
            else:
                msg_tags.append(icao + "_" + str(len(tags) + nb_new))
                nb_new += 1

        return msg_tags


    def predict(self, x:"list[dict[str,object]]", __EVAL__:bool=False) -> np.ndarray:

        per_icao:dict[str, list[dict[str, object]]] = {}

        for i in range(len(x)):
            icao = x[i]["icao24"]
            if (icao not in per_icao):
                per_icao[icao] = []
            per_icao[icao].append(x[i])

        for icao in per_icao.keys():
            msgs = per_icao[icao]
            TS = msgs[0]["timestamp"]
            sample, tags = self.dl.streamer.get_flights_with_icao(icao, TS)
            timestamps = [TS for _ in range(len(sample))]

            y_ = self.model.predict(sample, timestamps)

            if (__EVAL__):
                self.__plot__(sample, msgs, y_)

            msg_tags = self.__associate__(msgs, y_, tags)
            for i in range(len(msgs)):
                msgs[i]["tag"] = msg_tags[i]
                self.dl.streamer.stream(msgs[i])

        tags = [x[i]["tag"] for i in range(len(x))]
        return tags

# |====================================================================================================================
# |    PLOTTING PREDICTIONS
# |====================================================================================================================


    def __is_plot_interesting__(self, WIN:int,
                                sample:"list[np.ndarray]", msgs:"list[dict[str, object]]",
                                __force__:bool=False)\
            -> "tuple[bool, list[tuple[tuple[float, float], tuple[float, float]]]|None]":

        global NB_PLOT

        # if empty : no plot
        if (len(sample) <= 0):
            return False, None

        # # if too many plots
        if (NB_PLOT.get(msgs[0]["icao24"], 0) > MAX_PLOT and not __force__):
            return False, None

        if (np.random.rand() > 0.333 and not __force__):
            return False, None


        # only plot if there are at least 2 flights in the same area
        is_interesting = False
        boxs = []
        for s in range(len(sample)):

            mi = min(len(sample[s]), WIN)
            size = math.sqrt((sample[s][-1, 0] - sample[s][-mi, 0])**2 \
                            + (sample[s][-1, 1] - sample[s][-mi, 1])**2)

            if (size < 0.0001): size = 0.0001

            box = ((sample[s][-1, 0]-size, sample[s][-1, 1]-size),
                   (sample[s][-1, 0]+size, sample[s][-1, 1]+size))
            boxs.append(box)

            nb_flights = 0
            for i in range(len(sample)):
                lat, lon = FG.lat_lon(sample[i][-1])
                if (lat > box[0][0] and lat < box[1][0] and lon > box[0][1] and lon < box[1][1]):
                    nb_flights += 1

            if (nb_flights >= 2):
                is_interesting = True

        if (not is_interesting and not __force__):
            return False, None

        NB_PLOT[msgs[0]["icao24"]] = NB_PLOT.get(msgs[0]["icao24"], 0) + 1

        return True, boxs

    def __plot__(self, sample:"list[np.ndarray]", msgs:"list[dict[str, object]]", y_:np.ndarray) -> None:
        WIN = 20
        COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:cyan"]
        plot, boxs = self.__is_plot_interesting__(WIN, sample, msgs)
        if not plot:
            return

        # subplot for each flight
        c = math.ceil(math.sqrt(len(sample)))
        r = math.ceil(len(sample) / c)
        _, ax = plt.subplots(r, c, figsize=(10*c, 10*r))

        if (len(sample) == 1):
            ax = [ax]
        # flatten ax
        if (r > 1 and c > 1):
            ax = [a for b in ax for a in b]

        y = np.array([[
            msgs[i]["latitude"],
            msgs[i]["longitude"]]
                for i in range(len(msgs))], dtype=np.float32)

        mat = self.__loss_matrix__(y, y_)
        # print(mat)

        for s in range(len(ax)):
            if (s < len(sample)):
                for i in range(len(sample)):
                    ax[s].plot(sample[i][-WIN:, 0], sample[i][-WIN:, 1], "o-", c=COLORS[i%len(COLORS)])
                    ax[s].scatter(y_[:, 0], y_[:, 1], c="red", marker="x")

                for m in msgs:
                    ax[s].scatter(m["latitude"], m["longitude"], c="green", marker="+")

                ax[s].set_title(str(mat[:, s]))

            b = min(s, len(boxs)-1)
            ax[s].set_xlim(boxs[b][0][0], boxs[b][1][0])
            ax[s].set_ylim(boxs[b][0][1], boxs[b][1][1])
            ax[s].set_aspect("equal", "box")

        # save
        plt.savefig(os.path.join(
            self.ARTIFACTS,
            "Predictions_plot",
            "plot_"+str(msgs[0]["icao24"])+"_"+str(msgs[0]["timestamp"])+".png"))

# |====================================================================================================================
# |     EVALUATION
# |====================================================================================================================


    def __gen_eval_batch__(self, folders:"list[str]") \
            -> "tuple[dict[int, list[dict[str, object]]], dict[int, list[str]]]":

        messages:  "dict[int, list[dict[str, object]]]" = {}
        true_icaos:"dict[int, list[str]]" = {}

        for d in range(len(folders)):
            _, y, flight = self.dl.__load_dataset__(self.CTX, EVAL_FOLDER+folders[d])
            # transform flight df, into dict[timestamp, message]
            flight = flight.to_dict(orient="records")

            # insert in messages
            for i in range(len(flight)):
                t = flight[i]["timestamp"]
                if (t not in messages):
                    messages[t] = []
                    true_icaos[t] = []
                messages[t].append(flight[i])
                true_icaos[t].append(y[i])

        return messages, true_icaos


    def __next_msgs__(self, messages:"dict[int, list[dict[str, object]]]", icaos:"dict[int, list[str]]", t:int) \
        -> "tuple[list[dict[str:object]], list[str]]":
        return messages[t], icaos[t]



    def accuracy(self, y:"list[list[str]]", y_:"list[list[str]]")-> "dict[str, list[int, int]]":

        assoc:"dict[str, dict[str, str]]" = {}
        acc:"dict[str, list[int, int]]" = {}

        for t in range(len(y)):
            base_icao = [y[t][i].split("_")[0] for i in range(len(y[t]))]

            for i in range(len(y[t])):

                if (base_icao[i] not in assoc):
                    assoc[base_icao[i]] = {}
                assoc_icao = assoc[base_icao[i]]

                if (y[t][i] not in assoc_icao):
                    acc[base_icao[i]] = [0, 0]
                    assoc_icao[y[t][i]] = y_[t][i]

                if (assoc_icao[y[t][i]] == y_[t][i]):
                    acc[base_icao[i]][0] += 1

                assoc_icao[y[t][i]] = y_[t][i]
                acc[base_icao[i]][1] += 1

        return acc



    def eval(self):
        folders = [f for f in os.listdir(EVAL_FOLDER) if f != "base_files" and "." not in f]

        messages, icaos = self.__gen_eval_batch__(folders)

        BAR.reset(0, len(messages))
        n = 0
        df = pd.DataFrame()
        true = []
        preds = []
        for t in messages:
            x, y = self.__next_msgs__(messages, icaos, t)
            tags = self.predict(x, __EVAL__=True)

            true.append(y)
            preds.append(tags)

            for i in range(len(x)):
                x[i]["icao24"] = tags[i]

            df = pd.concat([df, pd.DataFrame(x)], ignore_index=True)

            n += 1
            BAR.update(n)

        df.to_csv(os.path.join(self.ARTIFACTS, "predictions.csv"), index=False)

        acc = self.accuracy(true, preds)
        mean_acc = 0
        for icao in acc:
            mean_acc += acc[icao][0]/acc[icao][1]
        prntC(C.INFO, "Accuracy : ", C.YELLOW, round(mean_acc/len(acc)*100.0, 2))
        for icao in acc:
            prntC(C.INFO_, icao, " : ", C.YELLOW, round(acc[icao][0]/acc[icao][1]*100.0, 2))


        return {"accuracy": mean_acc/len(acc)*100.0}

