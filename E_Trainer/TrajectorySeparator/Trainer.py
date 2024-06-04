
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools



from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.TrajectorySeparator.DataLoader import DataLoader
import D_DataLoader.TrajectorySeparator.Utils as SU
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer

import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.DebugGui import GUI
from   _Utils.ProgressBar import ProgressBar
import _Utils.geographic_maths as GEO
import _Utils.Limits as Limits
import _Utils.FeatureGetter as FG
from _Utils.numpy import np, ax



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


    def __loss_matrix__(self, y:np.float32_2d[ax.sample, ax.feature], y_:np.float32_2d[ax.sample, ax.feature])\
            -> np.float32_2d[ax.sample, ax.sample]:

        mat = np.zeros((len(y), len(y_)), dtype=np.float64)
        for i in range(len(y)):
            for j in range(len(y_)):
                mat[i, j] = GEO.distance(y[i][0], y[i][1], y_[j][0], y_[j][1])
        return mat

    def __give_icao__(self, assoc:np.int32_1d[ax.sample], msgs:"list[dict[str,object]]", tags:"list[str]")\
            -> "list[str]":

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





    def __print_mat__(self, mat:np.float64_2d[ax.sample, ax.sample]) -> None:

        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if (mat[i, j] == Limits.INT_MAX):
                    print(str("X").rjust(3), end=" ")
                else:
                    print(str(int(mat[i, j])).rjust(3), end=" ")
            print()
        print()
        print()

    @staticmethod
    def __min_i_2__(vec:np.float64_1d) -> "tuple[int, int]":
        min_i_1 = 0
        min_i_2 = 0
        min_val_1 = Limits.INT_MAX
        min_val_2 = Limits.INT_MAX

        for i in range(len(vec)):

            if (vec[i] < min_val_1):
                # transfer min_1 to min_2
                min_i_2 = min_i_1
                min_val_2 = min_val_1

                # update min_1
                min_i_1 = i
                min_val_1 = vec[i]

            elif (vec[i] < min_val_2):
                min_i_2 = i
                min_val_2 = vec[i]

        return min_i_1, min_i_2


# |====================================================================================================================
# |    UTILS TO MAKE ASSOCIATIONS ON SUB-SET OF OUR PROBLEM
# |====================================================================================================================

    def __get_remaining_mat__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.float64_2d[ax.sample, ax.sample], np.int32_1d[ax.sample], np.int32_1d[ax.sample]]":

        remain_y = np.where(mat[:, 0] != Limits.INT_MAX)[0]
        remain_y_ = np.where(mat[0, :] != Limits.INT_MAX)[0]
        return mat[remain_y, :][:, remain_y_], remain_y, remain_y_

    def __add_new_associations__(self,
                assoc:np.int32_1d, mat:np.float64_2d[ax.sample, ax.sample],
                sub_assoc:np.int32_1d, sub_mat:np.float64_2d[ax.sample, ax.sample],
                remain_y:np.int32_1d, remain_y_:np.int32_1d,) -> "tuple[np.int32_1d, np.int32_1d]":

        for i in range(len(remain_y)):
            if (sub_assoc[i] != -1):
                assoc[remain_y[i]] = remain_y_[sub_assoc[i]]

        for i in range(len(remain_y)):
            for j in range(len(remain_y_)):
                mat[remain_y[i], remain_y_[j]] = sub_mat[i, j]

        return assoc, mat

# |====================================================================================================================
# |     FIRST ASSOCIATION TO REDUCE THE COMPLEXITY OF NEXT ASSOCIATIONS
# |====================================================================================================================

    def __first_associate__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.int32_1d[ax.sample], np.int32_1d[ax.sample]]":

        VALID_DIST = 20

        assoc = np.full((mat.shape[0],), -1, dtype=int)
        assoc_inv = np.full((mat.shape[1],), -1, dtype=int)

        run = True
        while (run):
            run = False

            for y_i in range(mat.shape[1]):
                if (assoc_inv[y_i] == -1):
                    min_1, min_2 = self.__min_i_2__(mat[:, y_i])
                    if (min_1 == min_2 or mat[min_2, y_i] - mat[min_1, y_i] > VALID_DIST):
                        assoc_inv[y_i] = min_1
                        assoc[min_1] = y_i
                        mat[min_1, :] = Limits.INT_MAX
                        mat[:, y_i] = Limits.INT_MAX
                        run = True

        return assoc, mat


# |====================================================================================================================
# |    TESTING COMBINATIONS TO FIND THE BEST ASSOCIATION
# |====================================================================================================================

    def __compute_genetic_loss__(self, mat:np.float32_2d[ax.sample, ax.sample], combination:np.int32_1d[ax.sample])\
            -> np.float32:

        loss = 0
        for i in range(len(combination)):
            if (combination[i] != -1):
                loss += mat[i, combination[i]]
        return loss

    def __mutate_combination__(self, combination:np.int32_1d[ax.sample], n=3) -> np.int32_1d[ax.sample]:
        # permute n elements together in the combination
        if (len(combination) < n):
            return combination

        to_permute = np.random.choice(np.arange(len(combination)), n, replace=False)
        permuted = np.random.permutation(to_permute)
        new_combination = combination.copy()
        for i in range(n):
            new_combination[to_permute[i]] = combination[permuted[i]]
        return new_combination

    def __combination_associate__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.int32_1d[ax.sample], np.int32_1d[ax.sample]]":

        print(mat.shape)

        assoc = np.full((mat.shape[0],), -1, dtype=int)
        assoc[:mat.shape[1]] = np.arange(mat.shape[1])

        all_perms = itertools.permutations(assoc)
        best_loss = Limits.INT_MAX
        for perm in all_perms:
            perm = np.array(perm)
            loss = self.__compute_genetic_loss__(mat, perm)
            if (loss < best_loss):
                best_loss = loss
                assoc = perm

        return assoc, mat


    def __associate__(self, msgs:"list[dict[str,object]]",
                                y_:np.float32_2d[ax.sample, ax.feature], tags:"list[str]") -> "list[str]":

        if (len(tags) == 0):
            return self.__give_icao__(np.full((len(msgs),), -1, dtype=int), msgs, tags)

        y = np.array([[
            msgs[i]["latitude"],
            msgs[i]["longitude"]]
                for i in range(len(msgs))], dtype=np.float64)

        # mat give loss between [msg_i, pred_j]
        # if mat is INT_MAX, it means that the association is done and is now impossible
        mat = self.__loss_matrix__(y, y_)

        # associate the msg[i] with pred[j] as best as possible.
        # goal is to minimize : sum of mat[i, j] is minimal
        assoc = np.full((len(y),), -1, dtype=int)


        # easy case : make associations for flight where there is only one relevant prediction
        sub_mat, remain_y, remain_y_ = self.__get_remaining_mat__(mat)
        sub_assoc, sub_mat = self.__first_associate__(sub_mat)
        assoc, mat = self.__add_new_associations__(
                assoc, mat,
                sub_assoc, sub_mat,
                remain_y, remain_y_)

        print()
        self.__print_mat__(mat)

        # hard case : make associations for flight where there are multiple relevant predictions
        sub_mat, remain_y, remain_y_ = self.__get_remaining_mat__(mat)
        sub_assoc, sub_mat = self.__combination_associate__(sub_mat)
        assoc, mat = self.__add_new_associations__(
            assoc, mat,
            sub_assoc, sub_mat,
            remain_y, remain_y_)

        return self.__give_icao__(assoc, msgs, tags)





    def predict(self, x:"list[dict[str,object]]", __EVAL__:bool=False) -> "list[str]":

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

            # if (__EVAL__):
            #     self.__plot__(sample, msgs, y_)

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

    def __plot__(self, sample:"list[np.float64_2d[ax.time, ax.feature]]",
                       msgs:"list[dict[str, object]]", y_:np.ndarray) -> None:
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
                for i in range(len(msgs))], dtype=np.float64)

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


    def __gen_eval_batch__(self, folder:"str") \
            -> "tuple[dict[int, list[dict[str, object]]], dict[int, list[str]]]":

        messages:  "dict[int, list[dict[str, object]]]" = {}
        true_icaos:"dict[int, list[str]]" = {}

        _, y, flight = self.dl.__load_dataset__(self.CTX, EVAL_FOLDER+folder)
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



    def eval(self) -> dict:
        folders = [f for f in os.listdir(EVAL_FOLDER) if f != "base_files" and "." not in f]

        global_acc = 0

        for folder in folders:
            prntC(C.INFO, "Evaluating on", C.BLUE, folder)
            self.dl.streamer.clear()
            messages, icaos = self.__gen_eval_batch__(folder)
            BAR.reset(0, len(messages))

            n = 0
            df = pd.DataFrame()
            true, preds = [], []

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

            self.__generate_eval_artifacts__(df, folder)

            _, mean_acc = self.__eval_stats__(true, preds)
            global_acc += mean_acc



        return {"accuracy": global_acc/len(folders)*100.0}


# |====================================================================================================================
# |     EVALUATION STATISTICS
# |====================================================================================================================

    def __eval_stats__(self, true, preds):
        acc = self.accuracy(true, preds)
        mean_acc = 0
        for icao in acc:
            mean_acc += acc[icao][0]/acc[icao][1]
        mean_acc /= len(acc)

        self.__print_eval_stats__(acc, mean_acc)

        return acc, mean_acc

    def __print_eval_stats__(self, acc, mean_acc):
        prntC(C.INFO, "Accuracy : ", C.YELLOW, round(mean_acc*100.0, 2))
        for icao in acc:
            prntC(C.INFO_, icao, " : ", C.YELLOW, round(acc[icao][0]/acc[icao][1]*100.0, 2))
        prntC()

    def __generate_eval_artifacts__(self, df, folder)-> None:
        # check that artifacts folder exists
        if not os.path.exists(os.path.join(self.ARTIFACTS, folder)):
            os.makedirs(os.path.join(self.ARTIFACTS, folder))

        if not os.path.exists(os.path.join(self.ARTIFACTS, folder, "separated_csv")):
            os.makedirs(os.path.join(self.ARTIFACTS, folder, "separated_csv"))
        else:
            os.system("rm -rf "+os.path.join(self.ARTIFACTS, folder, "separated_csv")+"/*")

        # save model's predictions
        df.to_csv(os.path.join(self.ARTIFACTS, folder, "predictions.csv"), index=False)

        # save separated trajectories
        for icao24 in df["icao24"].unique():
            sub_df = df[df["icao24"] == icao24]
            sub_df.to_csv(os.path.join(self.ARTIFACTS, folder, "separated_csv", icao24+".csv"), index=False)


















    # def __associate__(self, msgs:"list[dict[str,object]]",
    #                           y_:np.float32_2d[ax.sample, ax.feature], tags:"list[str]") -> "list[str]":

    #     y = np.array([[
    #         msgs[i]["latitude"],
    #         msgs[i]["longitude"]]
    #             for i in range(len(msgs))], dtype=np.float64)
    #     mat = self.__loss_matrix__(y, y_)

    #     # -1 means no association
    #     # otherwise, the index of the associated prediction
    #     assoc = np.full((len(y),), -1, dtype=int)

    #     for _ in range(min(len(y), len(y_))):
    #         min_i = np.argmin(mat)
    #         min_msg, min_pred = min_i // len(y_), min_i % len(y_)


    #         if (mat[min_msg, min_pred] < Limits.INT_MAX):
    #             assoc[min_msg] = min_pred

    #         mat[min_msg, :] = Limits.INT_MAX
    #         mat[:, min_pred] = Limits.INT_MAX

    #     return self.__give_icao__(assoc, msgs, tags)