
import itertools
import matplotlib.pyplot as plt
import math
from _Utils.os_wrapper import os
import pandas as pd


from   B_Model.AbstractModel import Model as _Model_
from   D_DataLoader.TrajectorySeparator.DataLoader import DataLoader
import D_DataLoader.TrajectorySeparator.Utils as SU
import D_DataLoader.Utils as U
from   E_Trainer.AbstractTrainer import Trainer as AbstractTrainer
import E_Trainer.TrajectorySeparator.Utils as TU

import _Utils.Color as C
from   _Utils.Color import prntC
from   _Utils.DebugGui import GUI
import _Utils.FeatureGetter as FG
import _Utils.Limits as Limits
from   _Utils.numpy import np, ax
from   _Utils.plotADSB import PLT
from   _Utils.ProgressBar import ProgressBar




# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

PBM_NAME = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
ARTIFACTS = "./_Artifacts/"

EVAL_FOLDER = "./A_Dataset/TrajectorySeparator/"

BAR = ProgressBar(max = 100)

MAX_PLOT = 10
NB_PLOT = {}

DEBUG_PER_TIMESTEPS = False
DEBUG = True
NB_STEPS = 3
DEBUG_PLOT = "TrajectorySeparatorDebug"

# |====================================================================================================================
# | STEP ENUM FOR MESSAGE ASSOCIATION ALGORITHM (DONE IN MULTIPLE STEPS)
# |====================================================================================================================


class STEP:
    FIRST = 1
    AREA = 2
    COMBINATION = 3
    NEAREST = 4
    FINISHED = 5

    @staticmethod
    def to_string(step:int) -> str:
        if (step == STEP.FIRST):
            return "FIRST"
        if (step == STEP.AREA):
            return "AREA"
        if (step == STEP.COMBINATION):
            return "COMBINATION"
        if (step == STEP.NEAREST):
            return "NEAREST"
        if (step == STEP.FINISHED):
            return "FINISHED"
        return "UNKNOWN"

# |====================================================================================================================
# | TRAINER CLASS (THIS ONE DOES NOT TRAIN BUT I KEEP THE CLASS NAME FOR CONSISTENCY)
# |====================================================================================================================


class Trainer(AbstractTrainer):


# |====================================================================================================================
# |     INITIALIZATION
# |====================================================================================================================

    def __init__(self, CTX:dict, Model:"type[_Model_]") -> None:
        # Public attributes
        self.CTX = CTX
        self.model:_Model_ = Model(CTX)
        self.__makes_artifacts__()
        self.__init_GUI__()
        self.viz_model(self.ARTIFACTS)
        self.dl = DataLoader(CTX)

        # Private attributes
        self.__nth_debug__ = 0


    def __makes_artifacts__(self) -> None:
        self.ARTIFACTS = ARTIFACTS+PBM_NAME+"/"+self.model.name
        if not os.path.exists(ARTIFACTS):
            os.makedirs(ARTIFACTS)
        if not os.path.exists(ARTIFACTS+PBM_NAME):
            os.makedirs(ARTIFACTS+PBM_NAME)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)
        # os.system("rm -rf "+self.ARTIFACTS+"/*")
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
# |     TRAINING FUNCTIONS (NO TRAINING NEEDED FOR TRAJECTORY SEPARATOR AS IT IS SOLVED BY A CLASSICAL ALGORITHM)
# |====================================================================================================================

    def train(self) -> None:
        pass

# |====================================================================================================================
# |     MAKING PREDICTIONS FROM RAW ADSB MESSAGES
# |====================================================================================================================


    def __give_icao__(self, assoc:np.int64_1d[ax.sample], msgs:"list[dict[str,object]]", tags:"list[str]")\
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




# |====================================================================================================================
# |     FIRST ASSOCIATION TO REDUCE THE COMPLEXITY OF NEXT ASSOCIATIONS
# |====================================================================================================================

    def __first_associate__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.int64_1d[ax.sample], np.int64_1d[ax.sample]]":

        VALID_DIST = 20

        assoc = np.full((mat.shape[0],), -1, dtype=int)
        assoc_inv = np.full((mat.shape[1],), -1, dtype=int)

        run = True
        while (run):
            run = False

            for y_i in range(mat.shape[1]):
                if (assoc_inv[y_i] == -1):
                    min_1, min_2 = TU.argmin2(mat[:, y_i])
                    if (min_1 == min_2 or mat[min_2, y_i] - mat[min_1, y_i] > VALID_DIST):
                        assoc_inv[y_i] = min_1
                        assoc[min_1] = y_i
                        mat[min_1, :] = Limits.INT_MAX
                        mat[:, y_i] = Limits.INT_MAX
                        run = True



        return assoc, mat


# |====================================================================================================================
# |     SECOND ASSOCIATION : ASSOCIATE ONES THAT ARE ALONE IN THEIR AREA
# |====================================================================================================================

    def __area_associate__(self, mat:np.float64_2d[ax.sample, ax.sample],
                                 y_:np.float64_2d[ax.sample, ax.feature],
                                 y :np.float64_2d[ax.sample, ax.feature])\
            -> "tuple[np.int64_1d[ax.sample], np.int64_1d[ax.sample]]":

        assoc = np.full((mat.shape[0],), -1, dtype=int)
        assoc_inv = np.full((mat.shape[1],), -1, dtype=int)

        pred_mat = TU.loss_matrix(y_, y_)
        for i in range(len(pred_mat)):
            pred_mat[i, i] = Limits.INT_MAX

        true_mat = TU.loss_matrix(y, y)
        for i in range(len(true_mat)):
            true_mat[i, i] = Limits.INT_MAX


        run = True
        while (run):
            run = False

            for y_i in range(mat.shape[1]):
                if (assoc_inv[y_i] == -1):
                    yi = np.argmin(mat[:, y_i])
                    area = mat[yi, y_i]
                    if not(TU.have_n_inf_to(pred_mat[:, y_i], area) or TU.have_n_inf_to(true_mat[yi, :], area)):
                        assoc_inv[y_i] = yi
                        assoc[yi] = y_i
                        mat[yi, :] = Limits.INT_MAX
                        mat[:, y_i] = Limits.INT_MAX
                        pred_mat[yi, :] = Limits.INT_MAX
                        pred_mat[:, y_i] = Limits.INT_MAX
                        true_mat[yi, :] = Limits.INT_MAX
                        true_mat[:, y_i] = Limits.INT_MAX
                        run = True


        return assoc, mat


# |====================================================================================================================
# |    TESTING COMBINATIONS TO FIND THE BEST MSG ASSOCIATION POSSIBLE
# |====================================================================================================================

    def __combination_associate__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.int64_1d[ax.sample], np.int64_1d[ax.sample]]":

        assoc = np.full((mat.shape[0],), -1, dtype=int)
        assoc[:mat.shape[1]] = np.arange(mat.shape[1])

        all_perms = itertools.permutations(assoc)
        best_loss = Limits.INT_MAX
        for perm in all_perms:
            perm = np.array(perm)
            loss = TU.eval_association(mat, perm)
            if (loss < best_loss):
                best_loss = loss
                assoc = perm

        return assoc, mat

    def __nearest_associate__(self, mat:np.float64_2d[ax.sample, ax.sample])\
            -> "tuple[np.int64_1d[ax.sample], np.int64_1d[ax.sample]]":

        assoc = np.full((mat.shape[0],), -1, dtype=int)

        for _ in range(len(assoc)):
            min_yi, min_y_i = TU.mat_argmin(mat)
            assoc[min_yi] = min_y_i
            mat[min_yi, :] = Limits.INT_MAX
            mat[:, min_y_i] = Limits.INT_MAX

        return assoc, mat



    def __debug_assoc__(self, y:np.float64_2d[ax.sample, ax.feature], y_:np.float64_2d[ax.sample, ax.feature],
                        remain_y:np.int64_1d[ax.sample], remain_y_:np.int64_1d[ax.sample],
                        sub_assoc:np.int64_1d[ax.sample],
                        sample:"list[np.float64_2d[ax.time, ax.feature]]",
                        step:STEP) -> None:

        if (not DEBUG_PER_TIMESTEPS):
            return
        if (len(y_) <= 1):
            return

        if (self.__nth_debug__ == 0):
            self.__nth_debug__ = 1
            # first debug call : create figure
            PLT.figure(DEBUG_PLOT, 0, 0, 0, 0,
                    figsize = (15, 30), sub_plots=(NB_STEPS, 1), display_map=[[False]] * NB_STEPS)

        axis = -1
        if (step == STEP.FIRST):
            axis = 0
        if (step == STEP.AREA):
            axis = 1
        if (step == STEP.COMBINATION or step == STEP.NEAREST):
            axis = 2

        if (axis == -1):
            prntC(C.ERROR, "Unknown step in debug")

        self.__plot_assoc__(axis, y[remain_y],y_[remain_y_],
                            [sample[i] for i in remain_y_],
                            sub_assoc)

        PLT.subplot(DEBUG_PLOT, axis, 0).title("Step : "+STEP.to_string(step))

        if (axis == NB_STEPS-1):
            # last debug call : save figure
            PLT.savefig(DEBUG_PLOT, os.path.join(self.ARTIFACTS, "debug_assoc.png"))
            input("Press Enter to continue...")
            self.__nth_debug__ = 0

    def __associate__(self, msgs:"list[dict[str,object]]",
                            y_:np.float64_2d[ax.sample, ax.feature], tags:"list[str]",
                            sample:"list[np.float64_2d[ax.time, ax.feature]]") -> "list[str]":

        if (len(tags) == 0):
            return self.__give_icao__(np.full((len(msgs),), -1, dtype=int), msgs, tags)

        y = np.array([[
            msgs[i]["latitude"],
            msgs[i]["longitude"]]
                for i in range(len(msgs))], dtype=np.float64)

        # mat give loss between [msg_i, pred_j]
        # if mat is INT_MAX, it means that the association is done and is now impossible
        mat = TU.loss_matrix(y, y_)

        # associate the msg[i] with pred[j] as best as possible.
        # goal is to minimize : sum of mat[i, j] is minimal
        assoc = np.full((len(y),), -1, dtype=int)


        step = STEP.FIRST
        while step != STEP.FINISHED:
            sub_mat, remain_y, remain_y_ = TU.compute_remaining_loss_matrix(mat, assoc)

            # skip area-step if we can
            if (step == STEP.AREA):
                if (len(remain_y) <= 5 or len(remain_y_) <= 5):
                    step = NB_STEPS
            # skip combination-step if we must (too many combinations)
            # and do nearest-neighbor instead
            if (step == STEP.COMBINATION):
                if (len(remain_y) > 5 and len(remain_y_) > 5):
                    step = STEP.NEAREST

            # apply STEP
            if (step == STEP.FIRST):
                sub_assoc, sub_mat = self.__first_associate__(sub_mat)
            elif (step == STEP.AREA):
                sub_assoc, sub_mat = self.__area_associate__(sub_mat, y_[remain_y_], y[remain_y])
            elif (step == STEP.COMBINATION):
                sub_assoc, sub_mat = self.__combination_associate__(sub_mat)
            elif (step == STEP.NEAREST):
                sub_assoc, sub_mat = self.__nearest_associate__(sub_mat)


            self.__debug_assoc__(y, y_, remain_y, remain_y_, sub_assoc, sample, step)

            assoc, mat = TU.apply_sub_associations(assoc,     mat,
                                    sub_assoc, sub_mat,
                                    remain_y,  remain_y_)

            # next step
            if (step == STEP.FIRST):
                step = STEP.AREA
            elif (step == STEP.AREA):
                step = STEP.COMBINATION
            elif (step == STEP.COMBINATION):
                step = STEP.FINISHED
            elif (step == STEP.NEAREST):
                step = STEP.FINISHED





        return self.__give_icao__(assoc, msgs, tags)


# |--------------------------------------------------------------------------------------------------------------------
# | PLOTING ASSOCIATIONS
# |--------------------------------------------------------------------------------------------------------------------

    def __plot_assoc__(self, plot_axis:int,
                            y :np.float64_2d[ax.sample, ax.feature],
                            y_:np.float64_2d[ax.sample, ax.feature],
                            sample:"list[np.float64_2d[ax.time, ax.feature]]",
                            assoc:np.int64_1d[ax.sample]) -> None:

        SAMPLE_LEN = 5
        for i in range(len(sample)):
            label = "Trajectories" if i == 0 else None
            PLT.subplot(DEBUG_PLOT, plot_axis, 0)\
               .scatter(sample[i][-SAMPLE_LEN:, 0], sample[i][-SAMPLE_LEN:, 1], c="tab:blue", label=label)

            PLT.subplot(DEBUG_PLOT, plot_axis, 0)\
               .plot(sample[i][-SAMPLE_LEN:, 0], sample[i][-SAMPLE_LEN:, 1], c="tab:blue")



        PLT.subplot(DEBUG_PLOT, plot_axis, 0)\
           .scatter(y_[:, 0], y_[:, 1], c="tab:purple", marker="x", label="Predictions")

        PLT.subplot(DEBUG_PLOT, plot_axis, 0)\
           .scatter(y[:, 0], y[:, 1], c="tab:green", marker="+", label="Messages")

        for i in range(len(assoc)):
            if (assoc[i] != -1):
                PLT.subplot(DEBUG_PLOT, plot_axis, 0)\
                   .plot([y[i, 0], y_[assoc[i], 0]], [y[i, 1], y_[assoc[i], 1]], c="tab:orange", linestyle="--")


        if (plot_axis == 0):
            PLT.subplot(DEBUG_PLOT, plot_axis, 0).legend()

        PLT.subplot(DEBUG_PLOT, plot_axis, 0).set_aspect("equal")




# |====================================================================================================================
# |   PREDICTION FUNCTION
# |====================================================================================================================



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

            if (__EVAL__ and DEBUG):
                self.__plot__(sample, msgs, y_)

            msg_tags = self.__associate__(msgs, y_, tags, sample)
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

        mat = TU.loss_matrix(y, y_)

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

