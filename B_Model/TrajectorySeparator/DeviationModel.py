
from B_Model.AbstractModel import Model as AbstactModel

from _Utils.FeatureGetter import FG_separator as FG
import _Utils.geographic_maths as GEO
import _Utils.Limits as Limits
import D_DataLoader.Utils as U

import matplotlib.pyplot as plt
from _Utils.plotADSB import Color

from numpy_typing import np, ax

def lerp(a:float, b:float, t:float) -> float:
    return a + (b-a)*t

def lreg(x:np.float64_1d[ax.time], y:np.float64_1d[ax.feature], w:np.float64_1d) -> "tuple[float, float]":

    m_y = np.mean(y)
    v_t = y - m_y
    y = m_y + v_t * w

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    b = (sum_y - a * sum_x) / n

    return a, b

def dumb_forecast(history:np.float64_2d[ax.time, ax.feature], t:int) -> "tuple[float, float]":
    return FG.lat(history[-1]), FG.lon(history[-1])

FORECAST_LEN = 20
WEIGHT = np.linspace(0.1, 1, FORECAST_LEN)
def forecast(history:np.float64_2d[ax.time, ax.feature], t:int) -> "tuple[float, float]":

    if (len(history) <= 2):
        return dumb_forecast(history, t)

    history = history[-FORECAST_LEN:]
    lat, lon, timestamp = FG.lat(history), FG.lon(history), FG.timestamp(history)
    timestamp = timestamp - t
    t = 0

    win_start = 0
    while (win_start < len(timestamp) and timestamp[win_start] <= -FORECAST_LEN):
        win_start += 1
    lat, lon, timestamp = lat[win_start:], lon[win_start:], timestamp[win_start:]

    if (len(timestamp) <= 10):
        return dumb_forecast(history, t)



    # clean data : remove point with to short distance
    distance = GEO.np.distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    distance = np.insert(distance, 0, Limits.INT_MAX)
    loc = np.where(distance > 10)[0]
    lat, lon, timestamp = lat[loc], lon[loc], timestamp[loc]
    weights_locs = np.array(timestamp + FORECAST_LEN, dtype=int)

    weights = WEIGHT[weights_locs]

    if (len(timestamp) <= 10):
        return dumb_forecast(history, t)


    # normalize at lat lon at (0, 0) to reduce deformation
    # normalize timestamp to start at 0
    Olat, Olon = lat[-1], lon[-1]
    lat, lon, _ = U.normalize_trajectory({}, lat, lon, np.nan, Olat, Olon, 0, True, False, False)



    # compute first derivative : angle & speed
    vx = lat[1:] - lat[:-1]
    vy = lon[1:] - lon[:-1]
    y_angle = np.arctan2(vy, vx) * 180 / np.pi
    y_distance = np.sqrt(vx**2 + vy**2)
    x_first = (timestamp[:-1] + timestamp[1:]) / 2

    # second derivative : rotation
    y_rotation = np.array([U.angle_diff(y_angle[i], y_angle[i+1]) for i in range(len(y_angle)-1)])
    x_second = (x_first[:-1] + x_first[1:]) / 2


    # forecast
    # angle
    rotation_a, rotation_b = lreg(x_second, y_rotation, w = weights[2:])
    next_rotation = rotation_a * t + rotation_b
    next_angle = y_angle[-1] + next_rotation

    # distance
    next_distance = np.average(y_distance, weights=weights[1:])

    # lat lon
    next_lat, next_lon = lat[-1] + next_distance * np.cos(next_angle * np.pi / 180), \
                         lon[-1] + next_distance * np.sin(next_angle * np.pi / 180)

    # denormalize
    pred = U.denormalize_trajectory({}, [next_lat], [next_lon], Olat, Olon, 0, True, False)
    perd_lat, pred_lon = pred[0][0], pred[1][0]


    # fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # ax[0, 0].title.set_text("Bearing")
    # ax[0, 0].plot(x_second, y_rotation, c=Color.TRAJECTORY)
    # ax[0, 0].scatter(x_second, y_rotation, c=Color.TRAJECTORY)
    # ax[0, 0].scatter(t, next_rotation, c=Color.PREDICTION)
    # ax[0, 0].plot([timestamp[0], t],
    #               [rotation_a * timestamp[0] + rotation_b, rotation_a * t + rotation_b], c=Color.PREDICTION)

    # ax[0, 1].title.set_text("Angle")
    # ax[0, 1].plot(x_first, y_angle, c=Color.TRAJECTORY)
    # ax[0, 1].scatter(x_first, y_angle, c=Color.TRAJECTORY)
    # ax[0, 1].scatter(t, next_angle, c=Color.PREDICTION)

    # ax[1, 1].title.set_text("Distance")
    # ax[1, 1].plot(x_first, y_distance, c=Color.TRAJECTORY)
    # ax[1, 1].scatter(x_first, y_distance, c=Color.TRAJECTORY)
    # ax[1, 1].scatter(t, next_distance, c=Color.PREDICTION)


    # ax[0, 2].title.set_text("Lat lon")
    # ax[0, 2].plot(lat, lon, c=Color.TRAJECTORY)
    # ax[0, 2].scatter(lat, lon, c=Color.TRAJECTORY)
    # ax[0, 2].scatter(next_lat, next_lon, c=Color.PREDICTION)
    # ax[0, 2].axis("equal")

    # ax[1, 2].title.set_text("pred !")
    # ax[1, 2].plot(FG.lat(history), FG.lon(history), c=Color.TRAJECTORY)
    # ax[1, 2].scatter(FG.lat(history), FG.lon(history), c=Color.TRAJECTORY)
    # ax[1, 2].scatter(perd_lat, pred_lon, c=Color.PREDICTION)
    # ax[1, 2].axis("equal")


    # fig.tight_layout()
    # plt.savefig("test.png")
    # plt.clf()
    # input("...")

    return perd_lat, pred_lon



class Model(AbstactModel):

    name = "DEV"

    def __init__(self, CTX:dict) -> None:
        # load context
        self.CTX = CTX

    def predict(self, x:"list[np.float64_2d[ax.time, ax.feature]]", t:"list[int]") \
            -> np.float64_2d[ax.sample, ax.feature]:

        preds = np.zeros((len(x), 2))
        for i in range(len(x)):
            preds[i] = forecast(x[i], t[i])
        return preds


    def compute_loss(self) -> None:
        return None




    def training_step(self) -> None:
        return None



    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """
        pass


    def get_variables(self):
        """
        Return the variables of the model
        """
        return 0


    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        pass
