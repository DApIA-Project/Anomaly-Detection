import matplotlib.pyplot as plt
from numpy_typing import np, ax

from B_Model.AbstractModel import Model as AbstactModel

from   _Utils.os_wrapper import os
import _Utils.geographic_maths as GEO
import _Utils.Color as C
from   _Utils.Color import prntC

import pandas as pd


class Model(AbstactModel):

    name = "Math"

    def __init__(self, CTX:dict):
        # load context
        self.CTX = CTX


    def predict(self, x, training=False):
        y = np.zeros((len(x), 2))
        # print(x[0, :])
        
        for i in range(len(x)):
            y[i] = np.array(predict_next_pts(x[i, :, 0], x[i, :, 1], x[i, :, 2], self.CTX["HORIZON"]))
        return y
        
    def loss(self, y_, y):
        return np.mean(np.sqrt((y_[:, 0] - y[:, 0]) ** 2 + (y_[:, 1] - y[:, 1]) ** 2))


    def compute_loss(self, x, y, taining=False):
        y_ = self.predict(x, training=taining)
        return self.loss(y_, y), y_
    

    def training_step(self, x, y):
        prntC(C.INFO, "No need to train the model")
        return 0, y


    def visualize(self, save_path="./_Artifacts/"):
        pass


    def get_variables(self):
        return []
    
    
    def set_variables(self, variables):
        pass



debug = False
def predict_next_pts(x, y, t, h):
    t0 = t[0]
    t, t0 = t - t0, 0
    
    if (len(x) < 2):
        return x[-1], y[-1]
    
    if (len(x) == 2):
        vx = x[-1] - x[0]
        vy = y[-1] - y[0]
        return x[-1] + vx, y[-1] + vy
    
    d = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        # d[i] = np.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2)
        d = GEO.distance(x[i], y[i], x[i+1], y[i+1])
    d = np.mean(d)

    a = np.zeros(len(x)-1)
    a_x = np.zeros(len(x))
    for i in range(len(x)-1):
        vx = x[i+1] - x[i]
        vy = y[i+1] - y[i]
        if (vx == 0 and vy == 0 and i > 0):
            a[i] = a[i-1]
        else:
            a[i] = GEO.bearing(x[i], y[i], x[i+1], y[i+1])
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
        next_rot = reg_a * (rotx[-1] + h) + reg_b

    next_angle = a[-1] + next_rot / 2.0
    # nx, ny = x[-1] + d * np.cos(next_angle), y[-1] + d * np.sin(next_angle)
    if (np.isnan(next_angle)):
        nx = x[-1]
        ny = y[-1]
    else:
        nx, ny = GEO.predict(x[-1], y[-1], next_angle, d * h)


    if (debug):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].plot(x, y, 'o-')
        ax[0].plot(nx, ny, 'o')
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_title("Predicted next point")

        ax[1].plot(a_x, a, 'o')
        ax[1].plot(a_x[-1]+h, next_angle, 'o')
        ax[1].set_title("Angles")
        
        ax[2].plot(rotx, rots, 'o')
        ax[2].plot(rotx[-1]+h, next_rot, 'o')
        if (len(rots) > 1):
            ax[2].plot(rotx, reg_a * rotx + reg_b)
        ax[2].set_title("Rotations")

        # show the plot
        fig.tight_layout()
        print(plt.get_backend())
        plt.show()
        input("Press Enter to continue...")
        
    return nx, ny


def angle_shift(a1, a2):
    diff = a2 - a1
    # prntD(diff)
    q = np.floor(diff / (2 * np.pi))
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


    