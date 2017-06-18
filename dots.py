
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

from collections import defaultdict, deque
from functools import partial
import warnings
import copy
import time

# Initializing number of dots
N = 5
XC = 5
YC = 5
R = 1
R_SQR = R **2

# Creating dot class
class Dot(object):
    def __init__(self, ind, xmax=10, ymax=10):
        self._index = ind

        self.x = xmax * np.random.random_sample()
        self.y = ymax * np.random.random_sample()

        self._xmax = xmax
        self._ymax = ymax
        self._vel_scale = max(xmax, ymax)

        self.velx = self.generate_new_vel()
        self.vely = self.generate_new_vel()


    def generate_new_vel(self):
        # rand = np.random.random_sample() - 0.5 / 5
        # if rand > 0.5:
        #     return v + rand/ 10.
        # else:
        #     return v - rand/10.
        return (np.random.random_sample() - 0.5) * self._vel_scale /10.

    # def move(self) :
    #     if np.random.random_sample() < 0.95:
    #         self.x += self.velx
    #         self.y += self.vely
    #     else:
    #         self.velx = self.generate_new_vel()
    #         self.vely = self.generate_new_vel()
    #         self.x += self.velx
    #         self.y += self.vely
        # self.check_inside_circle()


class Box(Dot):
    def __init__(self, ind, xmax=10, ymax=10):
        super(Box,self).__init__(ind, xmax=xmax, ymax=ymax)

        self.w = self.generate_len()
        self.h = self.generate_len()
        self.rz = self.generate_rotation()

        self.dot_rz = self.generate_angular_vel()

    def generate_len(self):
        return np.random.choice([2.5, 5, 7.5])

    def generate_rotation(self):
        return np.random.randint(low=0, high=180)

    def generate_angular_vel(self):
        return (np.random.random_sample() - 0.5) * 30

    def _check_bounds(self):
        if self.x > self._xmax or self.x < 0 or self.y > self._ymax or self.y < 0:
            self._reset()

    def _reset(self):

        self.x = self._xmax * np.random.random_sample()
        self.y = self._ymax * np.random.random_sample()
        self.velx = self.generate_new_vel()
        self.vely = self.generate_new_vel()

        self.w = self.generate_len()
        self.h = self.generate_len()
        self.rz = self.generate_rotation()
        self.dot_rz = self.generate_angular_vel()


    def move(self):
        if np.random.random_sample() < 0.99:
            self.x += self.velx
            self.y += self.vely
            self.rz += self.dot_rz
        else:
            self.velx = self.generate_new_vel()
            self.vely = self.generate_new_vel()
            self.dot_rz = self.generate_angular_vel()
            self.x += self.velx
            self.y += self.vely
            self.rz += self.dot_rz

        self._check_bounds()


def simulate_dots():
    # Initializing dots
    dots = [Dot(i) for i in range(N)]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
    d, = ax.plot([dot.x for dot in dots],
                 [dot.y for dot in dots], 'ro', markersize=3)
    # circl# Initializing dots
    dots = [Dot(i) for i in range(N)]
    e = plt.Circle((XC, YC), R, color='b', fill=False)
    # ax.add_artist(circle)


    # animation function.  This is called sequentially
    def animate(i):
        for dot in dots:
            dot.move()
        d.set_data([dot.x for dot in dots],
                   [dot.y for dot in dots])
        return d,

    # call the animator.
    anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)
    # plt.axis('equal')
    plt.show()

def simulate_boxes(display=True):
    # Initializing dots
    dets = [Box(i, xmax=100, ymax=100) for i in range(N)]



    # First set up the figure, the axis, and the plot element we want to animate
    if display:
        plt.ion()
        fig = plt.figure()

    for i in range(100):
        bboxes = []

        if display:
            ax = fig.add_subplot(111, aspect='equal')
            ax = plt.axes(xlim=(0, 100), ylim=(0, 100))

        # insert new box
        if np.random.sample() < .05:
            dets.append(Box(len(dets), xmax=100, ymax=100))

        del_proba = np.random.sample()
        if del_proba < .01 and dets:
            id = np.random.choice(range(len(dets)))
            del dets[id]
            print('deleting {}'.format(id))

        colours = np.random.rand(32, 3)  # used only for display

        for e,d in enumerate(dets):
            d.move()
            x, y, w, h, rz = d.x, d.y, d.w, d.h, d.rz

            bboxes.append([x, y, w, h, rz])

            # if e == 0:
            #     print(e, rz, d.dot_rz)

            if display:
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h, fill=False, lw=3, ec=colours[int(e % 32), :],
                    angle=rz))
                ax.set_adjustable('box-forced')

        if display:
            fig.canvas.flush_events()
            plt.draw()
            ax.cla()


def box_generator():
    # Initializing dots
    dets = [Box(i, xmax=100, ymax=100) for i in range(N)]

    # for i in range(100):
    i = -1
    while True:
        i += 1
        bboxes = []
        # insert new box
        if np.random.sample() < .05:
            dets.append(Box(len(dets), xmax=100, ymax=100))

        del_proba = np.random.sample()
        if del_proba < .02 and dets:
            id = np.random.choice(range(len(dets)))
            del dets[id]

        colours = np.random.rand(32, 3)  # used only for display

        for e,d in enumerate(dets):
            d.move()
            x, y, w, h, rz = d.x, d.y, d.w, d.h, d.rz

            bboxes.append([x, y, w, h, rz, 0])

        time.sleep(.5)

        # add false positives
        for i in range(30):
            if np.random.sample() < .1:
                box = Box(len(dets), xmax=100, ymax=100)
                bboxes.append([box.x, box.y, box.w, box.h, box.rz, 0])

        # random signal dropping
        for i in range(len(copy.copy(bboxes))):
            if np.random.sample() < .1:
                id = np.random.choice(range(len(bboxes)))
                try:
                    del bboxes[id]
                    print('dropped signal {}'.format(id))
                except:
                    pass


        yield np.asarray(bboxes)



if __name__ == '__main__':
    # simulate_dots()

    simulate_boxes()

    # for b in box_generator():
    #     print(b)
