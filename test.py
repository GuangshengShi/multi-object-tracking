"""Multi object tracking test."""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time
import argparse

from collections import defaultdict, deque
from functools import partial


import simulator
import multi_object_tracker as mot


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true')

    parser.add_argument(
        '--show-false-positives',
        dest='show_false_positives',
        help='Display online tracker output (slow) [False]',
        action='store_true')

    parser.add_argument(
        '-t', '--distance-threshold',
        dest='distance_threshold',
        help='Distance Threshold',
        default=0.03,
        type=float)
    args = parser.parse_args()
    return args


def tracker(detections):
    """Tracks detections

    Args:
        detections (:obj:`numpy.array`) : array of detections
            each row of array : [x, y, w, h, rz, score]


    """

    args = parse_args()
    display = args.display
    show_false_positives = args.show_false_positives
    distance_threshold = args.distance_threshold
    if show_false_positives:
        distance_threshold = 0.

    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if(display):
        plt.ion()
        fig = plt.figure()

    # create instance of the SORT tracker
    mot_tracker = mot.Sort(distance_threshold=distance_threshold)
    tracked_tragets = defaultdict(partial(deque, maxlen=5))

    for dets in detections:
        total_frames += 1
        # print(dets.shape)
        # print(dets)

        if(display):
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1 = plt.axes(xlim=(0, 100), ylim=(0, 100))

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        tracked_ids = []
        for d in trackers:
            track_id = d[5]
            tracked_ids.append(track_id)
            tracked_tragets[track_id].append(d)

            if(display):
                d = d.astype(np.int32)
                # warnings.warn(str(track_id % 32))
                x, y, w, h, rz = d[0], d[1], d[2], d[3], d[4]
                ax1.add_patch(patches.Rectangle(
                    (x, y), w, h, fill=False, lw=3, ec=colours[int(track_id % 32), :],
                    angle=rz, label=str(track_id)))
                ax1.set_adjustable('box-forced')
                # ax1.add_patch(patches.Arrow(x, y, dx, dy, width=1.0, **kwargs))

        # Remove id of not tracked anymore
        for id in tracked_tragets.copy():
            if id not in tracked_ids:
                tracked_tragets.pop(id, None)

        if(display):
            for _, ds in tracked_tragets.items():
                for d in ds:
                    d = d.astype(np.int32)
                    x, y, w, h, rz, track_id = d[0], d[1], d[2], d[3], d[4], d[5]

                    ax1.add_patch(patches.Rectangle(
                        (x, y), w, h, fill=False, lw=3, ec=colours[int(track_id % 32), :],
                        angle=rz))
                    ax1.set_adjustable('box-forced')

                    # ax1.add_patch(patches.Rectangle(
                    #     (d[0], d[1]), .01,  .01, fill=False, lw=3,
                    #     ec=colours[track_id % 32, :]))
                    ax1.set_adjustable('box-forced')

                    # warnings.warn(colours[track_id % 32, :])
                    # ax1.plot(d[0], d[1], colours[track_id % 32, :])

        if(display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()






if __name__ == '__main__':
    # default_simulater()

    try:

        tracker(simulator.box_generator())
    except (KeyboardInterrupt, SystemExit):
        exit()
        raise
