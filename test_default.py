"""Multi object tracking test."""
from __future__ import print_function

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse

from collections import defaultdict, deque
from functools import partial

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


def default_simulator():
    # all train
    sequences = [
        'PETS09-S2L1',
        'TUD-Campus',
        'TUD-Stadtmitte',
        'ETH-Bahnhof',
        'ETH-Sunnyday',
        'ETH-Pedcross2',
        'KITTI-13',
        'KITTI-17',
        'ADL-Rundle-6',
        'ADL-Rundle-8',
        'Venice-2']
    args = parse_args()
    display = args.display
    show_false_positives = args.show_false_positives
    distance_threshold = args.distance_threshold
    if show_false_positives:
        distance_threshold = 0.

    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')

    for seq in sequences:
        # create instance of the SORT tracker
        mot_tracker = mot.Sort(
            max_age=50,
            min_hits=10,
            distance_threshold=distance_threshold)
        seq_dets = np.loadtxt(
            'data/%s/det.txt' %
            (seq), delimiter=',')  # load detections
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            tracked_tragets = defaultdict(partial(deque, maxlen=10))

            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                # dets[:, 2:4] += dets[:, 0:2]

                # convert to [x1,y1,w,h] to [x,y,w,h]
                dets[:, 0:2] += dets[:, 2:4] / 2.

                phi = 0
                dets = np.insert(dets, 4, phi, axis=1)  # .astype(np.float64)

                total_frames += 1
                # print(dets)

                if(display):
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (
                        phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                try:
                    start_time = time.time()
                    trackers = mot_tracker.update(dets)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time
                except BaseException:
                    raise

                tracked_ids = []
                for d in trackers:
                    print(
                        '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                        (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)

                    track_id = d[5]
                    tracked_ids.append(track_id)
                    tracked_tragets[track_id].append(d)

                    if(display):
                        d = d.astype(np.int32)
                        # warnings.warn(str(track_id % 32))
                        x, y, w, h = d[0], d[1], d[2], d[3]
                        ax1.add_patch(patches.Rectangle(
                            (x, y), w, h, fill=False, lw=3, ec=colours[int(track_id % 32), :]))
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
                            track_id = d[5]
                            # ax1.add_patch(patches.Rectangle(
                            #     (d[0], d[1]), d[2] , d[3] , fill=False, lw=3, ec=colours[track_id % 32, :]))
                            # ax1.set_adjustable('box-forced')

                            ax1.add_patch(patches.Rectangle(
                                (d[0], d[1]), 1, 1, fill=False, lw=3, ec=colours[track_id % 32, :]))
                            ax1.set_adjustable('box-forced')

                            # warnings.warn(colours[track_id % 32, :])
                            # ax1.plot(d[0], d[1], colours[track_id % 32, :])

                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))
    if(display):
        print("Note: to get real runtime results run without the option: --display")


if __name__ == "__main__":
    default_simulator()
