
This Multi Object Tracker implementation is primarily based on SORT, with some
modifications to the state vector to adapt for bounding boxes with different
orientations.

A simple online and realtime tracking (SORT) algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4).

By Alex Bewley  
[DynamicDetection.com](http://www.dynamicdetection.com)

you can read more about this in [README_SORT](README_SORT.md)

### Dependencies:

0. [`scikit-learn`](http://scikit-learn.org/stable/)
0. [`scikit-image`](http://scikit-image.org/download)
0. [`FilterPy`](https://github.com/rlabbe/filterpy)
```
$ pip search filterpy
```

( complete list of requirements in `requirements.txt`, install using
    `$pip install -r requirements.txt `)


### Demo:

To run the tracker with the simulated detections:

```
$ cd path/to/sort
$ python test.py --display  

# threshold can be set using '-t' flag

```



To run tracker on detections from original paper, and display the results you need to:


0. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```
