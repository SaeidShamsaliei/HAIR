from cProfile import label
import numpy as np
import time
import glob
import os

from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image

from scipy import ndimage, misc
from urllib3 import Retry


def label_to_rgb(label_2d, unknown_zero_flag=True, most_distant=False):
    """
    Convert a 2d lebel mask to rgb file, each color represents one class type
    most_distant: if set True then colors with most distance will be returned
    """
    label_2d = label_2d.cpu().detach().numpy()
    # to color the segmentation mask
    if label_2d.ndim == 3:
        label_2d = label_2d.argmax(0)

    if unknown_zero_flag:
        palette = np.array([[0, 0, 0],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [0, 128, 0],  # Green    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0, 255], ], dtype=np.uint8)  # Black  000000
    else:
        palette = np.array([[0, 0, 255],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [0, 128, 0],  # Green    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0,   0], ], dtype=np.uint8)  # Black  000000
    if most_distant:
        print('label to rgb most distance mode')
        palette = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
                            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)

    return palette[label_2d.astype(np.uint8)]
